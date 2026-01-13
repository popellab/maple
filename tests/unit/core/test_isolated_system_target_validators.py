#!/usr/bin/env python3
"""
Tests for IsolatedSystemTarget model validators.

Tests validators specific to IsolatedSystemTarget:
- t_span validation (positive, t_end > t_start)
- t_unit validation (valid Pint time unit)
- initial conditions validation (each state has exactly one input with initializes_state)
- parameters existence validation (params in submodel_code exist in model)
- submodel integration validation (ODE integrates without error)
- dimensional consistency validation (d(state)/dt has correct dimensions)
"""

import copy
import pytest
from unittest.mock import Mock
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration import IsolatedSystemTarget
from qsp_llm_workflows.core.model_structure import ModelStructure, ModelParameter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def species_units():
    """Minimal species_units for testing validators."""
    return {
        "V_T.C1": {"units": "cell", "description": "Tumor cells"},
        "V_T.CD8": {"units": "cell", "description": "CD8+ T cells"},
    }


@pytest.fixture
def model_structure():
    """ModelStructure with parameters for testing."""
    return ModelStructure(
        parameters=[
            ModelParameter(name="k_T_prolif", value=0.5, units="1/day"),
            ModelParameter(name="K_T_max", value=1e6, units="cell"),
            ModelParameter(name="k_T_death", value=0.1, units="1/day"),
        ],
        species=[],
        compartments=[],
        reactions=[],
    )


@pytest.fixture
def golden_isolated_target_data():
    """Complete valid IsolatedSystemTarget data that passes all validators.

    Uses the new nested submodel structure with:
    - submodel.code: ODE function
    - submodel.state_variables: List of SubmodelStateVariable objects
    - submodel.parameters: Parameter names from full model
    - submodel.observable: SubmodelObservable for computing measurement from state
    """
    return {
        "description": "T cell proliferation in vitro assay",
        "scenario": {
            "description": "In vitro T cell cytotoxicity assay",
            "interventions": [{"intervention_description": "Anti-CD3/CD28 stimulation"}],
        },
        "experimental_context": {
            "species": "human",
            "compartment": "in_vitro",
            "system": "in_vitro.cell_line",
        },
        "study_overview": "T cell cytotoxicity kinetics study",
        "study_design": "Time-course measurement of T cell count",
        "derivation_explanation": "Direct cell count measurements",
        "key_assumptions": [],
        "key_study_limitations": "In vitro conditions may differ from in vivo",
        "calibration_target_estimates": {
            "median": [1.0],
            "iqr": [0.6843],
            "ci95": [[0.3737, 2.7]],
            "units": "dimensionless",
            "inputs": [
                {
                    "name": "initial_T_cells",
                    "value": 1e5,
                    "units": "cell",
                    "description": "Initial T cell count",
                    "source_ref": "smith_2020",
                    "value_table_or_section": "Methods",
                    "value_snippet": "T cells were seeded at 1e5 cells/well",
                    "initializes_state": "T_cells",
                },
            ],
            "distribution_code": (
                "def derive_distribution(inputs, ureg):\n"
                "    import numpy as np\n"
                "    import math\n"
                "    np.random.seed(42)\n"
                "    mu_log = 0.0  # log(1.0) = 0\n"
                "    sigma_log = 0.5\n"
                "    samples = np.random.lognormal(mu_log, sigma_log, 10000)\n"
                "    median_obs = np.array([np.median(samples)]) * ureg.dimensionless\n"
                "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * ureg.dimensionless\n"
                "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
                "    ci95_obs = [[ci95[0] * ureg.dimensionless, ci95[1] * ureg.dimensionless]]\n"
                "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
            ),
        },
        "primary_data_source": {
            "source_tag": "smith_2020",
            "title": "T cell proliferation dynamics",
            "doi": "10.1000/test.2020.001",
            "first_author": "Smith",
            "year": 2020,
        },
        "secondary_data_sources": [],
        # IsolatedSystemTarget specific field: nested submodel structure
        "submodel": {
            "code": (
                "def submodel(t, y, params, inputs):\n"
                "    T = y[0]\n"
                "    k_prolif = params['k_T_prolif']\n"
                "    K_max = params['K_T_max']\n"
                "    return [k_prolif * T * (1 - T / K_max)]"
            ),
            "state_variables": [{"name": "T_cells", "units": "cell"}],
            "parameters": ["k_T_prolif", "K_T_max"],
            "t_span": [0, 3],
            "t_unit": "day",
            "observable": {
                "code": (
                    "def compute_observable(t, y, constants, ureg):\n"
                    "    T_cells = y['T_cells']\n"
                    "    # Normalize to initial condition (dimensionless fold-change)\n"
                    "    return (T_cells / 1e5) * ureg.dimensionless"
                ),
                "units": "dimensionless",
                "constants": [],
            },
            "rationale": (
                "Simple logistic growth model captures T cell expansion dynamics. "
                "Ignores death term for short-term assay."
            ),
        },
    }


@pytest.fixture
def mock_crossref_success(monkeypatch):
    """Mock successful CrossRef DOI resolution."""

    def mock_get(url, headers=None, timeout=None):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": ["T cell proliferation dynamics"],
            "author": [{"family": "Smith"}],
            "issued": {"date-parts": [[2020]]},
        }
        return mock_response

    monkeypatch.setattr("requests.get", mock_get)


# ============================================================================
# t_span Validation Tests
# ============================================================================


class TestTSpanValidation:
    """Tests for t_span validator."""

    def test_valid_t_span_passes(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid t_span passes validation."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target.submodel.t_span == [0, 3]

    def test_negative_t_start_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that negative t_start fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["t_span"] = [-1, 3]

        with pytest.raises(ValidationError, match="non-negative"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_t_end_not_greater_than_t_start_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that t_end <= t_start fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["t_span"] = [3, 3]  # Equal

        with pytest.raises(ValidationError, match="must be >"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

        data["submodel"]["t_span"] = [5, 3]  # t_end < t_start
        with pytest.raises(ValidationError, match="must be >"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


# ============================================================================
# t_unit Validation Tests
# ============================================================================


class TestTUnitValidation:
    """Tests for t_unit validator."""

    def test_valid_time_units_pass(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid time units pass validation."""
        valid_units = ["day", "hour", "minute", "second", "week"]

        for unit in valid_units:
            data = copy.deepcopy(golden_isolated_target_data)
            data["submodel"]["t_unit"] = unit

            target = IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )
            assert target.submodel.t_unit == unit

    def test_non_time_unit_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that non-time units fail validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["t_unit"] = "meter"

        with pytest.raises(ValidationError, match="time dimensionality"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_invalid_pint_unit_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that invalid Pint unit string fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["t_unit"] = "not_a_unit"

        with pytest.raises(ValidationError, match="not a valid Pint unit"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


# ============================================================================
# Initial Conditions Validation Tests
# ============================================================================


class TestInitialConditionsValidation:
    """Tests for initial conditions validator."""

    def test_valid_initial_conditions_pass(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid initial conditions pass validation."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target is not None
        assert len(target.submodel.state_variables) == 1
        assert target.submodel.state_variables[0].name == "T_cells"

    def test_missing_initial_condition_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that missing initial condition for a state variable fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Remove initializes_state from input
        data["calibration_target_estimates"]["inputs"][0]["initializes_state"] = None

        with pytest.raises(ValidationError, match="missing initial conditions"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_duplicate_initial_condition_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that multiple inputs initializing same state variable fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Add second input that also initializes T_cells
        data["calibration_target_estimates"]["inputs"].append(
            {
                "name": "another_initial",
                "value": 50.0,
                "units": "cell",
                "description": "Another initial condition",
                "source_ref": "modeling_assumption",
                "initializes_state": "T_cells",  # Duplicate!
            }
        )

        with pytest.raises(ValidationError, match="multiple initial conditions"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_invalid_state_variable_reference_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that initializes_state pointing to non-existent state fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["calibration_target_estimates"]["inputs"][0]["initializes_state"] = "NonexistentState"

        with pytest.raises(ValidationError, match="not a valid state variable"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_multiple_state_variables_each_need_initial_condition(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that multi-state system needs initial condition for each state."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Add second state variable
        data["submodel"]["state_variables"] = [
            {"name": "T_cells", "units": "cell"},
            {"name": "Tumor", "units": "cell"},
        ]
        # Update submodel to have 2 states
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T, C = y\n"
            "    k_prolif = params['k_T_prolif']\n"
            "    K_max = params['K_T_max']\n"
            "    return [\n"
            "        k_prolif * T * (1 - T / K_max),\n"
            "        0.1 * C  # Simple tumor growth\n"
            "    ]"
        )
        # Only have initial condition for T_cells, missing Tumor

        with pytest.raises(ValidationError, match="missing initial conditions.*Tumor"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


# ============================================================================
# Parameter Existence Validation Tests
# ============================================================================


class TestParameterExistenceValidation:
    """Tests for parameter existence validator."""

    def test_valid_parameters_pass(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid parameters pass validation."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target is not None

    def test_unknown_parameter_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that using unknown parameter fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Add unknown parameter to the submodel.parameters list
        data["submodel"]["parameters"] = ["k_T_prolif", "K_T_max", "k_unknown_param"]
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T = y[0]\n"
            "    k_unknown = params['k_unknown_param']\n"
            "    return [k_unknown * T]"
        )

        with pytest.raises(ValidationError, match="Unknown parameters.*k_unknown_param"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_model_structure_required_in_context(
        self, species_units, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that missing model_structure in context raises error."""
        with pytest.raises(ValidationError, match="model_structure is required"):
            IsolatedSystemTarget.model_validate(
                golden_isolated_target_data,
                context={"species_units": species_units},  # Missing model_structure
            )


# ============================================================================
# Submodel Integration Validation Tests
# ============================================================================


class TestSubmodelIntegrationValidation:
    """Tests for submodel integration validator."""

    def test_valid_submodel_integrates(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid submodel integrates successfully."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target is not None

    def test_submodel_with_syntax_error_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that submodel with syntax error fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T = y[0]\n"
            "    return [T *"  # Syntax error
        )

        with pytest.raises(ValidationError, match="syntax error"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_submodel_wrong_return_length_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that submodel returning wrong number of derivatives fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Return 2 derivatives for 1 state variable
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T = y[0]\n"
            "    k_prolif = params['k_T_prolif']\n"
            "    return [k_prolif * T, k_prolif * T]"  # Wrong length!
        )

        with pytest.raises(ValidationError, match="state variables"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_submodel_producing_nan_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that submodel producing NaN during integration fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Create submodel that will produce NaN (division by zero at initial value)
        # Initial value is 1e5, so 1.0 / (T - 1e5) will divide by zero immediately
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T = y[0]\n"
            "    # Division by zero at initial condition (T=1e5)\n"
            "    return [1.0 / (T - 1e5)]"
        )

        with pytest.raises(ValidationError, match="NaN|Inf|integration|division"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_submodel_with_exponential_blowup_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that submodel with exponential blowup fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Create submodel that will blow up exponentially
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T = y[0]\n"
            "    return [100 * T]"  # Very fast exponential growth
        )
        data["submodel"]["t_span"] = [0, 100]  # Long enough to blow up

        with pytest.raises(ValidationError, match="NaN|Inf|integration"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


# ============================================================================
# Dimensional Consistency Validation Tests
# ============================================================================


class TestDimensionalConsistencyValidation:
    """Tests for dimensional consistency validator."""

    def test_dimensionally_consistent_submodel_passes(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that dimensionally consistent submodel passes."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target is not None

    def test_dimensional_mismatch_raises_warning_or_error(
        self, species_units, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that dimensional mismatch is detected.

        Depending on how the submodel handles Pint quantities, this may either:
        - Warn about "Could not perform dimensional analysis" (if execution fails)
        - Raise DimensionalityMismatchError (if execution succeeds but dims mismatch)
        """
        from qsp_llm_workflows.core.calibration.exceptions import DimensionalityMismatchError

        # Model structure with wrong units for k_T_prolif
        bad_model_structure = ModelStructure(
            parameters=[
                ModelParameter(name="k_T_prolif", value=0.5, units="meter"),  # Wrong!
                ModelParameter(name="K_T_max", value=1e6, units="cell"),
            ],
            species=[],
            compartments=[],
            reactions=[],
        )

        # May raise DimensionalityMismatchError or ValidationError
        with pytest.raises((ValidationError, DimensionalityMismatchError)):
            IsolatedSystemTarget.model_validate(
                golden_isolated_target_data,
                context={"species_units": species_units, "model_structure": bad_model_structure},
            )


# ============================================================================
# Submodel Code Validation Tests
# ============================================================================


class TestSubmodelCodeValidation:
    """Tests for submodel.code structure validator."""

    def test_valid_submodel_signature_passes(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid submodel signature passes."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target is not None

    def test_wrong_function_name_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that wrong function name fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["code"] = (
            "def my_model(t, y, params, inputs):\n"  # Wrong name!
            "    return [0.5 * y[0]]"
        )

        with pytest.raises(ValidationError, match="named 'submodel'"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_wrong_signature_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that wrong function signature fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["code"] = (
            "def submodel(t, y):\n"  # Missing params, inputs
            "    return [0.5 * y[0]]"
        )

        with pytest.raises(ValidationError, match="signature must be"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


# ============================================================================
# State Variables Validation Tests
# ============================================================================


class TestStateVariablesValidation:
    """Tests for submodel.state_variables validator."""

    def test_valid_state_variables_pass(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that valid state variables pass validation."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert len(target.submodel.state_variables) == 1
        assert target.submodel.state_variables[0].name == "T_cells"
        assert target.submodel.state_variables[0].units == "cell"

    def test_empty_state_variables_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that empty state_variables fails validation."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["state_variables"] = []

        with pytest.raises(ValidationError, match="[Aa]t least one state variable"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


# ============================================================================
# get_parameters_used Method Tests
# ============================================================================


class TestGetParametersUsed:
    """Tests for get_parameters_used method."""

    def test_returns_parameters_from_submodel(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that parameters from submodel.parameters are returned."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )

        params = target.get_parameters_used()
        assert "k_T_prolif" in params
        assert "K_T_max" in params

    def test_returns_multiple_parameters(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that all submodel.parameters are returned."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["parameters"] = ["k_T_prolif", "K_T_max", "k_T_death"]
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T = y[0]\n"
            "    k_prolif = params['k_T_prolif']\n"
            "    K_max = params['K_T_max']\n"
            "    k_death = params['k_T_death']\n"
            "    return [k_prolif * T * (1 - T / K_max) - k_death * T]"
        )

        target = IsolatedSystemTarget.model_validate(
            data,
            context={"species_units": species_units, "model_structure": model_structure},
        )

        params = target.get_parameters_used()
        assert sorted(params) == ["K_T_max", "k_T_death", "k_T_prolif"]
