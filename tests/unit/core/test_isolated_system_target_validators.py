#!/usr/bin/env python3
"""
Tests for IsolatedSystemTarget model validators.

Tests validators specific to IsolatedSystemTarget:
- t_span validation (positive, t_end > t_start)
- t_unit validation (valid Pint time unit)
- state variable validation (self-contained with initial value and provenance)
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
        "rationale": "Logistic growth model captures T cell expansion; ignores death for short-term assay.",
        "calibration_target_estimates": {
            "median": [1.0],
            "ci95": [[0.3737, 2.7]],
            "units": "dimensionless",
            "sample_size": 3,
            "sample_size_rationale": "n=3 replicates per condition, standard for in vitro T cell assays",
            "inputs": [
                {
                    "name": "initial_T_cells",
                    "value": 1e5,
                    "units": "cell",
                    "description": "Initial T cell count",
                    "source_ref": "smith_2020",
                    "value_location": "Methods",
                    "value_snippet": "T cells were seeded at 1e5 cells/well",
                    "input_type": "experimental_condition",
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
            "inputs": [],  # Experimental conditions for ODE (empty for this simple example)
            "state_variables": [
                {
                    "name": "T_cells",
                    "units": "cell",
                    "initial_value": 1e5,
                    "source_ref": "smith_2020",
                    "value_location": "Methods",
                    "value_snippet": "T cells were seeded at 1e5 cells/well",
                }
            ],
            "parameters": ["k_T_prolif", "K_T_max"],
            "t_span": [0, 3],
            "t_unit": "day",
            "observable": {
                "code": (
                    "def compute_observable(t, y, constants, ureg):\n"
                    "    T_cells = y[0]  # same index as in ODE\n"
                    "    # Normalize to initial condition (dimensionless fold-change)\n"
                    "    return (T_cells / 1e5) * ureg.dimensionless"
                ),
                "units": "dimensionless",
                "constants": [],
            },
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
    """Tests for initial conditions and state variable validation."""

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
        assert target.submodel.state_variables[0].initial_value == 1e5

    def test_invalid_source_ref_in_state_variable_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that state variable with invalid source_ref fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Set source_ref to a non-existent source tag
        data["submodel"]["state_variables"][0]["source_ref"] = "nonexistent_source"

        with pytest.raises(ValidationError, match="not a valid source tag"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_multiple_state_variables_with_self_contained_values(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that multi-state system works with self-contained state variables."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Add second state variable with its own initial value (self-contained)
        data["submodel"]["state_variables"] = [
            {
                "name": "T_cells",
                "units": "cell",
                "initial_value": 1e5,
                "source_ref": "smith_2020",
                "value_location": "Methods",
                "value_snippet": "T cells were seeded at 1e5 cells/well",
            },
            {
                "name": "Tumor",
                "units": "cell",
                "initial_value": 1e4,
                "source_ref": "smith_2020",
                "value_location": "Methods",
                "value_snippet": "Tumor cells seeded at 1e4",
            },
        ]
        # Update submodel to have 2 states (use k_T_death for tumor growth to keep params valid)
        data["submodel"]["code"] = (
            "def submodel(t, y, params, inputs):\n"
            "    T, C = y\n"
            "    k_prolif = params['k_T_prolif']\n"
            "    K_max = params['K_T_max']\n"
            "    k_tumor = params['k_T_death']  # Reuse death param as tumor growth rate\n"
            "    return [\n"
            "        k_prolif * T * (1 - T / K_max),\n"
            "        k_tumor * C  # Simple tumor growth\n"
            "    ]"
        )
        data["submodel"]["parameters"].append("k_T_death")
        # This should pass because both state variables are self-contained
        target = IsolatedSystemTarget.model_validate(
            data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert len(target.submodel.state_variables) == 2
        assert target.submodel.state_variables[0].initial_value == 1e5
        assert target.submodel.state_variables[1].initial_value == 1e4

    def test_missing_initial_value_field_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that state variable without initial_value field fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Remove initial_value from state variable (field is required)
        del data["submodel"]["state_variables"][0]["initial_value"]

        with pytest.raises(ValidationError, match="initial_value"):
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

        with pytest.raises(ValidationError, match="(?i)syntax error"):
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

        with pytest.raises(ValidationError, match="wrong signature"):
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


class TestHardcodedConstantsValidation:
    """Tests for hardcoded constants validator."""

    def test_observable_code_with_hardcoded_constant_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that hardcoded constants in observable code are flagged."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Add hardcoded constant in observable code
        data["submodel"]["observable"]["code"] = (
            "def compute_observable(t, y, constants, ureg):\n"
            "    T_cells = y[0]\n"
            "    # Hardcoded constant with units - BAD!\n"
            "    cell_volume = 1766.0 * ureg.micrometer**3\n"
            "    return (T_cells * cell_volume).to('micrometer**3')"
        )
        data["submodel"]["observable"]["units"] = "micrometer**3"
        data["calibration_target_estimates"]["units"] = "micrometer**3"
        # Update distribution to match units
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    samples = np.random.lognormal(10, 0.5, 10000)\n"
            "    median_obs = np.array([np.median(samples)]) * ureg.micrometer**3\n"
            "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * ureg.micrometer**3\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * ureg.micrometer**3, ci95[1] * ureg.micrometer**3]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )
        # Exact values from lognormal(10, 0.5) with seed 42
        data["calibration_target_estimates"]["median"] = [21997.91]
        data["calibration_target_estimates"]["iqr"] = [15072.36]
        data["calibration_target_estimates"]["ci95"] = [[8231.66, 59104.89]]

        with pytest.raises(ValidationError, match="Hardcoded numeric constants"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_observable_code_with_constants_dict_passes(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that constants from constants dict are allowed."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Use constants dict properly
        data["submodel"]["observable"]["code"] = (
            "def compute_observable(t, y, constants, ureg):\n"
            "    T_cells = y[0]\n"
            "    cell_volume = constants['cell_volume']  # Properly declared\n"
            "    return (T_cells * cell_volume).to('micrometer**3')"
        )
        data["submodel"]["observable"]["units"] = "micrometer**3"
        data["submodel"]["observable"]["constants"] = [
            {
                "name": "cell_volume",
                "value": 1766.0,
                "units": "micrometer**3",
                "biological_basis": "PDAC cell ~15 μm diameter",
                "source_ref": "modeling_assumption",
            }
        ]
        data["calibration_target_estimates"]["units"] = "micrometer**3"
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    samples = np.random.lognormal(18, 0.5, 10000)\n"
            "    median_obs = np.array([np.median(samples)]) * ureg.micrometer**3\n"
            "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * ureg.micrometer**3\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * ureg.micrometer**3, ci95[1] * ureg.micrometer**3]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )
        # Exact values from lognormal(18, 0.5) with seed 42
        data["calibration_target_estimates"]["median"] = [6.56e7]
        data["calibration_target_estimates"]["iqr"] = [4.49e7]
        data["calibration_target_estimates"]["ci95"] = [[2.45e7, 1.76e8]]

        # Should pass - constants are properly declared
        target = IsolatedSystemTarget.model_validate(
            data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target is not None

    def test_default_observable_skips_hardcoded_check(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that default observable (no code) doesn't trigger hardcoded check."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Use default observable (no code)
        data["submodel"]["observable"]["code"] = None
        data["submodel"]["observable"]["units"] = "cell"
        data["calibration_target_estimates"]["units"] = "cell"
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    samples = np.random.lognormal(11.5, 0.5, 10000)\n"
            "    median_obs = np.array([np.median(samples)]) * ureg.cell\n"
            "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * ureg.cell\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * ureg.cell, ci95[1] * ureg.cell]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )
        data["calibration_target_estimates"]["median"] = [98587.77]
        data["calibration_target_estimates"]["iqr"] = [67549.62]
        data["calibration_target_estimates"]["ci95"] = [[36891.73, 264889.72]]

        # Should pass
        target = IsolatedSystemTarget.model_validate(
            data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target.submodel.observable.code is None


class TestObservableCodeValidation:
    """Tests for observable code validator."""

    def test_default_observable_when_code_omitted(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that observable code can be omitted and defaults to y[0] * units."""
        data = copy.deepcopy(golden_isolated_target_data)
        # Remove observable code - should default to y[0] * ureg(units)
        data["submodel"]["observable"]["code"] = None
        # State is in cells, observable should return cells
        data["submodel"]["observable"]["units"] = "cell"
        data["calibration_target_estimates"]["units"] = "cell"
        # Update distribution_code to return cell units
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    samples = np.random.lognormal(11.5, 0.5, 10000)  # Around 1e5 cells\n"
            "    median_obs = np.array([np.median(samples)]) * ureg.cell\n"
            "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * ureg.cell\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * ureg.cell, ci95[1] * ureg.cell]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )
        # Update median/iqr/ci95 values to match the distribution output
        # lognormal(11.5, 0.5) with seed 42 produces these exact values
        data["calibration_target_estimates"]["median"] = [98587.77]
        data["calibration_target_estimates"]["iqr"] = [67549.62]
        data["calibration_target_estimates"]["ci95"] = [[36891.73, 264889.72]]

        target = IsolatedSystemTarget.model_validate(
            data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target.submodel.observable.code is None
        assert target.submodel.observable.units == "cell"

    def test_custom_observable_code_works(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that custom observable code is validated and executed."""
        target = IsolatedSystemTarget.model_validate(
            golden_isolated_target_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target.submodel.observable.code is not None

    def test_observable_wrong_signature_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that observable with wrong signature fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["observable"]["code"] = (
            "def compute_observable(t, y):\n"  # Missing constants, ureg
            "    return y[0]"
        )

        with pytest.raises(ValidationError, match="wrong signature"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_observable_wrong_function_name_fails(
        self, species_units, model_structure, golden_isolated_target_data, mock_crossref_success
    ):
        """Test that observable with wrong function name fails."""
        data = copy.deepcopy(golden_isolated_target_data)
        data["submodel"]["observable"]["code"] = (
            "def get_observable(t, y, constants, ureg):\n"  # Wrong name
            "    return y[0] * ureg.dimensionless"
        )

        with pytest.raises(ValidationError, match="compute_observable"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )


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

    def test_returns_empty_for_direct_conversion_mode(
        self, species_units, model_structure, mock_crossref_success
    ):
        """Test that get_parameters_used returns empty list when submodel is None."""
        # Direct conversion mode data (no submodel)
        # CI values computed from: np.random.seed(42); samples = np.random.normal(8, 1, 10000);
        # samples = np.maximum(samples, 1.0); k_samples = np.log(2) / samples * 24
        # median = 2.078, ci95 = [1.668, 2.614]
        data = {
            "description": "Direct conversion of doubling time to proliferation rate",
            "scenario": {
                "description": "T cell doubling time measurement",
                "interventions": [],
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro.cell_line",
            },
            "study_overview": "T cell kinetics study",
            "study_design": "Doubling time measurement",
            "derivation_explanation": "Direct conversion: k = ln(2) / t_double",
            "key_assumptions": [],
            "key_study_limitations": "In vitro conditions",
            "rationale": "Simple analytical conversion from doubling time to rate.",
            "calibration_target_estimates": {
                "median": [2.0801],
                "ci95": [[1.6679, 2.7581]],
                "units": "1/day",
                "sample_size": 5,
                "sample_size_rationale": "n=5 replicates in doubling time experiment",
                "inputs": [
                    {
                        "name": "doubling_time",
                        "value": 8.0,
                        "units": "hour",
                        "description": "T cell doubling time",
                        "source_ref": "smith_2020",
                        "value_location": "Figure 2",
                        "value_snippet": "cells doubled every 8 hours",
                        "input_type": "proxy_measurement",
                        "conversion_formula": "k = ln(2) / doubling_time",
                    },
                ],
                "distribution_code": (
                    "def derive_distribution(inputs, ureg):\n"
                    "    import numpy as np\n"
                    "    np.random.seed(42)\n"
                    "    t_double = inputs['doubling_time'].magnitude\n"
                    "    t_sd = 1.0  # Assumed 1h SD\n"
                    "    samples = np.random.normal(t_double, t_sd, 10000)\n"
                    "    samples = np.maximum(samples, 1.0)\n"
                    "    k_samples = np.log(2) / samples * 24  # Convert to per day\n"
                    "    median_obs = np.array([np.median(k_samples)]) * ureg('1/day')\n"
                    "    ci95 = np.percentile(k_samples, [2.5, 97.5])\n"
                    "    ci95_obs = [[ci95[0] * ureg('1/day'), ci95[1] * ureg('1/day')]]\n"
                    "    return {'median_obs': median_obs, 'ci95_obs': ci95_obs}"
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
            "submodel": None,  # Direct conversion mode
        }

        target = IsolatedSystemTarget.model_validate(
            data,
            context={"species_units": species_units, "model_structure": model_structure},
        )

        assert target.submodel is None
        assert target.get_parameters_used() == []


# ============================================================================
# Direct Conversion Mode Tests
# ============================================================================


class TestDirectConversionMode:
    """Tests for direct conversion mode (submodel=None)."""

    @pytest.fixture
    def direct_conversion_data(self):
        """Valid direct conversion mode data.

        CI values computed from: np.random.seed(42); samples = np.random.normal(8, 1, 10000);
        samples = np.maximum(samples, 1.0); k_samples = np.log(2) / samples * 24
        median = 2.078, ci95 = [1.668, 2.614]
        """
        return {
            "description": "Direct conversion of doubling time to proliferation rate",
            "scenario": {
                "description": "T cell doubling time measurement",
                "interventions": [],
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro.cell_line",
            },
            "study_overview": "T cell kinetics study",
            "study_design": "Doubling time measurement",
            "derivation_explanation": "Direct conversion: k = ln(2) / t_double",
            "key_assumptions": [],
            "key_study_limitations": "In vitro conditions",
            "rationale": "Simple analytical conversion from doubling time to rate.",
            "calibration_target_estimates": {
                "median": [2.0801],
                "ci95": [[1.6679, 2.7581]],
                "units": "1/day",
                "sample_size": 5,
                "sample_size_rationale": "n=5 replicates in doubling time experiment",
                "inputs": [
                    {
                        "name": "doubling_time",
                        "value": 8.0,
                        "units": "hour",
                        "description": "T cell doubling time",
                        "source_ref": "smith_2020",
                        "value_location": "Figure 2",
                        "value_snippet": "cells doubled every 8 hours",
                        "input_type": "proxy_measurement",
                        "conversion_formula": "k = ln(2) / doubling_time",
                    },
                ],
                "distribution_code": (
                    "def derive_distribution(inputs, ureg):\n"
                    "    import numpy as np\n"
                    "    np.random.seed(42)\n"
                    "    t_double = inputs['doubling_time'].magnitude\n"
                    "    t_sd = 1.0\n"
                    "    samples = np.random.normal(t_double, t_sd, 10000)\n"
                    "    samples = np.maximum(samples, 1.0)\n"
                    "    k_samples = np.log(2) / samples * 24\n"
                    "    median_obs = np.array([np.median(k_samples)]) * ureg('1/day')\n"
                    "    ci95 = np.percentile(k_samples, [2.5, 97.5])\n"
                    "    ci95_obs = [[ci95[0] * ureg('1/day'), ci95[1] * ureg('1/day')]]\n"
                    "    return {'median_obs': median_obs, 'ci95_obs': ci95_obs}"
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
            "submodel": None,
        }

    def test_direct_conversion_mode_valid(
        self, species_units, model_structure, direct_conversion_data, mock_crossref_success
    ):
        """Test that valid direct conversion mode passes validation."""
        target = IsolatedSystemTarget.model_validate(
            direct_conversion_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )

        assert target.submodel is None
        assert target.get_parameters_used() == []

    def test_direct_conversion_no_submodel_inputs_needed(
        self, species_units, model_structure, direct_conversion_data, mock_crossref_success
    ):
        """Test that direct conversion mode works without submodel.inputs (no submodel)."""
        # Direct conversion mode has no submodel, so no submodel.inputs needed
        target = IsolatedSystemTarget.model_validate(
            direct_conversion_data,
            context={"species_units": species_units, "model_structure": model_structure},
        )
        assert target.submodel is None
        # All inputs are in calibration_target_estimates.inputs
        assert len(target.calibration_target_estimates.inputs) > 0


# ============================================================================
# Cancer Fields Validation Tests
# ============================================================================


class TestCancerFieldsValidation:
    """Tests for cancer fields validation with non-cancer data."""

    def test_cancer_stage_with_other_disease_warns(
        self, species_units, model_structure, mock_crossref_success
    ):
        """Test that cancer stage fields with other_disease indication triggers warning."""
        # Use same title as mock_crossref_success fixture returns
        data = {
            "description": "Viral infection data used for cancer model",
            "scenario": {
                "description": "LCMV infection in mice",
                "interventions": [],
            },
            "experimental_context": {
                "species": "mouse",
                "system": "animal_in_vivo.syngeneic",
                "indication": "other_disease",  # Not cancer
                "stage": {
                    "extent": "metastatic",  # Doesn't make sense for viral infection
                    "burden": "high",
                },
            },
            "study_overview": "T cell kinetics in viral infection",
            "study_design": "Time-course measurement",
            "derivation_explanation": "Direct conversion",
            "key_assumptions": [],
            "key_study_limitations": "Mouse model",
            "rationale": "Using viral infection data as proxy.",
            "calibration_target_estimates": {
                "median": [2.079],  # ln(2) / (8/24) = 2.079 per day
                "ci95": [[1.559, 2.599]],  # +/- 25%
                "units": "1/day",
                "sample_size": 8,
                "sample_size_rationale": "n=8 mice in viral infection cohort",
                "inputs": [
                    {
                        "name": "doubling_time",
                        "value": 8.0,
                        "units": "hour",
                        "description": "T cell doubling time",
                        "source_ref": "smith_2020",
                        "value_location": "Abstract",
                        "value_snippet": "doubling time 8h",
                        "input_type": "proxy_measurement",
                        "conversion_formula": "k = ln(2) / t",
                    },
                ],
                "distribution_code": (
                    "def derive_distribution(inputs, ureg):\n"
                    "    import numpy as np\n"
                    "    # k = ln(2) / doubling_time\n"
                    "    doubling_time = inputs['doubling_time']\n"
                    "    k = np.log(2) / doubling_time\n"
                    "    median_obs = np.array([k.to('1/day').magnitude]) * ureg('1/day')\n"
                    "    ci95_low = (k * 0.75).to('1/day').magnitude\n"
                    "    ci95_high = (k * 1.25).to('1/day').magnitude\n"
                    "    ci95_obs = [[ci95_low * ureg('1/day'), ci95_high * ureg('1/day')]]\n"
                    "    return {'median_obs': median_obs, 'ci95_obs': ci95_obs}"
                ),
            },
            "primary_data_source": {
                "source_tag": "smith_2020",
                "title": "T cell proliferation dynamics",  # Must match mock
                "doi": "10.1000/test.2020.001",
                "first_author": "Smith",
                "year": 2020,
            },
            "secondary_data_sources": [],
            "submodel": None,
        }

        # Should warn about cancer staging with non-cancer indication
        with pytest.warns(UserWarning, match="Cancer staging.*doesn't apply"):
            IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )

    def test_no_warning_when_stage_null_for_other_disease(
        self, species_units, model_structure, mock_crossref_success
    ):
        """Test that no warning when stage is null for other_disease."""
        # Use same title as mock_crossref_success fixture returns
        data = {
            "description": "Viral infection data used for cancer model",
            "scenario": {
                "description": "LCMV infection in mice",
                "interventions": [],
            },
            "experimental_context": {
                "species": "mouse",
                "system": "animal_in_vivo.syngeneic",
                "indication": "other_disease",
                "stage": None,  # Correctly set to null
            },
            "study_overview": "T cell kinetics in viral infection",
            "study_design": "Time-course measurement",
            "derivation_explanation": "Direct conversion",
            "key_assumptions": [],
            "key_study_limitations": "Mouse model",
            "rationale": "Using viral infection data as proxy.",
            "calibration_target_estimates": {
                "median": [2.079],  # ln(2) / (8/24) = 2.079 per day
                "ci95": [[1.559, 2.599]],  # +/- 25%
                "units": "1/day",
                "sample_size": 8,
                "sample_size_rationale": "n=8 mice in viral infection cohort",
                "inputs": [
                    {
                        "name": "doubling_time",
                        "value": 8.0,
                        "units": "hour",
                        "description": "T cell doubling time",
                        "source_ref": "smith_2020",
                        "value_location": "Abstract",
                        "value_snippet": "doubling time 8h",
                        "input_type": "proxy_measurement",
                        "conversion_formula": "k = ln(2) / t",
                    },
                ],
                "distribution_code": (
                    "def derive_distribution(inputs, ureg):\n"
                    "    import numpy as np\n"
                    "    # k = ln(2) / doubling_time\n"
                    "    doubling_time = inputs['doubling_time']\n"
                    "    k = np.log(2) / doubling_time\n"
                    "    median_obs = np.array([k.to('1/day').magnitude]) * ureg('1/day')\n"
                    "    ci95_low = (k * 0.75).to('1/day').magnitude\n"
                    "    ci95_high = (k * 1.25).to('1/day').magnitude\n"
                    "    ci95_obs = [[ci95_low * ureg('1/day'), ci95_high * ureg('1/day')]]\n"
                    "    return {'median_obs': median_obs, 'ci95_obs': ci95_obs}"
                ),
            },
            "primary_data_source": {
                "source_tag": "smith_2020",
                "title": "T cell proliferation dynamics",  # Must match mock
                "doi": "10.1000/test.2020.001",
                "first_author": "Smith",
                "year": 2020,
            },
            "secondary_data_sources": [],
            "submodel": None,
        }

        # Should NOT warn when stage is properly null
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Filter out unrelated warnings from validators we're not testing
            warnings.filterwarnings("ignore", message=".*IsolatedSystemTarget typically uses.*")
            warnings.filterwarnings(
                "ignore", message=".*inputs/assumptions are defined but not used.*"
            )
            target = IsolatedSystemTarget.model_validate(
                data,
                context={"species_units": species_units, "model_structure": model_structure},
            )
            assert target.experimental_context.stage is None
