#!/usr/bin/env python3
"""
Tests for CalibrationTarget model validators.

Tests all validators with:
- 1 golden test: Valid scalar data passes all validators
- Multiple negative tests: Validators fail on invalid data
  - DOI resolution fails
  - Title mismatch fails
  - Wrong measurement code units fails
  - Scalar measurement code return fails (time series length)
  - Wrong-length array measurement code return fails (time series length)
  - Derivation code value mismatch fails
  - Undefined source reference fails
  - Missing species fails
- Warning tests: Scientific best practices
  - Clipping suggests lognormal distribution
  - Large variance should be documented
  - Normal distribution inappropriate for size data
  - Conversion factors should be documented
  - Unused inputs emit warning

Vector-valued data tests:
- Vector calibration target passes validation (time-course data)
- Vector input length mismatch fails (input length != index_values length)
- Output array length mismatch fails (median/ci95 different lengths)
- Index fields required together (index_values needs index_unit and index_type)
- Scalar data requires length-1 arrays (no index_values → length must be 1)
- CI95 wrong inner structure fails ([lo, mid, hi] instead of [lo, hi])
"""

import copy
import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration import CalibrationTarget


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def species_units():
    """Minimal species_units for testing validators."""
    return {
        "V_T.C1": {"units": "cell", "description": "Tumor cells"},
        "V_T.CD8": {"units": "cell", "description": "CD8+ T cells"},
        "V_T.Treg": {"units": "cell", "description": "Regulatory T cells"},
        "V_T.TGFb": {"units": "nanomolarity", "description": "TGF-beta concentration"},
    }


@pytest.fixture
def golden_calibration_target_data():
    """Complete valid CalibrationTarget data that passes all validators."""
    return {
        "observable": {
            "code": (
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    cd8 = species_dict['V_T.CD8']\n"
                "    tumor = species_dict['V_T.C1']\n"
                "    ratio = cd8 / tumor\n"
                "    return ratio.to(ureg.dimensionless)"
            ),
            "units": "dimensionless",
            "species": ["V_T.CD8", "V_T.C1"],
            "constants": [],
            "support": "positive_unbounded",
            "mapping_rationale": (
                "CD8+ T cell density measured via IHC in tumor tissue sections, "
                "reported as dimensionless ratio (CD8+ cells / tumor cells)"
            ),
        },
        "experimental_context": {
            "species": "human",
            "indication": "PDAC",
            "system": "clinical.resection",
            "treatment": {"history": ["treatment_naive"], "status": "off_treatment"},
            "stage": {"extent": "resectable", "burden": "moderate"},
        },
        "rationale": (
            "CD8+ T cell density in resectable PDAC tumors measured via IHC. "
            "Parametric bootstrap from reported lognormal distribution (mean 1.0, sigma_log 0.5). "
            "Maps to CD8/tumor ratio in model for immune profiling comparison."
        ),
        "caveats": [
            "Single-center study with limited sample size",
            "Lognormal distribution assumed based on positive-only data",
        ],
        "calibration_target_estimates": {
            # Vector-valued outputs (length-1 for scalar data)
            "median": [1.0],
            "ci95": [[0.3737, 2.7]],
            "units": "dimensionless",
            "inputs": [
                {
                    "name": "cd8_ratio_mean",
                    "value": 1.0,
                    "units": "dimensionless",
                    "description": "Mean CD8/tumor ratio from lognormal fit",
                    "source_ref": "smith_2020",
                    "value_location": "Table 2",
                    "value_snippet": "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)",
                },
                {
                    "name": "cd8_ratio_sigma_log",
                    "value": 0.5,
                    "units": "dimensionless",
                    "description": "Log-scale SD of CD8/tumor ratio from lognormal fit",
                    "source_ref": "smith_2020",
                    "value_location": "Table 2",
                    "value_snippet": "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)",
                },
            ],
            "distribution_code": (
                "def derive_distribution(inputs, ureg):\n"
                "    import numpy as np\n"
                "    import math\n"
                "    np.random.seed(42)\n"
                "    mean = inputs['cd8_ratio_mean']\n"
                "    sigma_log = inputs['cd8_ratio_sigma_log']\n"
                "    n = 10000\n"
                "    mu_log = math.log(mean.magnitude)\n"
                "    samples = np.random.lognormal(mu_log, sigma_log.magnitude, n) * mean.units\n"
                "    median_obs = np.array([np.median(samples)]) * mean.units\n"
                "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
                "    ci95_obs = [[ci95[0] * mean.units, ci95[1] * mean.units]]\n"
                "    return {'median_obs': median_obs, 'ci95_obs': ci95_obs}"
            ),
        },
        "primary_data_source": {
            "source_tag": "smith_2020",
            "title": "Immune landscape of pancreatic ductal adenocarcinoma",
            "doi": "10.1000/test.2020.001",
            "first_author": "Smith",
            "year": 2020,
        },
        "secondary_data_sources": [],
    }


@pytest.fixture
def mock_crossref_success(monkeypatch):
    """Mock successful CrossRef DOI resolution."""

    def mock_get(url, headers=None, timeout=None):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": ["Immune landscape of pancreatic ductal adenocarcinoma"],
            "author": [{"family": "Smith"}],
            "issued": {"date-parts": [[2020]]},
        }
        return mock_response

    monkeypatch.setattr("requests.get", mock_get)


@pytest.fixture
def mock_crossref_failure(monkeypatch):
    """Mock failed CrossRef DOI resolution."""

    def mock_get(url, headers=None, timeout=None):
        mock_response = Mock()
        mock_response.status_code = 404
        return mock_response

    monkeypatch.setattr("requests.get", mock_get)


# ============================================================================
# Golden Test - All Validators Pass
# ============================================================================


class TestCalibrationTargetGolden:
    """Test that valid CalibrationTarget passes all validators."""

    def test_golden_yaml_passes_all_validators(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that golden YAML passes all 11 validators (including scale/control char)."""
        target = CalibrationTarget.model_validate(
            golden_calibration_target_data, context={"species_units": species_units}
        )

        assert target is not None
        assert "CD8+ T cell density" in target.rationale
        assert target.observable is not None
        assert target.observable.species == ["V_T.CD8", "V_T.C1"]
        # Scalar data uses length-1 lists
        assert target.calibration_target_estimates.median == [pytest.approx(1.0, rel=0.01)]
        assert len(target.calibration_target_estimates.ci95) == 1
        assert target.calibration_target_estimates.index_values is None  # Scalar case


# ============================================================================
# Negative Tests - Each Validator Fails
# ============================================================================


class TestCalibrationTargetValidators:
    """Tests for individual CalibrationTarget validators."""

    def test_observable_required_for_calibration_target(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """CalibrationTarget must have observable field - it's required for full model targets."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Remove observable field
        del data["observable"]

        with pytest.raises(ValidationError, match="observable"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_doi_resolution_fails_on_invalid_doi(
        self, mock_crossref_failure, species_units, golden_calibration_target_data
    ):
        """Validator should reject DOI that doesn't resolve."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["primary_data_source"]["doi"] = "10.9999/invalid.doi"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "failed to resolve" in error_str

    def test_validate_title_match_fails_on_title_mismatch(
        self, species_units, golden_calibration_target_data
    ):
        """Validator should reject mismatched paper title."""

        # Mock CrossRef to return different title
        def mock_get(url, headers=None, timeout=None):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "title": ["Completely Different Paper Title"],
                "author": [{"family": "Smith"}],
                "issued": {"date-parts": [[2020]]},
            }
            return mock_response

        with patch("requests.get", mock_get):
            data = copy.deepcopy(golden_calibration_target_data)

            with pytest.raises(ValidationError) as exc_info:
                CalibrationTarget.model_validate(data, context={"species_units": species_units})

            error_str = str(exc_info.value).lower()
            assert "title mismatch" in error_str or "mismatch" in error_str

    def test_validate_observable_code_units_fails_on_wrong_units(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject observable code with wrong output units."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change observable code to return wrong units (nanomolar instead of dimensionless)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    import numpy as np\n"
            "    return np.ones(len(time)) * 100.0 * ureg.nanomolar"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value).lower()
        assert "dimensionality mismatch" in error_str or "unit" in error_str

    def test_validate_derivation_code_fails_on_value_mismatch(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject when computed values don't match reported."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Report wrong median (code will compute ~1.0, report as 200)
        data["calibration_target_estimates"]["median"] = [200.0]

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value).lower()
        assert "does not match" in error_str and "median" in error_str

    def test_validate_source_refs_fails_on_undefined_source(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject input.source_ref that doesn't reference defined source."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Reference non-existent source
        data["calibration_target_estimates"]["inputs"][0]["source_ref"] = "nonexistent_source"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not defined" in error_str.lower() and "nonexistent_source" in error_str

    def test_validate_species_exist_fails_on_missing_species(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject species not in model."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Reference species that doesn't exist in species_units
        data["observable"]["species"] = [
            "V_T.CD8",
            "V_T.NonexistentSpecies",
        ]

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not found in model" in error_str and "NonexistentSpecies" in error_str

    def test_validate_inputs_used_warns_on_unused_inputs(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn about unused inputs."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add an assumption that's not used in derivation_code
        data["calibration_target_estimates"]["assumptions"] = [
            {
                "name": "unused_assumption",
                "value": 999.0,
                "units": "dimensionless",
                "description": "This assumption is not used",
                "rationale": "Testing unused input warning",
            }
        ]

        with pytest.warns(UserWarning, match="not used in distribution_code"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None
        assert len(target.calibration_target_estimates.assumptions) == 1

    def test_validate_observable_code_fails_on_scalar_return(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject observable code that returns a scalar instead of array."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change observable code to return scalar (using time indexing)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    ratio = cd8[-1] / tumor[-1]  # Returns scalar (last timepoint)\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value).lower()
        assert "returned a scalar" in error_str or "time indexing" in error_str

    def test_validate_observable_code_fails_on_wrong_length_array(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject observable code that returns array with wrong length."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change observable code to return wrong-length array
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Return only first 5 timepoints (wrong length)\n"
            "    ratio = cd8[:5] / tumor[:5]\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value).lower()
        assert "wrong length" in error_str or "time series" in error_str

    def test_validate_clipping_suggests_lognormal(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when distribution_code uses clipping."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add clipping to distribution_code (use new input names)
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    import math\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['cd8_ratio_mean']\n"
            "    sigma_log = inputs['cd8_ratio_sigma_log']\n"
            "    n = 10000\n"
            "    mu_log = math.log(mean.magnitude)\n"
            "    samples = np.random.lognormal(mu_log, sigma_log.magnitude, n)\n"
            "    samples = np.clip(samples, 0.01, None) * mean.units  # Clipping!\n"
            "    median_obs = np.array([np.median(samples)]) * mean.units\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * mean.units, ci95[1] * mean.units]]\n"
            "    return {'median_obs': median_obs, 'ci95_obs': ci95_obs}"
        )

        with pytest.warns(UserWarning, match="clipping.*lognormal"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None

    def test_validate_large_variance_documented(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when CV > 50% is not documented in limitations."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Large variance test requires inputs named with "mean" and "sd/std/se"
        # Add new inputs for this test that will trigger the CV check
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "ratio_mean",
                "value": 1.0,
                "units": "dimensionless",
                "description": "Mean ratio",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "ratio: 1.0 ± 1.0",
            },
            {
                "name": "ratio_sd",
                "value": 1.0,  # SD = mean gives CV = 100%
                "units": "dimensionless",
                "description": "SD of ratio",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "ratio: 1.0 ± 1.0",
            },
        ]
        data["calibration_target_estimates"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]

        # Update distribution_code to use these inputs
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['ratio_mean']\n"
            "    sd = inputs['ratio_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.abs(np.random.normal(mean.magnitude, sd.magnitude, n)) * mean.units\n"
            "    median_obs = np.array([np.median(samples)]) * mean.units\n"
            "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * mean.units\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * mean.units, ci95[1] * mean.units]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )

        # Update calibration target estimates to match code output (vector format)
        data["calibration_target_estimates"]["median"] = [1.0475]
        data["calibration_target_estimates"]["iqr"] = [1.1671]
        data["calibration_target_estimates"]["ci95"] = [[0.0496, 2.9741]]

        # Don't mention variance in limitations
        data["key_study_limitations"] = "Small sample size from single center"

        with pytest.warns(UserWarning, match="Large coefficient of variation"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None

    def test_validate_distribution_choice_for_size_data(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when using normal distribution for size data."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change units to size data (centimeter) and update values to be in cm range
        data["calibration_target_estimates"]["units"] = "centimeter"

        # Replace inputs with size-appropriate ones
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "diameter_mean",
                "value": 1.5,
                "units": "centimeter",
                "description": "Mean tumor diameter",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "tumor diameter: 1.5 ± 0.25 cm",
            },
            {
                "name": "diameter_sd",
                "value": 0.25,
                "units": "centimeter",
                "description": "SD of tumor diameter",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "tumor diameter: 1.5 ± 0.25 cm",
            },
        ]
        data["calibration_target_estimates"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]
        data["calibration_target_estimates"]["median"] = [1.4994]
        data["calibration_target_estimates"]["iqr"] = [0.3359]
        data["calibration_target_estimates"]["ci95"] = [[1.0079, 1.9935]]

        # Also update observable code to return centimeter (to avoid unit mismatch error)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Hypothetical tumor diameter in centimeters\n"
            "    # Scale cell counts to reasonable diameter range (1-2 cm)\n"
            "    diameter_cm = 1.5 + 0.0 * (cd8 / tumor).magnitude  # ~1.5 cm\n"
            "    return diameter_cm * ureg.centimeter"
        )
        data["observable"]["units"] = "centimeter"

        # Use normal distribution (not lognormal)
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['diameter_mean']\n"
            "    sd = inputs['diameter_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, n)\n"
            "    median_val = np.median(samples)\n"
            "    iqr_val = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
            "    ci95_vals = np.percentile(samples, [2.5, 97.5])\n"
            "    median_obs = np.array([median_val]) * ureg.centimeter\n"
            "    iqr_obs = np.array([iqr_val]) * ureg.centimeter\n"
            "    ci95_obs = [[ci95_vals[0] * ureg.centimeter, ci95_vals[1] * ureg.centimeter]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )

        with pytest.warns(UserWarning, match="normal distribution for size.*lognormal"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None

    def test_validate_conversion_factors_documented(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when observable code has undocumented magic numbers."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add magic number (cell size) to observable code without documenting
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Magic number: 10 = cell radius in micrometers (UNDOCUMENTED)\n"
            "    cell_volume = 10  # This should trigger warning (not using ureg)\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.warns(UserWarning, match="numeric literals.*conversion factors"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None

    def test_validate_dimensionality_error(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch Pint DimensionalityError (e.g., day² → day)."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Create code that produces dimensional mismatch
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Intentional dimension error: time squared can't convert to dimensionless\n"
            "    bad_value = (time * time).to(ureg.dimensionless)  # day² → dimensionless fails\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError, match="unit error"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_undefined_unit_error(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch Pint UndefinedUnitError for unknown units."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Create code that uses undefined unit
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Intentional undefined unit error\n"
            "    bad_value = 5.0 * ureg.foobar  # 'foobar' is not a defined unit\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError, match="unit error"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_scale_mismatch_ratio_vs_score(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch scale mismatch: 0-1 ratio code vs 0-3 score target."""
        data = copy.deepcopy(golden_calibration_target_data)

        # Change observable code to return values in 0-0.1 range (small fractions)
        # Use division by 100 which is an allowed number
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Return small fraction ~0.01 to 0.1 by dividing by 100\n"
            "    ratio = (cd8 / tumor) / 100\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        # But calibration target is on 0-3 score scale (>100x mismatch)
        data["calibration_target_estimates"]["median"] = [2.42]
        data["calibration_target_estimates"]["iqr"] = [0.50]
        data["calibration_target_estimates"]["ci95"] = [[1.69, 3.15]]

        # Update inputs to produce values in score range
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "score_mean",
                "value": 2.42,
                "units": "dimensionless",
                "description": "Mean score",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "score: 2.42 ± 0.37",
            },
            {
                "name": "score_sd",
                "value": 0.37,
                "units": "dimensionless",
                "description": "SD of score",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "score: 2.42 ± 0.37",
            },
        ]
        data["calibration_target_estimates"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]

        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['score_mean']\n"
            "    sd = inputs['score_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, n) * mean.units\n"
            "    median_obs = np.array([np.median(samples)]) * mean.units\n"
            "    iqr_obs = np.array([np.percentile(samples, 75) - np.percentile(samples, 25)]) * mean.units\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_obs = [[ci95[0] * mean.units, ci95[1] * mean.units]]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )

        with pytest.raises(ValidationError, match="Scale mismatch|Magnitude mismatch"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_control_characters(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch control characters in text fields."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add control character to rationale (CalibrationTarget doesn't have description)
        data["rationale"] = "CD8+ T cell\x03 density in PDAC"  # ETX control character

        with pytest.raises(ValidationError, match="Control character"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_hardcoded_constants_fails_on_inline_units(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject hardcoded numbers with units in observable code."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add hardcoded constant with units (1e-8 * ureg.mm**2)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # BAD: hardcoded constant with units\n"
            "    area_per_cell = 2.27e-4 * ureg.mm**2\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError, match="Hardcoded numeric constants"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_hardcoded_constants_passes_with_observable_constants(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should pass when constants are properly declared and accessed."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add constant to observable constants
        data["observable"]["constants"] = [
            {
                "name": "area_per_cancer_cell",
                "value": 2.27e-4,
                "units": "mm**2/cell",
                "biological_basis": "Cancer cell ~17 μm diameter → π×(8.5 μm)² = 2.27e-4 mm²",
                "source_ref": "modeling_assumption",
            }
        ]
        # Use constant via constants dict (no hardcoded numbers with units)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # GOOD: use constant from constants dict\n"
            "    area_per_cell = constants['area_per_cancer_cell']\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        # Should pass without error
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None

    def test_validate_species_completeness_warns_on_missing_related_species(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when measuring 'total' but missing related species."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add CD8_exh to species_units for this test
        species_units_extended = {
            **species_units,
            "V_T.CD8_exh": {"units": "cell", "description": "Exhausted CD8+ T cells"},
        }
        # Change rationale to mention "total" CD8 (validator checks rationale, not description)
        data["rationale"] = "Total CD8+ T cell to tumor cell ratio measured via IHC"
        # Only use V_T.CD8, missing V_T.CD8_exh
        data["observable"]["species"] = ["V_T.CD8", "V_T.C1"]

        with pytest.warns(UserWarning, match="CD8.*CD8_exh"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units_extended}
            )
        assert target is not None

    def test_validate_support_fails_on_invalid_value(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject invalid support type."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Set invalid support type
        data["observable"]["support"] = "invalid_support"

        with pytest.raises(ValidationError, match="support"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_support_field_accepts_valid_types(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should accept all valid support types."""
        valid_support_types = [
            "positive",
            "non_negative",
            "unit_interval",
            "positive_unbounded",
            "real",
        ]

        for support_type in valid_support_types:
            data = copy.deepcopy(golden_calibration_target_data)
            data["observable"]["support"] = support_type

            # Should not raise for valid support types
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )
            assert target.observable.support == support_type


# ============================================================================
# Vector-Valued Data Tests
# ============================================================================


class TestVectorValuedCalibrationTarget:
    """Tests for vector-valued calibration target data (time-course, dose-response)."""

    def test_vector_calibration_target_passes_validation(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that vector-valued calibration target passes all validators."""
        from qsp_llm_workflows.core.calibration import IndexType

        data = copy.deepcopy(golden_calibration_target_data)

        # Use actual computed values from the distribution_code with seed=42
        # These are the values that lognormal(mu_log, 0.5) produces with the given means
        data["calibration_target_estimates"]["median"] = [0.799, 1.008, 1.1965, 1.0943]
        data["calibration_target_estimates"]["iqr"] = [0.5474, 0.6965, 0.8269, 0.7674]
        data["calibration_target_estimates"]["ci95"] = [
            [0.299, 2.1467],
            [0.3768, 2.6694],
            [0.4546, 3.1404],
            [0.4106, 2.9389],
        ]
        data["calibration_target_estimates"]["index_values"] = [0, 7, 14, 21]
        data["calibration_target_estimates"]["index_unit"] = "day"
        data["calibration_target_estimates"]["index_type"] = "time"

        # Update inputs to be vector-valued
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "cd8_ratio_mean",
                "value": [0.8, 1.0, 1.2, 1.1],  # Vector input
                "units": "dimensionless",
                "description": "Mean CD8/tumor ratio at each time point",
                "source_ref": "smith_2020",
                "value_location": "Figure 3",
                "value_snippet": "CD8/tumor ratio increased from 0.8 at baseline to 1.2 at day 14",
            },
            {
                "name": "cd8_ratio_sigma_log",
                "value": 0.5,  # Scalar input (broadcast)
                "units": "dimensionless",
                "description": "Log-scale SD (assumed constant)",
                "source_ref": "smith_2020",
                "value_location": "Figure 3",
                "value_snippet": "variability approximately constant across time",
            },
        ]
        data["calibration_target_estimates"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]

        # Update distribution_code to return vector outputs
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    import math\n"
            "    np.random.seed(42)\n"
            "    means = inputs['cd8_ratio_mean'].magnitude\n"
            "    sigma_log = inputs['cd8_ratio_sigma_log'].magnitude\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    units = inputs['cd8_ratio_mean'].units\n"
            "    n_points = len(means)\n"
            "    medians, iqrs, ci95s = [], [], []\n"
            "    for i in range(n_points):\n"
            "        mu_log = math.log(means[i])\n"
            "        samples = np.random.lognormal(mu_log, sigma_log, n)\n"
            "        medians.append(np.median(samples))\n"
            "        iqrs.append(np.percentile(samples, 75) - np.percentile(samples, 25))\n"
            "        ci95s.append([np.percentile(samples, 2.5) * units, np.percentile(samples, 97.5) * units])\n"
            "    return {\n"
            "        'median_obs': np.array(medians) * units,\n"
            "        'iqr_obs': np.array(iqrs) * units,\n"
            "        'ci95_obs': ci95s\n"
            "    }"
        )

        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})

        assert target is not None
        assert len(target.calibration_target_estimates.median) == 4
        assert target.calibration_target_estimates.index_values == [0, 7, 14, 21]
        assert target.calibration_target_estimates.index_type == IndexType.TIME
        assert target.calibration_target_estimates.index_unit == "day"

    def test_vector_input_length_mismatch_fails(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that vector input with wrong length fails validation."""
        data = copy.deepcopy(golden_calibration_target_data)

        # Set up vector data with 4 time points
        data["calibration_target_estimates"]["median"] = [0.8, 1.0, 1.2, 1.1]
        data["calibration_target_estimates"]["iqr"] = [0.5, 0.68, 0.82, 0.75]
        data["calibration_target_estimates"]["ci95"] = [
            [0.3, 2.2],
            [0.37, 2.7],
            [0.45, 3.2],
            [0.40, 2.95],
        ]
        data["calibration_target_estimates"]["index_values"] = [0, 7, 14, 21]
        data["calibration_target_estimates"]["index_unit"] = "day"
        data["calibration_target_estimates"]["index_type"] = "time"

        # Vector input with wrong length (3 instead of 4)
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "cd8_ratio_mean",
                "value": [0.8, 1.0, 1.2],  # Wrong length!
                "units": "dimensionless",
                "description": "Mean CD8/tumor ratio",
                "source_ref": "smith_2020",
                "value_location": "Figure 3",
                "value_snippet": "data",
            },
        ]

        with pytest.raises(ValidationError, match="length"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_output_length_mismatch_fails(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that output arrays with different lengths fail validation."""
        data = copy.deepcopy(golden_calibration_target_data)

        # median has 4 elements, iqr has 3 - mismatch
        data["calibration_target_estimates"]["median"] = [0.8, 1.0, 1.2, 1.1]
        data["calibration_target_estimates"]["iqr"] = [0.5, 0.68, 0.82]  # Wrong length!
        data["calibration_target_estimates"]["ci95"] = [
            [0.3, 2.2],
            [0.37, 2.7],
            [0.45, 3.2],
            [0.40, 2.95],
        ]
        # Must set index_values to enable length mismatch validation (otherwise fails scalar check)
        data["calibration_target_estimates"]["index_values"] = [0, 7, 14, 21]
        data["calibration_target_estimates"]["index_unit"] = "day"
        data["calibration_target_estimates"]["index_type"] = "time"

        with pytest.raises(ValidationError, match="(lengths must match|must be a list of 4)"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_index_fields_required_together(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that index_values requires index_unit and index_type."""
        data = copy.deepcopy(golden_calibration_target_data)

        # Vector outputs with index_values but missing index_unit
        data["calibration_target_estimates"]["median"] = [0.8, 1.0, 1.2, 1.1]
        data["calibration_target_estimates"]["iqr"] = [0.5, 0.68, 0.82, 0.75]
        data["calibration_target_estimates"]["ci95"] = [
            [0.3, 2.2],
            [0.37, 2.7],
            [0.45, 3.2],
            [0.40, 2.95],
        ]
        data["calibration_target_estimates"]["index_values"] = [0, 7, 14, 21]
        # Missing index_unit and index_type

        with pytest.raises(ValidationError, match="index_unit is required"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_scalar_data_requires_length_one(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that scalar data (no index_values) must have length-1 arrays."""
        data = copy.deepcopy(golden_calibration_target_data)

        # No index_values but median has length > 1
        data["calibration_target_estimates"]["median"] = [0.8, 1.0]  # Length 2 without index_values
        data["calibration_target_estimates"]["iqr"] = [0.5, 0.68]
        data["calibration_target_estimates"]["ci95"] = [[0.3, 2.2], [0.37, 2.7]]
        # No index_values set

        with pytest.raises(ValidationError, match="outputs must have length 1"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_ci95_wrong_inner_structure_fails(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that ci95 entries must be [lower, upper] pairs."""
        data = copy.deepcopy(golden_calibration_target_data)

        # ci95 with wrong inner structure (3 elements instead of 2)
        data["calibration_target_estimates"]["ci95"] = [[0.3, 1.0, 2.7]]  # Wrong!

        with pytest.raises(ValidationError, match="\\[lower, upper\\] pair"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})
