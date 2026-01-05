#!/usr/bin/env python3
"""
Tests for CalibrationTarget model validators.

Tests all 11 active validators with:
- 1 golden test: Valid data passes all validators
- 8 negative tests: Validators fail on invalid data
  - DOI resolution fails
  - Title mismatch fails
  - Wrong measurement code units fails
  - Scalar measurement code return fails (time series length)
  - Wrong-length array measurement code return fails (time series length)
  - Derivation code value mismatch fails
  - Undefined source reference fails
  - Missing species fails
- 5 warning tests: Scientific best practices
  - Clipping suggests lognormal distribution
  - Large variance should be documented
  - Normal distribution inappropriate for size data
  - Conversion factors should be documented
  - Unused inputs emit warning
"""

import copy
import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration_target_models import CalibrationTarget


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
        "description": "CD8+ T cell density in PDAC tumor at resection",
        "scenario": {
            "description": "Baseline PDAC tumor at resection",
            "interventions": [
                {"intervention_description": "No intervention (natural disease progression)"}
            ],
            "measurements": [
                {
                    "measurement_description": (
                        "CD8+ T cell density measured via IHC in tumor tissue sections, "
                        "reported as dimensionless ratio (CD8+ cells / tumor cells)"
                    ),
                    "measurement_species": ["V_T.CD8", "V_T.C1"],
                    "measurement_code": (
                        "def compute_measurement(time, species_dict, ureg):\n"
                        "    cd8 = species_dict['V_T.CD8']\n"
                        "    tumor = species_dict['V_T.C1']\n"
                        "    ratio = cd8 / tumor\n"
                        "    return ratio.to(ureg.dimensionless)"
                    ),
                    "threshold_description": (
                        "At tumor resection when tumor burden reaches approximately 1e9 cells (~500 mm³)"
                    ),
                }
            ],
        },
        "experimental_context": {
            "species": "human",
            "indication": "PDAC",
            "compartment": "tumor.primary",
            "system": "clinical.resection",
            "treatment": {"history": ["treatment_naive"], "status": "off_treatment"},
            "stage": {"extent": "resectable", "burden": "moderate"},
        },
        "study_overview": "Immune profiling of resectable PDAC tumors",
        "study_design": "IHC analysis of CD8+ T cells in tumor sections",
        "derivation_explanation": "Parametric bootstrap from reported mean and SD",
        "key_assumptions": [],
        "key_study_limitations": "Single-center study, limited sample size",
        "calibration_target_estimates": {
            "median": 149.94,
            "iqr": 33.59,
            "ci95": [100.79, 199.35],
            "units": "dimensionless",
            "inputs": [
                {
                    "name": "cd8_density_mean",
                    "value": 150.0,
                    "units": "dimensionless",
                    "description": "Mean CD8/tumor ratio",
                    "source_ref": "smith_2020",
                    "value_table_or_section": "Table 2",
                    "value_snippet": "CD8+ T cell to tumor cell ratio: 150 ± 25",
                },
                {
                    "name": "cd8_density_sd",
                    "value": 25.0,
                    "units": "dimensionless",
                    "description": "SD of CD8/tumor ratio",
                    "source_ref": "smith_2020",
                    "value_table_or_section": "Table 2",
                    "value_snippet": "CD8+ T cell to tumor cell ratio: 150 ± 25",
                },
                {
                    "name": "n_mc_samples",
                    "value": 10000.0,
                    "units": "dimensionless",
                    "description": "MC samples",
                    "source_ref": "modeling_assumption",
                    "value_table_or_section": None,
                    "value_snippet": None,
                },
            ],
            "distribution_code": (
                "def derive_distribution(inputs, ureg):\n"
                "    import numpy as np\n"
                "    np.random.seed(42)\n"
                "    mean = inputs['cd8_density_mean']\n"
                "    sd = inputs['cd8_density_sd']\n"
                "    n = int(inputs['n_mc_samples'].magnitude)\n"
                "    samples = np.random.normal(mean.magnitude, sd.magnitude, n) * mean.units\n"
                "    median_obs = np.median(samples)\n"
                "    iqr_obs = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
                "    ci95_obs = np.percentile(samples, [2.5, 97.5])\n"
                "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
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
        """Test that golden YAML passes all 9 validators."""
        target = CalibrationTarget.model_validate(
            golden_calibration_target_data, context={"species_units": species_units}
        )

        assert target is not None
        assert target.description == "CD8+ T cell density in PDAC tumor at resection"
        assert len(target.scenario.measurements) == 1
        assert target.calibration_target_estimates.median == 149.94


# ============================================================================
# Negative Tests - Each Validator Fails
# ============================================================================


class TestCalibrationTargetValidators:
    """Tests for individual CalibrationTarget validators."""

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

    def test_validate_measurement_code_units_fails_on_wrong_units(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject measurement_code with wrong output units."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change measurement_code to return wrong units (nanomolar instead of dimensionless)
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg):\n"
            "    return 100.0 * ureg.nanomolar"
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
        # Report wrong median (code will compute ~150, report as 200)
        data["calibration_target_estimates"]["median"] = 200.0

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
        data["scenario"]["measurements"][0]["measurement_species"] = [
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
        # Add an input that's not used in derivation_code
        data["calibration_target_estimates"]["inputs"].append(
            {
                "name": "unused_input",
                "value": 999.0,
                "units": "dimensionless",
                "description": "This input is not used",
                "source_ref": "modeling_assumption",
                "value_table_or_section": None,
                "value_snippet": None,
            }
        )

        with pytest.warns(UserWarning, match="not used in distribution_code"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None
        assert len(target.calibration_target_estimates.inputs) == 4  # 3 original + 1 unused

    def test_validate_measurement_code_fails_on_scalar_return(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject measurement_code that returns a scalar instead of array."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change measurement_code to return scalar (using time indexing)
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    ratio = cd8[-1] / tumor[-1]  # Returns scalar (last timepoint)\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value).lower()
        assert "returned a scalar" in error_str or "time indexing" in error_str

    def test_validate_measurement_code_fails_on_wrong_length_array(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject measurement_code that returns array with wrong length."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change measurement_code to return wrong-length array
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg):\n"
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
        # Add clipping to distribution_code
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['cd8_density_mean']\n"
            "    sd = inputs['cd8_density_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, n)\n"
            "    samples = np.clip(samples, 0.01, None) * mean.units  # Clipping!\n"
            "    median_obs = np.median(samples)\n"
            "    iqr_obs = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
            "    ci95_obs = np.percentile(samples, [2.5, 97.5])\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
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
        # Make SD very large (CV = 2.0 = 200%)
        data["calibration_target_estimates"]["inputs"][1]["value"] = 300.0  # SD = 300, mean = 150

        # Update expected values to match what distribution_code will produce with new SD
        # (Must match code output within 1% or derivation validator will fail first)
        data["calibration_target_estimates"]["median"] = 149.94
        data["calibration_target_estimates"]["iqr"] = 403.10
        data["calibration_target_estimates"]["ci95"] = [-438.01, 737.89]

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
        # Change units to size data (centimeter)
        data["calibration_target_estimates"]["units"] = "centimeter"

        # Also update measurement_code to return centimeter (to avoid unit mismatch error)
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Hypothetical size calculation returning pure centimeters\n"
            "    # Use .magnitude to strip cell units, then apply centimeter\n"
            "    size = (cd8.magnitude + tumor.magnitude) * 1e-6 * ureg.centimeter\n"
            "    return size"
        )

        # Use normal distribution (not lognormal)
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['cd8_density_mean']\n"
            "    sd = inputs['cd8_density_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, n) * ureg.centimeter\n"
            "    median_obs = np.median(samples)\n"
            "    iqr_obs = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
            "    ci95_obs = np.percentile(samples, [2.5, 97.5])\n"
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
        """Validator should warn when measurement_code has undocumented magic numbers."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add magic number (cell size) to measurement_code without documenting
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Magic number: 10 = cell radius in micrometers (UNDOCUMENTED)\n"
            "    cell_volume = 10 * ureg.micrometer  # This should trigger warning\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.warns(UserWarning, match="numeric literals.*conversion factors"):
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )

        assert target is not None
