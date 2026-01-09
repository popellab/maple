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
                    "measurement_constants": [],
                    "measurement_code": (
                        "def compute_measurement(time, species_dict, ureg, constants):\n"
                        "    cd8 = species_dict['V_T.CD8']\n"
                        "    tumor = species_dict['V_T.C1']\n"
                        "    ratio = cd8 / tumor\n"
                        "    return ratio.to(ureg.dimensionless)"
                    ),
                    "threshold_description": (
                        "At tumor resection when tumor burden reaches approximately 1e9 cells (~500 mm³)"
                    ),
                    "support": "positive_unbounded",
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
            "median": 1.0,
            "iqr": 0.6843,
            "ci95": [0.3737, 2.7],
            "units": "dimensionless",
            "inputs": [
                {
                    "name": "cd8_ratio_mean",
                    "value": 1.0,
                    "units": "dimensionless",
                    "description": "Mean CD8/tumor ratio",
                    "source_ref": "smith_2020",
                    "value_table_or_section": "Table 2",
                    "value_snippet": "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)",
                },
                {
                    "name": "cd8_ratio_sigma_log",
                    "value": 0.5,
                    "units": "dimensionless",
                    "description": "Log-scale SD of CD8/tumor ratio",
                    "source_ref": "smith_2020",
                    "value_table_or_section": "Table 2",
                    "value_snippet": "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)",
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
                "    import math\n"
                "    np.random.seed(42)\n"
                "    mean = inputs['cd8_ratio_mean']\n"
                "    sigma_log = inputs['cd8_ratio_sigma_log']\n"
                "    n = int(inputs['n_mc_samples'].magnitude)\n"
                "    mu_log = math.log(mean.magnitude)\n"
                "    samples = np.random.lognormal(mu_log, sigma_log.magnitude, n) * mean.units\n"
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
        """Test that golden YAML passes all 11 validators (including scale/control char)."""
        target = CalibrationTarget.model_validate(
            golden_calibration_target_data, context={"species_units": species_units}
        )

        assert target is not None
        assert target.description == "CD8+ T cell density in PDAC tumor at resection"
        assert len(target.scenario.measurements) == 1
        assert target.calibration_target_estimates.median == pytest.approx(1.0, rel=0.01)


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
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    mu_log = math.log(mean.magnitude)\n"
            "    samples = np.random.lognormal(mu_log, sigma_log.magnitude, n)\n"
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
        # Large variance test requires inputs named with "mean" and "sd/std/se"
        # Add new inputs for this test that will trigger the CV check
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "ratio_mean",
                "value": 1.0,
                "units": "dimensionless",
                "description": "Mean ratio",
                "source_ref": "smith_2020",
                "value_table_or_section": "Table 2",
                "value_snippet": "ratio: 1.0 ± 1.0",
            },
            {
                "name": "ratio_sd",
                "value": 1.0,  # SD = mean gives CV = 100%
                "units": "dimensionless",
                "description": "SD of ratio",
                "source_ref": "smith_2020",
                "value_table_or_section": "Table 2",
                "value_snippet": "ratio: 1.0 ± 1.0",
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
            "    median_obs = np.median(samples)\n"
            "    iqr_obs = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
            "    ci95_obs = np.percentile(samples, [2.5, 97.5])\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )

        # Update calibration target estimates to match code output
        data["calibration_target_estimates"]["median"] = 1.0475
        data["calibration_target_estimates"]["iqr"] = 1.1671
        data["calibration_target_estimates"]["ci95"] = [0.0496, 2.9741]

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
                "value_table_or_section": "Table 2",
                "value_snippet": "tumor diameter: 1.5 ± 0.25 cm",
            },
            {
                "name": "diameter_sd",
                "value": 0.25,
                "units": "centimeter",
                "description": "SD of tumor diameter",
                "source_ref": "smith_2020",
                "value_table_or_section": "Table 2",
                "value_snippet": "tumor diameter: 1.5 ± 0.25 cm",
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
        ]
        data["calibration_target_estimates"]["median"] = 1.4994
        data["calibration_target_estimates"]["iqr"] = 0.3359
        data["calibration_target_estimates"]["ci95"] = [1.0079, 1.9935]

        # Also update measurement_code to return centimeter (to avoid unit mismatch error)
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg, constants):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Hypothetical tumor diameter in centimeters\n"
            "    # Scale cell counts to reasonable diameter range (1-2 cm)\n"
            "    diameter_cm = 1.5 + 0.0 * (cd8 / tumor).magnitude  # ~1.5 cm\n"
            "    return diameter_cm * ureg.centimeter"
        )
        data["scenario"]["measurements"][0]["support"] = "positive"

        # Use normal distribution (not lognormal)
        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['diameter_mean']\n"
            "    sd = inputs['diameter_sd']\n"
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
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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

        # Change measurement_code to return values in 0-0.1 range (small fractions)
        # Use division by 100 which is an allowed number
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg, constants):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Return small fraction ~0.01 to 0.1 by dividing by 100\n"
            "    ratio = (cd8 / tumor) / 100\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        # But calibration target is on 0-3 score scale (>100x mismatch)
        data["calibration_target_estimates"]["median"] = 2.42
        data["calibration_target_estimates"]["iqr"] = 0.50
        data["calibration_target_estimates"]["ci95"] = [1.69, 3.15]

        # Update inputs to produce values in score range
        data["calibration_target_estimates"]["inputs"] = [
            {
                "name": "score_mean",
                "value": 2.42,
                "units": "dimensionless",
                "description": "Mean score",
                "source_ref": "smith_2020",
                "value_table_or_section": "Table 2",
                "value_snippet": "score: 2.42 ± 0.37",
            },
            {
                "name": "score_sd",
                "value": 0.37,
                "units": "dimensionless",
                "description": "SD of score",
                "source_ref": "smith_2020",
                "value_table_or_section": "Table 2",
                "value_snippet": "score: 2.42 ± 0.37",
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
        ]

        data["calibration_target_estimates"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['score_mean']\n"
            "    sd = inputs['score_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, n) * mean.units\n"
            "    median_obs = np.median(samples)\n"
            "    iqr_obs = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
            "    ci95_obs = np.percentile(samples, [2.5, 97.5])\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )

        with pytest.raises(ValidationError, match="Scale mismatch|Magnitude mismatch"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_control_characters(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch control characters in text fields."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add control character to description
        data["description"] = "CD8+ T cell\x03 density in PDAC"  # ETX control character

        with pytest.raises(ValidationError, match="Control character"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_hardcoded_constants_fails_on_inline_units(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject hardcoded numbers with units in measurement_code."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add hardcoded constant with units (1e-8 * ureg.mm**2)
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # BAD: hardcoded constant with units\n"
            "    area_per_cell = 2.27e-4 * ureg.mm**2\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )

        with pytest.raises(ValidationError, match="Hardcoded numeric constants"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_validate_hardcoded_constants_passes_with_measurement_constants(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should pass when constants are properly declared and accessed."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add constant to measurement_constants
        data["scenario"]["measurements"][0]["measurement_constants"] = [
            {
                "name": "area_per_cancer_cell",
                "value": 2.27e-4,
                "units": "mm**2/cell",
                "biological_basis": "Cancer cell ~17 μm diameter → π×(8.5 μm)² = 2.27e-4 mm²",
                "source_ref": "modeling_assumption",
            }
        ]
        # Use constant via constants dict (no hardcoded numbers with units)
        data["scenario"]["measurements"][0]["measurement_code"] = (
            "def compute_measurement(time, species_dict, ureg, constants):\n"
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
        # Change description to mention "total" CD8
        data["scenario"]["measurements"][0][
            "measurement_description"
        ] = "Total CD8+ T cell to tumor cell ratio measured via IHC"
        # Only use V_T.CD8, missing V_T.CD8_exh
        data["scenario"]["measurements"][0]["measurement_species"] = ["V_T.CD8", "V_T.C1"]

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
        data["scenario"]["measurements"][0]["support"] = "invalid_support"

        with pytest.raises(ValidationError, match="support"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_support_field_accepts_valid_types(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should accept all valid support types that are compatible with measurement_code output."""
        # Note: unit_interval is excluded because the golden measurement_code produces
        # ratios that can exceed 1, which violates unit_interval support
        compatible_support_types = [
            "positive",
            "non_negative",
            "positive_unbounded",
            "real",
        ]

        for support_type in compatible_support_types:
            data = copy.deepcopy(golden_calibration_target_data)
            data["scenario"]["measurements"][0]["support"] = support_type

            # Should not raise for valid support types
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )
            assert target.scenario.measurements[0].support == support_type
