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
        "study_interpretation": (
            "CD8+ T cell density in resectable PDAC tumors measured via IHC. "
            "Parametric bootstrap from reported lognormal distribution (mean 1.0, sigma_log 0.5). "
            "Maps to CD8/tumor ratio in model for immune profiling comparison."
        ),
        "key_assumptions": [
            "CD8+ T cell density follows lognormal distribution",
        ],
        "key_study_limitations": [
            "Single-center study with limited sample size",
            "Lognormal distribution assumed based on positive-only data",
        ],
        "empirical_data": {
            # Vector-valued outputs (length-1 for scalar data)
            "median": [1.0],
            "ci95": [[0.3737, 2.7]],
            "units": "dimensionless",
            "sample_size": 42,
            "sample_size_rationale": "n=42 patients in resected PDAC cohort, Table 1",
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
                    "dispersion_type": "sd",
                    "dispersion_type_rationale": "Paper explicitly states lognormal sigma = 0.5 (log-scale SD)",
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
                "    median_obs = np.median(samples)\n"
                "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
                "    ci95_lower = ci95[0]\n"
                "    ci95_upper = ci95[1]\n"
                "    return {'median_obs': median_obs, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}"
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
        assert "CD8+ T cell density" in target.study_interpretation
        assert target.observable is not None
        assert target.observable.species == ["V_T.CD8", "V_T.C1"]
        # Scalar data uses length-1 lists
        assert target.empirical_data.median == [pytest.approx(1.0, rel=0.01)]
        assert len(target.empirical_data.ci95) == 1
        assert target.empirical_data.index_values is None  # Scalar case


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

    def test_validate_secondary_doi_resolution_fails_on_invalid_doi(
        self, species_units, golden_calibration_target_data
    ):
        """Validator should reject secondary source DOI that doesn't resolve."""
        # Mock primary DOI success but secondary DOI failure
        call_count = [0]

        def mock_get(url, headers=None, timeout=None):
            call_count[0] += 1
            mock_response = Mock()
            # First call is primary DOI (success), subsequent calls for secondary (fail)
            if "10.1000/test.2020.001" in url:
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "title": ["Immune landscape of pancreatic ductal adenocarcinoma"],
                    "author": [{"family": "Smith"}],
                    "issued": {"date-parts": [[2020]]},
                }
            else:
                mock_response.status_code = 404
            return mock_response

        with patch("requests.get", mock_get):
            data = copy.deepcopy(golden_calibration_target_data)
            # Add secondary source with invalid DOI
            data["secondary_data_sources"] = [
                {
                    "source_tag": "jones_2019",
                    "title": "Some reference paper",
                    "first_author": "Jones",
                    "year": 2019,
                    "doi_or_url": "10.9999/invalid.secondary.doi",
                }
            ]

            with pytest.raises(ValidationError) as exc_info:
                CalibrationTarget.model_validate(data, context={"species_units": species_units})

            error_str = str(exc_info.value)
            assert "failed to resolve" in error_str
            assert "jones_2019" in error_str or "Secondary source" in error_str

    def test_validate_secondary_title_match_fails_on_title_mismatch(
        self, species_units, golden_calibration_target_data
    ):
        """Validator should reject secondary source with mismatched title."""

        def mock_get(url, headers=None, timeout=None):
            mock_response = Mock()
            mock_response.status_code = 200
            if "10.1000/test.2020.001" in url:
                mock_response.json.return_value = {
                    "title": ["Immune landscape of pancreatic ductal adenocarcinoma"],
                    "author": [{"family": "Smith"}],
                    "issued": {"date-parts": [[2020]]},
                }
            else:
                # Secondary DOI returns different title than provided
                mock_response.json.return_value = {
                    "title": ["Completely Different Secondary Paper Title"],
                    "author": [{"family": "Jones"}],
                    "issued": {"date-parts": [[2019]]},
                }
            return mock_response

        with patch("requests.get", mock_get):
            data = copy.deepcopy(golden_calibration_target_data)
            # Add secondary source with DOI that resolves to different title
            data["secondary_data_sources"] = [
                {
                    "source_tag": "jones_2019",
                    "title": "Provided title that doesn't match CrossRef",
                    "first_author": "Jones",
                    "year": 2019,
                    "doi_or_url": "10.1000/secondary.2019.001",
                }
            ]

            with pytest.raises(ValidationError) as exc_info:
                CalibrationTarget.model_validate(data, context={"species_units": species_units})

            error_str = str(exc_info.value).lower()
            assert "title mismatch" in error_str or "mismatch" in error_str
            assert "jones_2019" in error_str or "secondary" in error_str

    def test_validate_secondary_source_url_skipped(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should skip URL validation for secondary sources (only DOIs)."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add secondary source with URL (not DOI) - should not be validated
        data["secondary_data_sources"] = [
            {
                "source_tag": "reference_website",
                "title": "Reference Values Database",
                "first_author": "Database",
                "year": 2023,
                "doi_or_url": "https://example.com/reference-values",
            }
        ]

        # Should pass - URL is not validated via CrossRef
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None
        assert len(target.secondary_data_sources) == 1

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
        data["empirical_data"]["median"] = [200.0]

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
        data["empirical_data"]["inputs"][0]["source_ref"] = "nonexistent_source"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not defined" in error_str.lower() and "nonexistent_source" in error_str

    def test_validate_input_values_in_snippets_fails_on_missing_value(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject inputs where value is not found in value_snippet."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change value but keep snippet the same (snippet says 1.0, value is 999)
        data["empirical_data"]["inputs"][0]["value"] = 999.0
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str or "SnippetValueMismatch" in error_str
        assert "999" in error_str

    def test_validate_input_values_in_snippets_passes_with_matching_value(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should pass when values are found in snippets."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Ensure snippet contains the value
        data["empirical_data"]["inputs"][0]["value"] = 1.0
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)"

        # Should pass
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None

    def test_validate_input_values_in_snippets_handles_scientific_notation(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should find values in scientific notation format (Unicode superscripts)."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Use value with Unicode superscript notation in snippet
        # Value 1.0 can be written as 1×10⁰ to test superscript parsing
        data["empirical_data"]["inputs"][0]["value"] = 1.0
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "CD8+ T cell to tumor cell ratio: 1×10⁰ (mean from Figure 2)"

        # Should pass - Unicode superscript notation is handled
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None

    def test_validate_input_values_in_snippets_handles_percentage(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should find percentage values (0.5 in snippet as 50%)."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Use value that matches the observable scale (~1.0) but expressed as percentage
        # The check_value_in_text function should recognize 0.5 in "50%"
        data["empirical_data"]["inputs"][1]["value"] = 0.5
        data["empirical_data"]["inputs"][1][
            "value_snippet"
        ] = "The log-scale SD was 50% of the mean"

        # Should pass - percentage format is handled (0.5 matches "50%")
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None

    def test_validate_input_values_in_snippets_fails_on_vector_missing_value(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject vector inputs where not all values are in snippet."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Vector input where one value is not in snippet
        data["empirical_data"]["inputs"][0]["value"] = [1.0, 2.0, 999.0]
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "Values were 1.0 at baseline, 2.0 at day 7"  # Missing 999.0
        # Set up as vector data
        data["empirical_data"]["median"] = [1.0, 2.0, 2.5]
        data["empirical_data"]["ci95"] = [[0.5, 1.5], [1.0, 3.0], [1.2, 3.8]]
        data["empirical_data"]["index_values"] = [0, 7, 14]
        data["empirical_data"]["index_unit"] = "day"
        data["empirical_data"]["index_type"] = "time"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str or "SnippetValueMismatch" in error_str
        assert "999" in error_str

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
        data["empirical_data"]["assumptions"] = [
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
        assert len(target.empirical_data.assumptions) == 1

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
        data["empirical_data"]["distribution_code"] = (
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
            "    median_obs = np.median(samples)\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_lower = ci95[0]\n"
            "    ci95_upper = ci95[1]\n"
            "    return {'median_obs': median_obs, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}"
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
        data["empirical_data"]["inputs"] = [
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
                "dispersion_type": "sd",
                "dispersion_type_rationale": "Paper states mean ± SD in table legend",
            },
        ]
        data["empirical_data"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]

        # Update distribution_code to use these inputs
        data["empirical_data"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['ratio_mean']\n"
            "    sd = inputs['ratio_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.abs(np.random.normal(mean.magnitude, sd.magnitude, n)) * mean.units\n"
            "    median_obs = np.median(samples)\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_lower = ci95[0]\n"
            "    ci95_upper = ci95[1]\n"
            "    return {'median_obs': median_obs, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}"
        )

        # Update calibration target estimates to match code output (vector format)
        data["empirical_data"]["median"] = [1.0475]
        data["empirical_data"]["iqr"] = [1.1671]
        data["empirical_data"]["ci95"] = [[0.0496, 2.9741]]

        # Don't mention variance in limitations
        data["key_study_limitations"] = ["Small sample size from single center"]

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
        data["empirical_data"]["units"] = "centimeter"

        # Replace inputs with size-appropriate ones
        data["empirical_data"]["inputs"] = [
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
                "dispersion_type": "sd",
                "dispersion_type_rationale": "Explicitly reported as SD in paper methods.",
            },
        ]
        data["empirical_data"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]
        data["empirical_data"]["median"] = [1.4994]
        data["empirical_data"]["iqr"] = [0.3359]
        data["empirical_data"]["ci95"] = [[1.0079, 1.9935]]

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
        data["empirical_data"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['diameter_mean']\n"
            "    sd = inputs['diameter_sd']\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, n)\n"
            "    median_obs = np.median(samples) * ureg.centimeter\n"
            "    ci95_vals = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_lower = ci95_vals[0] * ureg.centimeter\n"
            "    ci95_upper = ci95_vals[1] * ureg.centimeter\n"
            "    return {'median_obs': median_obs, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}"
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

    def test_validate_control_characters(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch control characters in text fields."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add control character to study_interpretation (CalibrationTarget doesn't have description)
        data["study_interpretation"] = "CD8+ T cell\x03 density in PDAC"  # ETX control character

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
                "biological_basis": "From reference DB pdac_cancer_cell_diameter (17 μm) → π×(8.5 μm)² = 2.27e-4 mm²",
                "source_type": "derived_from_reference_db",
                "reference_db_names": ["pdac_cancer_cell_diameter"],
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
        # Change study_interpretation to mention "total" CD8 (validator checks study_interpretation, not description)
        data["study_interpretation"] = "Total CD8+ T cell to tumor cell ratio measured via IHC"
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
        data["empirical_data"]["median"] = [0.799, 1.008, 1.1965, 1.0943]
        data["empirical_data"]["iqr"] = [0.5474, 0.6965, 0.8269, 0.7674]
        data["empirical_data"]["ci95"] = [
            [0.299, 2.1467],
            [0.3768, 2.6694],
            [0.4546, 3.1404],
            [0.4106, 2.9389],
        ]
        data["empirical_data"]["index_values"] = [0, 7, 14, 21]
        data["empirical_data"]["index_unit"] = "day"
        data["empirical_data"]["index_type"] = "time"

        # Update inputs to be vector-valued
        data["empirical_data"]["inputs"] = [
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
                "value_snippet": "variability σ=0.5 approximately constant across time",
                "dispersion_type": "sd",
                "dispersion_type_rationale": "Paper states σ=0.5 as log-scale standard deviation",
            },
        ]
        data["empirical_data"]["assumptions"] = [
            {
                "name": "n_mc_samples",
                "value": 10000.0,
                "units": "dimensionless",
                "description": "MC samples",
                "rationale": "Standard sample size for stable percentile estimates",
            },
        ]

        # Update distribution_code to return vector outputs
        data["empirical_data"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    import math\n"
            "    np.random.seed(42)\n"
            "    means = inputs['cd8_ratio_mean'].magnitude\n"
            "    sigma_log = inputs['cd8_ratio_sigma_log'].magnitude\n"
            "    n = int(inputs['n_mc_samples'].magnitude)\n"
            "    units = inputs['cd8_ratio_mean'].units\n"
            "    n_points = len(means)\n"
            "    medians, lowers, uppers = [], [], []\n"
            "    for i in range(n_points):\n"
            "        mu_log = math.log(means[i])\n"
            "        samples = np.random.lognormal(mu_log, sigma_log, n)\n"
            "        medians.append(np.median(samples))\n"
            "        lowers.append(np.percentile(samples, 2.5))\n"
            "        uppers.append(np.percentile(samples, 97.5))\n"
            "    return {\n"
            "        'median_obs': np.array(medians) * units,\n"
            "        'ci95_lower': np.array(lowers) * units,\n"
            "        'ci95_upper': np.array(uppers) * units,\n"
            "    }"
        )

        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})

        assert target is not None
        assert len(target.empirical_data.median) == 4
        assert target.empirical_data.index_values == [0, 7, 14, 21]
        assert target.empirical_data.index_type == IndexType.TIME
        assert target.empirical_data.index_unit == "day"

    def test_vector_input_length_mismatch_fails(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that vector input with wrong length fails validation."""
        data = copy.deepcopy(golden_calibration_target_data)

        # Set up vector data with 4 time points
        data["empirical_data"]["median"] = [0.8, 1.0, 1.2, 1.1]
        data["empirical_data"]["iqr"] = [0.5, 0.68, 0.82, 0.75]
        data["empirical_data"]["ci95"] = [
            [0.3, 2.2],
            [0.37, 2.7],
            [0.45, 3.2],
            [0.40, 2.95],
        ]
        data["empirical_data"]["index_values"] = [0, 7, 14, 21]
        data["empirical_data"]["index_unit"] = "day"
        data["empirical_data"]["index_type"] = "time"

        # Vector input with wrong length (3 instead of 4)
        data["empirical_data"]["inputs"] = [
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
        data["empirical_data"]["median"] = [0.8, 1.0, 1.2, 1.1]
        data["empirical_data"]["iqr"] = [0.5, 0.68, 0.82]  # Wrong length!
        data["empirical_data"]["ci95"] = [
            [0.3, 2.2],
            [0.37, 2.7],
            [0.45, 3.2],
            [0.40, 2.95],
        ]
        # Must set index_values to enable length mismatch validation (otherwise fails scalar check)
        data["empirical_data"]["index_values"] = [0, 7, 14, 21]
        data["empirical_data"]["index_unit"] = "day"
        data["empirical_data"]["index_type"] = "time"

        with pytest.raises(ValidationError, match="(length mismatch|must be a list of 4)"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_index_fields_required_together(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that index_values requires index_unit and index_type."""
        data = copy.deepcopy(golden_calibration_target_data)

        # Vector outputs with index_values but missing index_unit
        data["empirical_data"]["median"] = [0.8, 1.0, 1.2, 1.1]
        data["empirical_data"]["iqr"] = [0.5, 0.68, 0.82, 0.75]
        data["empirical_data"]["ci95"] = [
            [0.3, 2.2],
            [0.37, 2.7],
            [0.45, 3.2],
            [0.40, 2.95],
        ]
        data["empirical_data"]["index_values"] = [0, 7, 14, 21]
        # Missing index_unit and index_type

        with pytest.raises(ValidationError, match="index_unit is required"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_scalar_data_requires_length_one(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that scalar data (no index_values) must have length-1 arrays."""
        data = copy.deepcopy(golden_calibration_target_data)

        # No index_values but median has length > 1
        data["empirical_data"]["median"] = [0.8, 1.0]  # Length 2 without index_values
        data["empirical_data"]["iqr"] = [0.5, 0.68]
        data["empirical_data"]["ci95"] = [[0.3, 2.2], [0.37, 2.7]]
        # No index_values set

        with pytest.raises(ValidationError, match="outputs must have length 1"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

    def test_ci95_wrong_inner_structure_fails(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that ci95 entries must be [lower, upper] pairs."""
        data = copy.deepcopy(golden_calibration_target_data)

        # ci95 with wrong inner structure (3 elements instead of 2)
        data["empirical_data"]["ci95"] = [[0.3, 1.0, 2.7]]  # Wrong!

        with pytest.raises(ValidationError, match="\\[lower, upper\\] pair"):
            CalibrationTarget.model_validate(data, context={"species_units": species_units})


# ============================================================================
# Regression Tests - Bugs Found in Logfire Traces
# ============================================================================


class TestRegressionBugsFromLogfire:
    """Regression tests for bugs discovered in Logfire traces.

    These tests verify fixes for issues found in production when running
    the IsolatedSystemTarget workflow. Each test corresponds to a specific
    error pattern observed in failed LLM extraction attempts.
    """

    def test_inferred_estimate_skips_snippet_validation(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """REGRESSION: INFERRED_ESTIMATE inputs should skip snippet value check.

        Bug: LLM interpreted qualitative text as numeric (e.g., "maintained viability" → 0.95)
        and validator rejected because 0.95 doesn't appear literally in snippet.

        Fix: Added input_type='inferred_estimate' that skips snippet validation.
        """
        from qsp_llm_workflows.core.calibration.shared_models import InputType

        data = copy.deepcopy(golden_calibration_target_data)

        # Input where value is INTERPRETED from qualitative text (not literal)
        # Use values that match the golden observable code scale (~1.0 mean)
        data["empirical_data"]["inputs"] = [
            {
                "name": "cd8_ratio_mean",
                "value": 0.95,  # NOT in snippet - derived from qualitative statement
                "units": "dimensionless",
                "description": "Interpreted ratio from qualitative statement",
                "source_ref": "smith_2020",
                "value_location": "Methods, section 3",
                "value_snippet": "PSC cultures may be maintained for several months without losing their viability",
                "input_type": "inferred_estimate",  # This should skip validation
            },
            {
                "name": "cd8_ratio_sigma_log",
                "value": 0.5,  # Use same sigma_log as golden
                "units": "dimensionless",
                "description": "Log-scale SD from lognormal fit",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "σ = 0.5 log-scale",  # This IS in snippet
                "input_type": "direct_parameter",
                "dispersion_type": "sd",
                "dispersion_type_rationale": "Paper explicitly states lognormal sigma = 0.5 (log-scale SD)",
            },
        ]
        data["empirical_data"]["assumptions"] = []

        # Use lognormal distribution_code to match golden pattern
        data["empirical_data"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    import math\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['cd8_ratio_mean']\n"
            "    sigma_log = inputs['cd8_ratio_sigma_log']\n"
            "    n = 10000\n"
            "    mu_log = math.log(mean.magnitude)\n"
            "    samples = np.random.lognormal(mu_log, sigma_log.magnitude, n) * mean.units\n"
            "    median_obs = np.median(samples)\n"
            "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
            "    ci95_lower = ci95[0]\n"
            "    ci95_upper = ci95[1]\n"
            "    return {'median_obs': median_obs, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}"
        )

        # Update expected values to match distribution_code output (seed=42)
        data["empirical_data"]["median"] = [0.95]
        data["empirical_data"]["ci95"] = [[0.355, 2.565]]

        # Should pass - inferred_estimate skips snippet validation
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None
        assert target.empirical_data.inputs[0].input_type == InputType.INFERRED_ESTIMATE

    def test_snippet_error_suggests_inferred_estimate(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """REGRESSION: Snippet validation error should suggest inferred_estimate option.

        Bug: Error message didn't tell LLM about inferred_estimate alternative.

        Fix: Error message now includes guidance about input_type='inferred_estimate'.
        """
        data = copy.deepcopy(golden_calibration_target_data)

        # Value not in snippet (without inferred_estimate flag)
        data["empirical_data"]["inputs"][0]["value"] = 0.95
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "cultures maintained without losing viability"  # No 0.95 here

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str
        assert "inferred_estimate" in error_str  # NEW: suggests the alternative

    def test_figure_source_type_skips_snippet_validation(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Figure-sourced inputs should skip snippet value check.

        When source_type='figure', the numeric value is read from a plot and
        won't appear literally in the text snippet (which contains the caption).
        """
        from qsp_llm_workflows.core.calibration.enums import SourceType

        data = copy.deepcopy(golden_calibration_target_data)

        # Input where value is read from a figure (not in snippet text)
        data["empirical_data"]["inputs"] = [
            {
                "name": "cd8_ratio_mean",
                "value": 1.0,
                "units": "dimensionless",
                "description": "Mean CD8/tumor ratio read from scatter plot",
                "source_ref": "smith_2020",
                "value_location": "Figure 2A",
                "value_snippet": "CD8+ T cell infiltration across patient cohort (Figure 2A)",
                "source_type": "figure",
                "figure_id": "Figure 2A",
                "extraction_method": "manual",
                "extraction_notes": "Read from y-axis median marker",
            },
            {
                "name": "cd8_ratio_sigma_log",
                "value": 0.5,
                "units": "dimensionless",
                "description": "Log-scale SD from lognormal fit",
                "source_ref": "smith_2020",
                "value_location": "Table 2",
                "value_snippet": "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)",
                "dispersion_type": "sd",
                "dispersion_type_rationale": "Paper explicitly states lognormal sigma = 0.5 (log-scale SD)",
            },
        ]

        # Should pass - figure source_type skips snippet validation
        target = CalibrationTarget.model_validate(data, context={"species_units": species_units})
        assert target is not None
        assert target.empirical_data.inputs[0].source_type == SourceType.FIGURE

    def test_snippet_error_mentions_figure_source_type(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """Snippet validation error should mention source_type='figure' as option.

        When a value fails snippet validation, the error message should guide
        the LLM to use source_type='figure' if the value comes from a plot.
        """
        data = copy.deepcopy(golden_calibration_target_data)

        # Value not in snippet and no escape hatch set
        data["empirical_data"]["inputs"][0]["value"] = 42.7
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "See Figure 3B for CD8 density across the cohort"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str
        assert "source_type='figure'" in error_str

    def test_distribution_code_array_error_has_helpful_message(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """REGRESSION: "setting an array element with a sequence" error should explain fix.

        Bug: Numpy error was passed through without context, LLM couldn't fix it.

        Fix: Error message now includes helpful guidance for common errors.

        Note: This test verifies the error handling path exists. The specific numpy
        error is hard to trigger reliably, so we verify that distribution_code
        execution errors include helpful guidance.
        """
        data = copy.deepcopy(golden_calibration_target_data)

        # Distribution code that produces a KeyError (easier to trigger than array error)
        data["empirical_data"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    # BUG: Access nonexistent key\n"
            "    mean = inputs['nonexistent_key']\n"
            "    return {'median_obs': mean, 'ci95_lower': mean, 'ci95_upper': mean}"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        # Should include error context and guidance for missing keys
        assert "distribution_code" in error_str.lower() or "key" in error_str.lower()
        # NEW: Should include guidance about where to define inputs
        assert "inputs" in error_str.lower() or "assumptions" in error_str.lower()

    def test_pint_quantity_missing_error_has_helpful_message(
        self, species_units, golden_calibration_target_data, mock_crossref_success
    ):
        """REGRESSION: "median_obs must be Pint Quantity" error should explain fix.

        Bug: Error didn't explain WHY units were missing or HOW to fix.

        Fix: Error message explains units are stripped by sampling and how to reattach.
        """
        data = copy.deepcopy(golden_calibration_target_data)

        # Distribution code that returns plain number without units
        data["empirical_data"]["distribution_code"] = (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    np.random.seed(42)\n"
            "    mean = inputs['cd8_ratio_mean']\n"
            "    # BUG: Return plain number, not Pint Quantity\n"
            "    median_obs = 1.0  # Missing units!\n"
            "    ci95_lower = 0.5  # Also missing units\n"
            "    ci95_upper = 1.5  # Also missing units\n"
            "    return {'median_obs': median_obs, 'ci95_lower': ci95_lower, 'ci95_upper': ci95_upper}"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "Pint Quantity" in error_str
        # NEW: Should include guidance
        assert "reattach" in error_str.lower() or "units" in error_str.lower()

    def test_crossref_empty_title_skips_validation(
        self, species_units, golden_calibration_target_data
    ):
        """REGRESSION: CrossRef returning empty title should skip validation, not fail.

        Bug: CrossRef returned '[]' for title, validator compared against empty string.

        Fix: Skip title validation when CrossRef has no title data.
        """

        def mock_get(url, headers=None, timeout=None):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "title": [],  # CrossRef sometimes returns empty list
                "author": [{"family": "Smith"}],
                "issued": {"date-parts": [[2020]]},
            }
            return mock_response

        with patch("requests.get", mock_get):
            data = copy.deepcopy(golden_calibration_target_data)

            # Should pass - empty CrossRef title means we skip title validation
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )
            assert target is not None

    def test_doi_error_has_verification_link(
        self, mock_crossref_failure, species_units, golden_calibration_target_data
    ):
        """REGRESSION: DOI resolution error should include verification URL.

        Bug: Error just said "failed to resolve" without actionable guidance.

        Fix: Error now includes https://doi.org/<doi> for manual verification.
        """
        data = copy.deepcopy(golden_calibration_target_data)
        data["primary_data_source"]["doi"] = "10.9999/nonexistent.doi"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"species_units": species_units})

        error_str = str(exc_info.value)
        assert "failed to resolve" in error_str
        # NEW: Should include verification guidance
        assert "doi.org" in error_str.lower() or "verify" in error_str.lower()

    def test_secondary_crossref_empty_title_skips_validation(
        self, species_units, golden_calibration_target_data
    ):
        """REGRESSION: Secondary source with empty CrossRef title should skip validation.

        Bug: Same as primary source - CrossRef '[]' title caused spurious failures.
        """

        def mock_get(url, headers=None, timeout=None):
            mock_response = Mock()
            mock_response.status_code = 200
            if "10.1000/test.2020.001" in url:
                # Primary source - normal response
                mock_response.json.return_value = {
                    "title": ["Immune landscape of pancreatic ductal adenocarcinoma"],
                    "author": [{"family": "Smith"}],
                    "issued": {"date-parts": [[2020]]},
                }
            else:
                # Secondary source - empty title
                mock_response.json.return_value = {
                    "title": [],  # Empty!
                    "author": [{"family": "Jones"}],
                    "issued": {"date-parts": [[2019]]},
                }
            return mock_response

        with patch("requests.get", mock_get):
            data = copy.deepcopy(golden_calibration_target_data)
            data["secondary_data_sources"] = [
                {
                    "source_tag": "jones_2019",
                    "title": "Some paper with missing CrossRef title",
                    "first_author": "Jones",
                    "year": 2019,
                    "doi_or_url": "10.1000/secondary.2019.001",
                }
            ]

            # Should pass - empty CrossRef title means we skip title validation
            target = CalibrationTarget.model_validate(
                data, context={"species_units": species_units}
            )
            assert target is not None
            assert len(target.secondary_data_sources) == 1
