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

Scalar output shape tests:
- Multi-element median fails (every target is a scalar; median must be length 1)
- CI95 with more than one pair fails (must be a single [[lo, hi]])
- CI95 wrong inner structure fails ([lo, mid, hi] instead of [lo, hi])

Note: calibration targets no longer carry an index axis (index_values /
index_unit / index_type are gone). The reduction from the observable
time-series to the single compared scalar is declared on the Observable as
either ``readout_time`` (+ ``readout_time_unit``) or ``reduce_observable``.
"""

import copy
import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from maple.core.calibration import CalibrationTarget, Observable


DEFAULT_CLINICAL_SOURCE_RELEVANCE = {
    "indication_match": "exact",
    "indication_match_justification": "Human PDAC resection specimens with quantitative IHC, directly matching the model indication.",
    "species_source": "human",
    "species_target": "human",
    "source_quality": "primary_human_clinical",
    "perturbation_type": "physiological_baseline",
    "perturbation_relevance": "Observational measurement of treatment-naive PDAC specimens; no experimental perturbation applied.",
    "tme_compatibility": "high",
    "tme_compatibility_notes": "Treatment-naive PDAC with characteristic desmoplastic, immune-excluded TME matching model assumptions.",
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model_structure():
    """Minimal ModelStructure for testing validators."""
    from maple.core.model_structure import ModelStructure, ModelSpecies

    return ModelStructure(
        species=[
            ModelSpecies(
                name="V_T.C1",
                compartment="V_T",
                base_name="C1",
                units="cell",
                description="Tumor cells",
            ),
            ModelSpecies(
                name="V_T.CD8",
                compartment="V_T",
                base_name="CD8",
                units="cell",
                description="CD8+ T cells",
            ),
            ModelSpecies(
                name="V_T.Treg",
                compartment="V_T",
                base_name="Treg",
                units="cell",
                description="Regulatory T cells",
            ),
            ModelSpecies(
                name="V_T.TGFb",
                compartment="V_T",
                base_name="TGFb",
                units="nanomolarity",
                description="TGF-beta concentration",
            ),
        ],
    )


@pytest.fixture
def golden_calibration_target_data():
    """Complete valid CalibrationTarget data that passes all validators."""
    return {
        "observable": {
            "code": (
                "def compute_observable(time, species_dict, constants):\n"
                "    cd8 = species_dict['V_T.CD8']\n"
                "    tumor = species_dict['V_T.C1']\n"
                "    return cd8 / tumor"
            ),
            "units": "dimensionless",
            "species": ["V_T.CD8", "V_T.C1"],
            "constants": [],
            # Baseline / diagnosis snapshot: read the observable series at t=0.
            "readout_time": 0.0,
            "readout_time_unit": "day",
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
            "source_relevance": DEFAULT_CLINICAL_SOURCE_RELEVANCE,
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
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that golden YAML passes all 11 validators (including scale/control char)."""
        target = CalibrationTarget.model_validate(
            golden_calibration_target_data, context={"model_structure": model_structure}
        )

        assert target is not None
        assert "CD8+ T cell density" in target.study_interpretation
        assert target.observable is not None
        assert target.observable.species == ["V_T.CD8", "V_T.C1"]
        # Scalar data uses length-1 lists
        assert target.empirical_data.median == [pytest.approx(1.0, rel=0.01)]
        assert len(target.empirical_data.ci95) == 1
        # Reduction is declared on the observable: a t=0 baseline snapshot.
        assert target.observable.readout_time == 0.0
        assert target.observable.readout_time_unit == "day"
        assert target.observable.reduce_observable is None


class TestCalibrationTargetContextOptional:
    """Model-structure-dependent validators must DEFER when no validation
    context is supplied, and still ENFORCE when it is.

    Rationale: pydantic-ai runs CalibrationTarget output validation during the
    Agent loop with no context. If ``validate_species_exist`` /
    ``validate_observable_code_units`` hard-raised there, every retry would fail
    and cal-mode extraction could never complete. ``run_complete`` re-validates
    WITH context immediately after the agent call, so deferring here loses
    nothing (regression: cal-mode extraction wedged at 7/7 retries).
    """

    def test_validate_without_context_defers(
        self, golden_calibration_target_data, mock_crossref_success
    ):
        """Golden target validates with NO context (agent-loop path)."""
        target = CalibrationTarget.model_validate(golden_calibration_target_data)
        assert target is not None
        assert target.observable is not None

    def test_bad_species_still_rejected_with_context(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """The authoritative context-carrying check still catches bad species."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["species"] = ["V_T.NOT_A_REAL_SPECIES"]
        with pytest.raises(ValidationError):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_bad_species_slips_through_without_context(
        self, golden_calibration_target_data, mock_crossref_success
    ):
        """Without context the species check is deferred, not enforced —
        confirms the deferral is real (the post-agent check is what guards)."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["species"] = ["V_T.NOT_A_REAL_SPECIES"]
        target = CalibrationTarget.model_validate(data)  # no context -> defers
        assert target is not None

    def test_wrong_signature_still_rejected_without_context(
        self, golden_calibration_target_data, mock_crossref_success
    ):
        """The context-FREE syntax/signature check must run even in the
        context-less Agent loop, so a wrong observable.code signature raises a
        retriable ValidationError (letting the model self-correct) rather than
        passing the loop and only hard-failing at the post-agent check."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["code"] = data["observable"]["code"].replace(
            "def compute_observable(time, species_dict, constants)",
            "def compute_observable(time, species_dict)",
        )
        with pytest.raises(ValidationError, match="wrong signature"):
            CalibrationTarget.model_validate(data)  # no context


# ============================================================================
# Negative Tests - Each Validator Fails
# ============================================================================


class TestCalibrationTargetValidators:
    """Tests for individual CalibrationTarget validators."""

    def test_observable_required_for_calibration_target(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """CalibrationTarget must have observable field - it's required for full model targets."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Remove observable field
        del data["observable"]

        with pytest.raises(ValidationError, match="observable"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_validate_doi_resolution_fails_on_invalid_doi(
        self, mock_crossref_failure, model_structure, golden_calibration_target_data
    ):
        """Validator should reject DOI that doesn't resolve."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["primary_data_source"]["doi"] = "10.9999/invalid.doi"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "failed to resolve" in error_str

    def test_validate_secondary_doi_resolution_fails_on_invalid_doi(
        self, model_structure, golden_calibration_target_data
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
                    "source_relevance": DEFAULT_CLINICAL_SOURCE_RELEVANCE,
                }
            ]

            with pytest.raises(ValidationError) as exc_info:
                CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

            error_str = str(exc_info.value)
            assert "failed to resolve" in error_str
            assert "jones_2019" in error_str or "Secondary source" in error_str

    def test_validate_secondary_title_match_fails_on_title_mismatch(
        self, model_structure, golden_calibration_target_data
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
                    "source_relevance": DEFAULT_CLINICAL_SOURCE_RELEVANCE,
                }
            ]

            with pytest.raises(ValidationError) as exc_info:
                CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

            error_str = str(exc_info.value).lower()
            assert "title mismatch" in error_str or "mismatch" in error_str
            assert "jones_2019" in error_str or "secondary" in error_str

    def test_validate_secondary_source_url_skipped(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
                "source_relevance": DEFAULT_CLINICAL_SOURCE_RELEVANCE,
            }
        ]

        # Should pass - URL is not validated via CrossRef
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None
        assert len(target.secondary_data_sources) == 1

    def test_validate_title_match_fails_on_title_mismatch(
        self, model_structure, golden_calibration_target_data
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
                CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

            error_str = str(exc_info.value).lower()
            assert "title mismatch" in error_str or "mismatch" in error_str

    def test_validate_observable_code_rejects_pint_return(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Post-Pint-strip: observable.code must return raw floats, not Pint Quantities."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    import numpy as np\n"
            "    from maple.core.unit_registry import ureg\n"
            "    return np.ones(len(time)) * 100.0 * ureg.dimensionless"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value).lower()
        assert "raw float" in error_str or "pint quantity" in error_str

    def test_validate_derivation_code_fails_on_value_mismatch(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject when computed values don't match reported."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Report wrong median (code will compute ~1.0, report as 200)
        data["empirical_data"]["median"] = [200.0]

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value).lower()
        assert "does not match" in error_str and "median" in error_str

    def test_validate_source_refs_fails_on_undefined_source(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject input.source_ref that doesn't reference defined source."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Reference non-existent source
        data["empirical_data"]["inputs"][0]["source_ref"] = "nonexistent_source"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "not defined" in error_str.lower() and "nonexistent_source" in error_str

    def test_validate_input_values_in_snippets_fails_on_missing_value(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject inputs where value is not found in value_snippet."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change value but keep snippet the same (snippet says 1.0, value is 999)
        data["empirical_data"]["inputs"][0]["value"] = 999.0
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str or "SnippetValueMismatch" in error_str
        assert "999" in error_str

    def test_validate_input_values_in_snippets_passes_with_matching_value(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should pass when values are found in snippets."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Ensure snippet contains the value
        data["empirical_data"]["inputs"][0]["value"] = 1.0
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "CD8+ T cell to tumor cell ratio: 1.0 ± 0.5 (lognormal)"

        # Should pass
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None

    def test_validate_input_values_in_snippets_handles_scientific_notation(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None

    def test_validate_input_values_in_snippets_handles_percentage(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None

    def test_validate_input_values_in_snippets_fails_on_vector_missing_value(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject vector inputs where not all values are in snippet."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Vector input where one value is not in snippet
        data["empirical_data"]["inputs"][0]["value"] = [1.0, 2.0, 999.0]
        data["empirical_data"]["inputs"][0][
            "value_snippet"
        ] = "Values were 1.0 at baseline, 2.0 at day 7"  # Missing 999.0
        # The estimate itself stays scalar (median/ci95 are length-1); only the
        # INPUT is list-valued, and every element must appear in the snippet.

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str or "SnippetValueMismatch" in error_str
        assert "999" in error_str

    def test_validate_species_exist_fails_on_missing_species(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject species not in model."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Reference species that doesn't exist in species_units
        data["observable"]["species"] = [
            "V_T.CD8",
            "V_T.NonexistentSpecies",
        ]

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "not found in model" in error_str and "NonexistentSpecies" in error_str

    def test_validate_inputs_used_warns_on_unused_inputs(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
                data, context={"model_structure": model_structure}
            )

        assert target is not None
        assert len(target.empirical_data.assumptions) == 1

    def test_validate_observable_code_fails_on_scalar_return(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject observable code that returns a scalar instead of array."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change observable code to return scalar (using time indexing)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    ratio = cd8[-1] / tumor[-1]  # Returns scalar (last timepoint)\n"
            "    return ratio"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value).lower()
        assert "returned a scalar" in error_str or "time indexing" in error_str

    def test_validate_observable_code_fails_on_wrong_length_array(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject observable code that returns array with wrong length."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Change observable code to return wrong-length array
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Return only first 5 timepoints (wrong length)\n"
            "    ratio = cd8[:5] / tumor[:5]\n"
            "    return ratio"
        )

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value).lower()
        assert "wrong length" in error_str or "time series" in error_str

    def test_validate_clipping_suggests_lognormal(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
                data, context={"model_structure": model_structure}
            )

        assert target is not None

    def test_validate_large_variance_documented(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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

        data["empirical_data"]["ci95"] = [[0.0496, 2.9741]]

        # Don't mention variance in limitations
        data["key_study_limitations"] = ["Small sample size from single center"]

        with pytest.warns(UserWarning, match="Large coefficient of variation"):
            target = CalibrationTarget.model_validate(
                data, context={"model_structure": model_structure}
            )

        assert target is not None

    def test_validate_distribution_choice_for_size_data(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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

        data["empirical_data"]["ci95"] = [[1.0079, 1.9935]]

        # Observable returns raw float array in centimeters (no Pint).
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    import numpy as np\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Hypothetical tumor diameter; scale cell counts to ~1.5 cm.\n"
            "    return 1.5 + 0.0 * (cd8 / tumor)  # cm"
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
                data, context={"model_structure": model_structure}
            )

        assert target is not None

    def test_validate_conversion_factors_documented(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when observable code has undocumented magic numbers."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add magic number (cell size) to observable code without documenting
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # Magic number: 10 = cell radius in micrometers (UNDOCUMENTED)\n"
            "    cell_volume = 10  # This should trigger warning (not using ureg)\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio"
        )

        with pytest.warns(UserWarning, match="numeric literals.*conversion factors"):
            target = CalibrationTarget.model_validate(
                data, context={"model_structure": model_structure}
            )

        assert target is not None

    # Pint dimensionality and undefined-unit tests removed 2026-05-10:
    # observable.code is now Pintless. Unit consistency is enforced inside the
    # function body via explicit numerical conversions, which surface as
    # ordinary numerical/typing errors at derive time.

    def test_validate_control_characters(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should catch control characters in text fields."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add control character to study_interpretation (CalibrationTarget doesn't have description)
        data["study_interpretation"] = "CD8+ T cell\x03 density in PDAC"  # ETX control character

        with pytest.raises(ValidationError, match="Control character"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_validate_hardcoded_constants_fails_on_inline_units(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject hardcoded numbers with units in observable code."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Add hardcoded constant with units (1e-8 * ureg.mm**2)
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # BAD: hardcoded constant with units\n"
            "    area_per_cell = 2.27e-4 * ureg.mm**2\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio"
        )

        with pytest.raises(ValidationError, match="Hardcoded numeric constants"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_validate_hardcoded_constants_passes_with_observable_constants(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
            "def compute_observable(time, species_dict, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    # GOOD: use constant from constants dict\n"
            "    area_per_cell = constants['area_per_cancer_cell']\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio"
        )

        # Should pass without error
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None

    def test_auxiliary_parameter_accessible_in_observable_code(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Observable code can reference an auxiliary parameter via the constants dict.

        Auxiliary parameters are sampled per-simulation at inference time but at
        schema-validation time maple injects a 1.0-magnitude stub of the right
        units so observable.code can be exec'd and unit-checked.
        """
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["auxiliary_parameters"] = [
            {
                "name": "f_serum_to_tumor",
                "group": "serum_to_tumor",
                "biological_basis": (
                    "Serum:tumor concentration ratio for the cytokine; "
                    "observable.code multiplies the model tumor concentration by "
                    "this factor to predict the human serum concentration."
                ),
                "units": "dimensionless",
            }
        ]
        data["observable"]["code"] = (
            "def compute_observable(time, species_dict, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    f = constants['f_serum_to_tumor']\n"
            "    ratio = (cd8 / tumor) * f\n"
            "    return ratio"
        )

        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert len(target.observable.auxiliary_parameters) == 1
        aux = target.observable.auxiliary_parameters[0]
        assert aux.name == "f_serum_to_tumor"
        assert aux.group == "serum_to_tumor"
        assert aux.units == "dimensionless"

    def test_auxiliary_parameter_units_validated(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Bad pint units on an auxiliary parameter should fail validation."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["auxiliary_parameters"] = [
            {
                "name": "bad_aux",
                "group": "test_group",
                "biological_basis": (
                    "Auxiliary parameter declared with a unit string that pint "
                    "cannot parse, used as a regression test for unit validation."
                ),
                "units": "this_is_not_a_unit_string",
            }
        ]
        with pytest.raises(ValidationError, match="auxiliary_parameters"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_auxiliary_parameter_biological_basis_min_length(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """biological_basis must be substantive (>=20 chars), like ObservableConstant."""
        data = copy.deepcopy(golden_calibration_target_data)
        data["observable"]["auxiliary_parameters"] = [
            {
                "name": "f_test",
                "group": "test_group",
                "biological_basis": "too short",
                "units": "dimensionless",
            }
        ]
        with pytest.raises(ValidationError):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_auxiliary_parameters_default_empty(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Existing cal targets without auxiliary_parameters keep working."""
        data = copy.deepcopy(golden_calibration_target_data)
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target.observable.auxiliary_parameters == []

    def test_validate_species_completeness_warns_on_missing_related_species(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should warn when measuring 'total' but missing related species."""
        from maple.core.model_structure import ModelStructure, ModelSpecies

        data = copy.deepcopy(golden_calibration_target_data)
        # Add CD8_exh to model structure for this test
        ms_extended = ModelStructure(
            species=list(model_structure.species)
            + [
                ModelSpecies(
                    name="V_T.CD8_exh",
                    compartment="V_T",
                    base_name="CD8_exh",
                    units="cell",
                    description="Exhausted CD8+ T cells",
                ),
            ],
        )
        # Change study_interpretation to mention "total" CD8 (validator checks study_interpretation, not description)
        data["study_interpretation"] = "Total CD8+ T cell to tumor cell ratio measured via IHC"
        # Only use V_T.CD8, missing V_T.CD8_exh
        data["observable"]["species"] = ["V_T.CD8", "V_T.C1"]

        with pytest.warns(UserWarning, match="CD8.*CD8_exh"):
            target = CalibrationTarget.model_validate(
                data, context={"model_structure": ms_extended}
            )
        assert target is not None

    def test_validate_support_fails_on_invalid_value(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Validator should reject invalid support type."""
        data = copy.deepcopy(golden_calibration_target_data)
        # Set invalid support type
        data["observable"]["support"] = "invalid_support"

        with pytest.raises(ValidationError, match="support"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_support_field_accepts_valid_types(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
                data, context={"model_structure": model_structure}
            )
            assert target.observable.support == support_type


# ============================================================================
# Scalar Output Shape Tests
# ============================================================================


class TestScalarOutputShape:
    """Every calibration target is a SCALAR measurement: ``median`` is a
    length-1 list and ``ci95`` a single ``[[lower, upper]]`` pair. The
    time-series reduction that produces that scalar is declared on the
    observable (``readout_time`` XOR ``reduce_observable``), not by an index
    axis on the empirical data.
    """

    def test_median_longer_than_one_fails(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """A multi-element median is no longer a vector target — it is an error."""
        data = copy.deepcopy(golden_calibration_target_data)

        data["empirical_data"]["median"] = [0.8, 1.0]  # Length 2
        data["empirical_data"]["ci95"] = [[0.3, 2.2], [0.37, 2.7]]

        with pytest.raises(ValidationError, match="median must be a length-1 list"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_ci95_more_than_one_pair_fails(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """ci95 must carry exactly one pair, matching the length-1 median."""
        data = copy.deepcopy(golden_calibration_target_data)

        # median stays scalar, but ci95 supplies two pairs -> shape mismatch.
        data["empirical_data"]["ci95"] = [[0.3, 2.2], [0.37, 2.7]]

        with pytest.raises(ValidationError, match=r"ci95 must be a single \[lower, upper\] pair"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_ci95_wrong_inner_structure_fails(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Test that ci95 entries must be [lower, upper] pairs."""
        data = copy.deepcopy(golden_calibration_target_data)

        # ci95 with wrong inner structure (3 elements instead of 2)
        data["empirical_data"]["ci95"] = [[0.3, 1.0, 2.7]]  # Wrong!

        with pytest.raises(ValidationError, match=r"ci95 must be \[\[lower, upper\]\]"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})


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
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """REGRESSION: INFERRED_ESTIMATE inputs should skip snippet value check.

        Bug: LLM interpreted qualitative text as numeric (e.g., "maintained viability" → 0.95)
        and validator rejected because 0.95 doesn't appear literally in snippet.

        Fix: Added input_type='inferred_estimate' that skips snippet validation.
        """
        from maple.core.calibration.shared_models import InputType

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
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None
        assert target.empirical_data.inputs[0].input_type == InputType.INFERRED_ESTIMATE

    def test_snippet_error_suggests_inferred_estimate(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str
        assert "inferred_estimate" in error_str  # NEW: suggests the alternative

    def test_figure_source_type_skips_snippet_validation(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        """Figure-sourced inputs should skip snippet value check.

        When source_type='figure', the numeric value is read from a plot and
        won't appear literally in the text snippet (which contains the caption).
        """
        from maple.core.calibration.enums import SourceType

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
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target is not None
        assert target.empirical_data.inputs[0].source_type == SourceType.FIGURE

    def test_snippet_error_mentions_figure_source_type(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "not found in value_snippet" in error_str
        assert "source_type='figure'" in error_str

    def test_distribution_code_array_error_has_helpful_message(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        # Should include error context and guidance for missing keys
        assert "distribution_code" in error_str.lower() or "key" in error_str.lower()
        # NEW: Should include guidance about where to define inputs
        assert "inputs" in error_str.lower() or "assumptions" in error_str.lower()

    def test_pint_quantity_missing_error_has_helpful_message(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
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
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "Pint Quantity" in error_str
        # NEW: Should include guidance
        assert "reattach" in error_str.lower() or "units" in error_str.lower()

    def test_crossref_empty_title_skips_validation(
        self, model_structure, golden_calibration_target_data
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
                data, context={"model_structure": model_structure}
            )
            assert target is not None

    def test_doi_error_has_verification_link(
        self, mock_crossref_failure, model_structure, golden_calibration_target_data
    ):
        """REGRESSION: DOI resolution error should include verification URL.

        Bug: Error just said "failed to resolve" without actionable guidance.

        Fix: Error now includes https://doi.org/<doi> for manual verification.
        """
        data = copy.deepcopy(golden_calibration_target_data)
        data["primary_data_source"]["doi"] = "10.9999/nonexistent.doi"

        with pytest.raises(ValidationError) as exc_info:
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

        error_str = str(exc_info.value)
        assert "failed to resolve" in error_str
        # NEW: Should include verification guidance
        assert "doi.org" in error_str.lower() or "verify" in error_str.lower()

    def test_secondary_crossref_empty_title_skips_validation(
        self, model_structure, golden_calibration_target_data
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
                    "source_relevance": DEFAULT_CLINICAL_SOURCE_RELEVANCE,
                }
            ]

            # Should pass - empty CrossRef title means we skip title validation
            target = CalibrationTarget.model_validate(
                data, context={"model_structure": model_structure}
            )
            assert target is not None
            assert len(target.secondary_data_sources) == 1


class TestObservableDenominatorAudit:
    """Tests for the Observable denominator audit validator."""

    def _make_observable(self, **overrides):
        """Helper to create Observable with sensible defaults."""
        base = {
            "code": (
                "def compute_observable(time, species_dict, constants):\n"
                "    return species_dict['V_T.CD8']"
            ),
            "units": "dimensionless",
            "species": ["V_T.CD8"],
            "support": "positive_unbounded",
            # Every Observable must declare its reduction; these denominator
            # tests are indifferent to which, so use a t=0 baseline snapshot.
            "readout_time": 0.0,
            "readout_time_unit": "day",
        }
        base.update(overrides)
        return Observable(**base)

    def test_observable_without_denominator_fields_passes(self):
        """Observable without denominator fields passes for non-density units."""
        obs = self._make_observable()
        assert obs.experimental_denominator is None
        assert obs.model_denominator_species is None

    def test_density_observable_without_experimental_denominator_fails(self):
        """Density observable (cell/mm**2) must declare experimental_denominator."""
        with pytest.raises(ValidationError, match="experimental_denominator"):
            self._make_observable(
                units="cell / millimeter**2",
                support="positive",
            )

    def test_density_observable_with_denominator_audit_passes(self):
        """Density observable with full denominator audit passes."""
        obs = self._make_observable(
            units="cell / millimeter**2",
            support="positive",
            experimental_denominator="mm^2 of tumor tissue (whole section including stroma)",
            model_denominator_species=["V_T.C1"],
        )
        assert obs.experimental_denominator is not None
        assert obs.model_denominator_species == ["V_T.C1"]

    def test_experimental_denominator_without_model_species_fails(self):
        """Setting experimental_denominator without model_denominator_species fails."""
        with pytest.raises(ValidationError, match="model_denominator_species"):
            self._make_observable(
                experimental_denominator="CD3+ T cells",
            )

    def test_fraction_with_full_denominator_audit_passes(self):
        """Fraction observable with all denominator fields passes."""
        obs = self._make_observable(
            units="dimensionless",
            support="unit_interval",
            experimental_denominator="all cells in ROI (all nucleated cells)",
            model_denominator_species=["V_T.CD8", "V_T.Th", "V_T.Treg", "V_T.Mac_M1"],
            unmodeled_denominator_components=(
                "B cells (50-70% of LA cells) not modeled; model prediction "
                "will be ~2-3x higher than experimental value."
            ),
        )
        assert obs.unmodeled_denominator_components is not None

    def test_non_density_units_with_slash_no_cell_passes(self):
        """Non-cell density units like nanomolarity (nM) don't trigger the audit."""
        obs = self._make_observable(
            units="nanomolarity",
            support="positive",
        )
        assert obs.experimental_denominator is None


class TestCalibrationTargetPopulationSample:
    """The optional declared 'samples' population draw + population_spread gate."""

    # A lognormal population draw whose median matches the golden reported median (1.0).
    _GOOD_CODE = (
        "def derive_distribution(inputs, ureg):\n"
        "    import numpy as np, math\n"
        "    np.random.seed(42)\n"
        "    mean = inputs['cd8_ratio_mean']\n"
        "    sigma_log = inputs['cd8_ratio_sigma_log']\n"
        "    mu_log = math.log(mean.magnitude)\n"
        "    samples = np.random.lognormal(mu_log, sigma_log.magnitude, 10000) * mean.units\n"
        "    ci95 = np.percentile(samples, [2.5, 97.5])\n"
        "    return {'median_obs': np.median(samples), 'ci95_lower': ci95[0],\n"
        "            'ci95_upper': ci95[1], 'samples': samples}"
    )

    def _with_code(self, golden, code=None, **ed_overrides):
        data = copy.deepcopy(golden)
        if code is not None:
            data["empirical_data"]["distribution_code"] = code
        data["empirical_data"].update(ed_overrides)
        return data

    def test_defaults_center_only(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        # The golden distribution_code returns no samples; the default is center_only,
        # so it validates and is excluded from omega.
        target = CalibrationTarget.model_validate(
            golden_calibration_target_data, context={"model_structure": model_structure}
        )
        assert target.empirical_data.population_spread == "center_only"

    def test_across_patient_with_samples_validates(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        data = self._with_code(
            golden_calibration_target_data, self._GOOD_CODE, population_spread="across_patient"
        )
        target = CalibrationTarget.model_validate(
            data, context={"model_structure": model_structure}
        )
        assert target.empirical_data.population_spread == "across_patient"

    def test_across_patient_without_samples_rejected(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        # Declaring across_patient but returning no samples is a hard error.
        data = self._with_code(golden_calibration_target_data, population_spread="across_patient")
        with pytest.raises(ValidationError, match="requires distribution_code to"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_center_only_with_samples_rejected(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        # Returning a population sample while declaring center_only is contradictory.
        data = self._with_code(
            golden_calibration_target_data, self._GOOD_CODE
        )  # default center_only
        with pytest.raises(ValidationError, match="must NOT return a 'samples'"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_samples_median_mismatch_rejected(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        # samples centered 5x off the reported/computed median -> rejected
        code = self._GOOD_CODE.replace("'samples': samples}", "'samples': samples * 5.0}")
        data = self._with_code(
            golden_calibration_target_data, code, population_spread="across_patient"
        )
        with pytest.raises(ValidationError, match=r"median\(samples\)"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    def test_degenerate_samples_rejected(
        self, model_structure, golden_calibration_target_data, mock_crossref_success
    ):
        # A flat (zero-variance) sample is not a usable population spread.
        code = self._GOOD_CODE.replace(
            "'samples': samples}",
            "'samples': np.ones(10000) * mean.magnitude * mean.units}",
        )
        data = self._with_code(
            golden_calibration_target_data, code, population_spread="across_patient"
        )
        with pytest.raises(ValidationError, match="zero variance"):
            CalibrationTarget.model_validate(data, context={"model_structure": model_structure})


# ============================================================================
# Cal-side parity for the submodel bounded->logit_normal validator: a bounded
# observable's moments-form observed_distribution must use shape=logit_normal.
# ============================================================================

from maple.core.calibration.calibration_target_models import CalibrationTargetEstimates


def _bounded_cal_estimates(shape: str) -> dict:
    return {
        "median": [0.5],
        "ci95": [[0.3, 0.7]],
        "units": "percent",
        "sample_size": 40,
        "sample_size_rationale": "n=40 patients, Table 1",
        "inputs": [
            {
                "name": "resp_fraction",
                "value": 0.5,
                "units": "percent",
                "description": "objective response fraction",
                "source_ref": "smith_2020",
                "value_location": "Table 1",
                "value_snippet": "response rate 0.5",
            }
        ],
        "distribution_code": (
            "def derive_distribution(inputs, ureg):\n"
            "    v = inputs['resp_fraction']\n"
            "    return {'median_obs': v, 'ci95_lower': v * 0.6, 'ci95_upper': v * 1.4}"
        ),
        "population_spread": "center_only",
        "observed_distribution": {
            "moments": {
                "center": 0.5,
                "center_type": "median",
                "scale": 0.1,
                "scale_type": "sd",
                "shape": shape,
            },
            "spread_source": "center_only",
        },
    }


class TestCalBoundedObservableLogitNormal:
    """CalibrationTargetEstimates: bounded moments-form observable must use logit_normal."""

    def test_percent_with_normal_shape_raises(self):
        with pytest.raises(ValidationError, match="logit_normal"):
            CalibrationTargetEstimates.model_validate(_bounded_cal_estimates("normal"))

    def test_percent_with_logit_normal_passes(self):
        CalibrationTargetEstimates.model_validate(_bounded_cal_estimates("logit_normal"))
