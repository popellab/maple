#!/usr/bin/env python3
"""
Tests for SubmodelTarget validators.

Tests the validators added to SubmodelTarget:
- validate_distribution_code_required_with_formula
- validate_prior_predictive_scale
- validate_clipping_suggests_lognormal
- validate_large_variance_documented
"""

import math
import pytest
import warnings
from unittest.mock import patch
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration.submodel_target import (
    SubmodelTarget,
    Input,
    InputType,
    InputRole,
    ExtractionMethod,
    Parameter,
    Prior,
    PriorDistribution,
    Measurement,
    Likelihood,
    PrimaryDataSource,
)
from qsp_llm_workflows.core.model_structure import ModelStructure, ModelParameter


# ============================================================================
# Mock DOI resolution for all tests
# ============================================================================


@pytest.fixture(autouse=True)
def mock_doi_resolution():
    """Mock DOI resolution to return valid metadata for all tests."""

    def mock_resolve(doi):
        """Return metadata that matches the DOI."""
        return {
            "title": "Test Paper",
            "year": 2023,
            "first_author": "Test",
        }

    with patch(
        "qsp_llm_workflows.core.calibration.validators.resolve_doi",
        side_effect=mock_resolve,
    ):
        yield


@pytest.fixture
def mock_model_structure():
    """
    Mock model structure for unit validation.

    Contains common test parameters with expected units.
    """
    return ModelStructure(
        parameters=[
            ModelParameter(name="k_test", units="1/day"),
            ModelParameter(name="k_prolif", units="1/day"),
            ModelParameter(name="k_death", units="1/day"),
            ModelParameter(name="k_CCL2_sec", units="nanomole/cell/day"),
            ModelParameter(name="test_param", units="1/day"),
        ],
        species=[],
        compartments=[],
        reactions=[],
    )


def validate_target(data: dict, model_structure: ModelStructure) -> SubmodelTarget:
    """
    Helper to validate SubmodelTarget with model_structure context.

    All tests should use this instead of SubmodelTarget(**data) directly.
    """
    return SubmodelTarget.model_validate(
        data,
        context={"model_structure": model_structure},
    )


# ============================================================================
# Fixtures - Minimal valid data for testing
# ============================================================================


@pytest.fixture
def minimal_input():
    """Minimal valid input."""
    return Input(
        name="test_value",
        value=10.0,
        units="1/day",
        input_type=InputType.DIRECT_MEASUREMENT,
        role=InputRole.TARGET,
        source_ref="TestSource2023",
        source_location="Table 1",
        extraction_method=ExtractionMethod.MANUAL,
    )


@pytest.fixture
def minimal_prior():
    """Minimal valid prior."""
    return Prior(
        distribution=PriorDistribution.LOGNORMAL,
        mu=math.log(10.0),  # median = 10
        sigma=0.5,
        rationale="Test prior",
    )


@pytest.fixture
def minimal_parameter(minimal_prior):
    """Minimal valid parameter."""
    return Parameter(
        name="k_test",
        units="1/day",
        prior=minimal_prior,
    )


@pytest.fixture
def minimal_measurement():
    """Minimal valid measurement with observation_code."""
    return Measurement(
        name="test_measurement",
        units="1/day",
        uses_inputs=["test_value"],
        evaluation_points=[0.0],
        observation_code="""
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
""",
        likelihood=Likelihood(distribution="lognormal"),
    )


@pytest.fixture
def minimal_primary_source():
    """Minimal valid primary data source."""
    return PrimaryDataSource(
        doi="10.1234/test.2023",
        title="Test Paper",
        source_tag="TestSource2023",
    )


def make_algebraic_target(
    input_value: float,
    prior_mu: float,
    formula: str = "k = value",
    measurement_error_code: str = None,
):
    """Helper to create an algebraic SubmodelTarget for testing."""
    data = {
        "target_id": "test_target_001",
        "inputs": [
            {
                "name": "test_value",
                "value": input_value,
                "units": "1/day",
                "input_type": "direct_measurement",
                "role": "target",
                "source_ref": "Test2023",
                "source_location": "Table 1",
                "extraction_method": "manual",
            }
        ],
        "calibration": {
            "parameters": [
                {
                    "name": "k_test",
                    "units": "1/day",
                    "prior": {
                        "distribution": "lognormal",
                        "mu": prior_mu,
                        "sigma": 0.5,
                        "rationale": "Test",
                    },
                }
            ],
            "forward_model": {
                "type": "algebraic",
                "formula": formula,
                "code": """
def compute(params, inputs):
    return params['k_test']
""",
                "code_julia": """
function compute(params, inputs)
    return params["k_test"]
end
""",
                "data_rationale": "Test",
                "submodel_rationale": "Test",
            },
            "error_model": [
                {
                    "name": "test_measurement",
                    "units": "1/day",
                    "uses_inputs": ["test_value"],
                    "evaluation_points": [0.0],
                    "observation_code": measurement_error_code,
                    "likelihood": {"distribution": "lognormal"},
                }
            ],
            "identifiability_notes": "Test notes",
        },
        "experimental_context": {
            "species": "human",
            "system": "in_vitro",
        },
        "source_relevance": {
            "indication_match": "exact",
            "indication_match_justification": "Test justification for source relevance with exact indication match.",
            "species_source": "human",
            "species_target": "human",
            "source_quality": "primary_human_in_vitro",
            "perturbation_type": "physiological_baseline",
            "estimated_translation_uncertainty_fold": 1.0,
        },
        "study_interpretation": "Test interpretation of the study data",
        "key_assumptions": ["Assumption 1 for testing"],
        "primary_data_source": {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "source_tag": "Test2023",
        },
        "secondary_data_sources": [],
    }
    return data


# ============================================================================
# Tests for validate_distribution_code_required_with_formula
# ============================================================================


class TestObservationCodeRequired:
    """Tests for observation_code requirement validation."""

    def test_algebraic_without_observation_code_fails(self):
        """Algebraic model without observation_code should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value / 2",
            measurement_error_code=None,
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "observation_code" in str(exc_info.value).lower()

    def test_algebraic_with_observation_code_passes(self):
        """Algebraic model with observation_code should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}  # 100% CV
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        # Should not raise observation_code error
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            # Check it's not failing on observation_code
            errors = str(e)
            assert "observation_code" not in errors.lower()


# ============================================================================
# Tests for validate_prior_predictive_scale
# ============================================================================


class TestPriorPredictiveScale:
    """Tests for validate_prior_predictive_scale."""

    def test_scale_mismatch_raises_error(self):
        """Prior and observation differing by >3 orders of magnitude should fail."""
        # Prior median = exp(-20) ≈ 2e-9
        # Observation = 10
        # Difference = ~10 orders of magnitude
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=-20.0,  # exp(-20) ≈ 2e-9
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value)
        assert "Prior predictive check failed" in error_str or "orders of magnitude" in error_str

    def test_matching_scale_passes(self):
        """Prior and observation on same scale should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),  # median = 10
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        # Should pass - prior median ≈ observation
        try:
            target = SubmodelTarget(**data)
            assert target is not None
        except ValidationError as e:
            # If it fails, should not be due to scale mismatch
            assert "orders of magnitude" not in str(e)

    def test_observation_code_error_raises(self):
        """Error in observation_code execution should raise."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    # This will raise KeyError
    value = inputs['nonexistent_input']
    return {'value': value, 'sd': value}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "observation_code" in str(exc_info.value).lower()


# ============================================================================
# Tests for validate_clipping_suggests_lognormal
# ============================================================================


class TestClippingSuggestsLognormal:
    """Tests for validate_clipping_suggests_lognormal."""

    def test_clipping_in_observation_code_warns(self):
        """Using np.clip in observation_code should warn."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    import numpy as np
    value = inputs['test_value']
    # Clipping to avoid negatives
    sd = np.clip(value, 0, None)
    return {'value': value, 'sd': float(sd)}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                SubmodelTarget(**data)
            except ValidationError:
                pass  # Other validators may fail

            # Check for clipping warning
            clipping_warnings = [x for x in w if "clipping" in str(x.message).lower()]
            assert len(clipping_warnings) > 0

    def test_np_maximum_warns(self):
        """Using np.maximum in observation_code should warn."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    import numpy as np
    value = inputs['test_value']
    sd = np.maximum(value, 0)
    return {'value': value, 'sd': float(sd)}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                SubmodelTarget(**data)
            except ValidationError:
                pass

            clipping_warnings = [x for x in w if "clipping" in str(x.message).lower()]
            assert len(clipping_warnings) > 0


# ============================================================================
# Tests for validate_large_variance_documented
# ============================================================================


class TestLargeVarianceDocumented:
    """Tests for validate_large_variance_documented."""

    def test_high_cv_without_documentation_warns(self):
        """CV > 50% without mention in identifiability_notes should warn."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    # SD = 8, mean = 10, CV = 80%
    return {'value': inputs['mean_value'], 'sd': inputs['sd_value']}
"""
        data = {
            "target_id": "test_high_cv_001",
            "inputs": [
                {
                    "name": "mean_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                },
                {
                    "name": "sd_value",
                    "value": 8.0,  # CV = 8/10 = 80%
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "auxiliary",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                },
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_test",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(10.0),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "algebraic",
                    "formula": "k = mean_value",
                    "code": """
def compute(params, inputs):
    return params['k_test']
""",
                    "code_julia": """
function compute(params, inputs)
    return params["k_test"]
end
""",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                },
                "error_model": [
                    {
                        "name": "test",
                        "units": "1/day",
                        "uses_inputs": ["mean_value", "sd_value"],
                        "evaluation_points": [0.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Parameter is identifiable from single observation.",  # No variance keywords
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                SubmodelTarget(**data)
            except ValidationError:
                pass

            variance_warnings = [
                x for x in w if "coefficient of variation" in str(x.message).lower()
            ]
            assert len(variance_warnings) > 0

    def test_high_cv_with_documentation_no_warning(self):
        """CV > 50% with variance mentioned should not warn."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['mean_value'], 'sd': inputs['sd_value']}
"""
        data = {
            "target_id": "test_high_cv_documented_001",
            "inputs": [
                {
                    "name": "mean_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                },
                {
                    "name": "sd_value",
                    "value": 8.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "auxiliary",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                },
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_test",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(10.0),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "algebraic",
                    "formula": "k = mean_value",
                    "code": """
def compute(params, inputs):
    return params['k_test']
""",
                    "code_julia": """
function compute(params, inputs)
    return params["k_test"]
end
""",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                },
                "error_model": [
                    {
                        "name": "test",
                        "units": "1/day",
                        "uses_inputs": ["mean_value", "sd_value"],
                        "evaluation_points": [0.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                # Mentions variability
                "identifiability_notes": "High variability in the data reflects biological heterogeneity",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                SubmodelTarget(**data)
            except ValidationError:
                pass

            variance_warnings = [
                x for x in w if "coefficient of variation" in str(x.message).lower()
            ]
            assert len(variance_warnings) == 0


# ============================================================================
# Tests for validate_parameter_units_match_model
# ============================================================================


class TestParameterUnitsMatchModel:
    """Tests for validate_parameter_units_match_model."""

    def test_unit_dimensionality_mismatch_fails(self, mock_model_structure):
        """Parameter with wrong dimensionality should fail when context provided."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}
"""
        data = make_algebraic_target(
            input_value=1e-9,
            prior_mu=-20.0,
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )
        # Change parameter units to concentration (wrong dimensionality)
        # k_CCL2_sec expects nanomole/cell/day but we'll use nanomolar/cell/day
        data["calibration"]["parameters"][0]["name"] = "k_CCL2_sec"
        data["calibration"]["parameters"][0]["units"] = "nanomolar/cell/day"
        data["calibration"]["error_model"][0]["units"] = "nanomolar/cell/day"
        # Update forward model code to access the renamed parameter
        data["calibration"]["forward_model"]["code"] = """
def compute(params, inputs):
    return params['k_CCL2_sec']
"""

        with pytest.raises(ValidationError) as exc_info:
            validate_target(data, mock_model_structure)

        assert "dimensionality mismatch" in str(exc_info.value).lower()

    def test_correct_units_passes(self, mock_model_structure):
        """Parameter with correct dimensionality should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}
"""
        data = make_algebraic_target(
            input_value=1e-9,
            prior_mu=-20.0,
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )
        # Use correct units for k_CCL2_sec
        data["calibration"]["parameters"][0]["name"] = "k_CCL2_sec"
        data["calibration"]["parameters"][0]["units"] = "nanomole/cell/day"
        data["calibration"]["error_model"][0]["units"] = "nanomole/cell/day"
        # Update forward model code to access the renamed parameter
        data["calibration"]["forward_model"]["code"] = """
def compute(params, inputs):
    return params['k_CCL2_sec']
"""

        # Should not raise dimensionality error
        try:
            target = validate_target(data, mock_model_structure)
            assert target.calibration.parameters[0].name == "k_CCL2_sec"
        except ValidationError as e:
            # Make sure it's not a dimensionality error
            assert "dimensionality mismatch" not in str(e).lower()

    def test_missing_context_warns(self):
        """Missing model_structure context should warn, not error."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            measurement_error_code=measurement_error_code,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                # No context provided
                SubmodelTarget(**data)
            except ValidationError:
                pass  # May fail other validators

            context_warnings = [x for x in w if "model_structure not provided" in str(x.message)]
            assert len(context_warnings) >= 1


# ============================================================================
# Tests for validate_input_refs
# ============================================================================


class TestValidateInputRefs:
    """Tests for validate_input_refs validator."""

    def test_measurement_references_unknown_input_fails(self):
        """Measurement uses_inputs referencing unknown input should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Change uses_inputs to reference a non-existent input
        data["calibration"]["error_model"][0]["uses_inputs"] = ["nonexistent_input"]

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "unknown input" in str(exc_info.value).lower()
        assert "nonexistent_input" in str(exc_info.value)

    def test_valid_input_refs_passes(self):
        """Valid input references should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # uses_inputs already references "test_value" which exists
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            # Should not fail on input refs
            assert "unknown input" not in str(e).lower()


# ============================================================================
# Tests for validate_source_refs
# ============================================================================


class TestValidateSourceRefs:
    """Tests for validate_source_refs validator."""

    def test_input_references_unknown_source_fails(self):
        """Input with source_ref not matching any source_tag should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Change source_ref to a non-existent source tag
        data["inputs"][0]["source_ref"] = "NonexistentSource2099"

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "invalid source references" in str(exc_info.value).lower()

    def test_valid_source_refs_passes(self):
        """Valid source references should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # source_ref already matches primary_data_source.source_tag ("Test2023")
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            # Should not fail on source refs
            assert "invalid source references" not in str(e).lower()


# ============================================================================
# Tests for validate_parameter_roles
# ============================================================================


class TestValidateParameterRoles:
    """Tests for validate_parameter_roles validator."""

    def test_model_references_undefined_parameter_fails(self):
        """Model parameter_role referencing undefined parameter should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_param_role_001",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_test",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(10.0),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    # rate_constant references a parameter that doesn't exist
                    "rate_constant": "k_nonexistent",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    "state_variables": [
                        {
                            "name": "A",
                            "units": "dimensionless",
                            "initial_condition": {
                                "value": 1.0,
                                "rationale": "Initial condition for testing purposes",
                            },
                        }
                    ],
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                        "span": [0.0, 10.0],
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "invalid parameter references" in str(exc_info.value).lower()
        assert "k_nonexistent" in str(exc_info.value)

    def test_valid_parameter_roles_passes(self):
        """Valid parameter role references should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_param_role_002",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_decay",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(0.1),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    "rate_constant": "k_decay",  # Matches parameter name
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    "state_variables": [
                        {
                            "name": "A",
                            "units": "dimensionless",
                            "initial_condition": {
                                "value": 1.0,
                                "rationale": "Initial condition for testing purposes",
                            },
                        }
                    ],
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                        "span": [0.0, 10.0],
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        # Should not fail on parameter role validation
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "invalid parameter references" not in str(e).lower()


# ============================================================================
# Tests for validate_custom_code_syntax
# ============================================================================


class TestValidateCustomCodeSyntax:
    """Tests for validate_custom_code_syntax validator."""

    def test_algebraic_code_syntax_error_fails(self):
        """Syntax error in AlgebraicModel.code should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
""",
        )
        # Introduce syntax error in model code
        data["calibration"]["forward_model"]["code"] = """
def compute(params, inputs):
    return params['k_test'  # Missing closing bracket - syntax error
"""

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "syntax error" in str(exc_info.value).lower()

    def test_algebraic_wrong_function_name_fails(self):
        """AlgebraicModel.code with wrong function name should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
""",
        )
        # Wrong function name (should be 'compute')
        data["calibration"]["forward_model"]["code"] = """
def wrong_name(params, inputs, ureg):
    return params['k_test']
"""

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "compute" in str(exc_info.value).lower()

    def test_observation_code_syntax_error_fails(self):
        """Syntax error in observation_code should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return {'sd': 1.0  # Missing closing brace - syntax error
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "syntax error" in str(exc_info.value).lower()

    def test_observation_code_wrong_function_name_fails(self):
        """observation_code with wrong function name should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def wrong_name(inputs, ureg):
    return {'value': inputs['test_value'], 'sd': 1.0}
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "derive_observation" in str(exc_info.value).lower()

    def test_valid_code_passes(self):
        """Valid code should pass syntax validation."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
""",
        )

        # Should pass syntax validation
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "syntax error" not in str(e).lower()


# ============================================================================
# Tests for validate_observation_code_execution
# ============================================================================


class TestValidateObservationCodeExecution:
    """Tests for validate_observation_code_execution validator."""

    def test_missing_derive_observation_function_fails(self):
        """observation_code without derive_observation function should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def some_other_function(inputs, ureg):
    return {'value': inputs['test_value'], 'sd': 1.0}
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        # Should fail on function name check
        error_str = str(exc_info.value).lower()
        assert "derive_observation" in error_str

    def test_return_not_dict_fails(self):
        """derive_observation returning non-dict should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return 1.0  # Should return dict
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "dict" in str(exc_info.value).lower()

    def test_missing_sd_key_fails(self):
        """derive_observation returning dict without 'sd' key should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return {'wrong_key': 1.0}
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "sd" in str(exc_info.value).lower()

    def test_negative_sd_fails(self):
        """derive_observation returning negative sd should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': -1.0}  # Negative SD invalid
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "negative" in str(exc_info.value).lower() or "sd" in str(exc_info.value).lower()

    def test_execution_error_fails(self):
        """Runtime error in derive_observation should fail."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    # This will raise KeyError at runtime
    return {'value': inputs['test_value'], 'sd': inputs['nonexistent_input']}
""",
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "error" in str(exc_info.value).lower()

    def test_valid_observation_code_passes(self):
        """Valid observation_code should pass execution validation."""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code="""
def derive_observation(inputs, sample_size):
    value = inputs['test_value']
    return {'value': value, 'sd': value}
""",
        )

        # Should pass execution validation
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "observation_code execution error" not in str(e).lower()


# ============================================================================
# Tests for validate_input_values_in_snippets
# ============================================================================


class TestValidateInputValuesInSnippets:
    """Tests for validate_input_values_in_snippets validator."""

    def test_value_not_in_snippet_fails(self):
        """Input value not appearing in value_snippet should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Add a snippet that doesn't contain the value
        data["inputs"][0]["value_snippet"] = "The rate was measured at 999 per day"

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "not found in snippet" in str(exc_info.value).lower()

    def test_value_in_snippet_passes(self):
        """Input value appearing in value_snippet should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Add a snippet that contains the value
        data["inputs"][0]["value_snippet"] = "The rate constant was 10.0 per day"

        # Should pass - value appears in snippet
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "not found in snippet" not in str(e).lower()

    def test_scientific_notation_in_snippet_passes(self):
        """Value in scientific notation should match snippet."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': inputs['test_value']}
"""
        data = make_algebraic_target(
            input_value=1e-9,
            prior_mu=math.log(1e-9),
            measurement_error_code=measurement_error_code,
        )
        # Value in different scientific notation format
        data["inputs"][0]["value_snippet"] = "The concentration was 1.0 × 10⁻⁹ M"

        # Should pass - scientific notation match
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "not found in snippet" not in str(e).lower()

    def test_skips_experimental_condition(self):
        """Experimental condition inputs should skip snippet validation."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Change input type to experimental_condition with non-matching snippet
        data["inputs"][0]["input_type"] = "experimental_condition"
        data["inputs"][0]["value_snippet"] = "Cells were treated with drug"

        # Should pass - experimental conditions skip snippet validation
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "not found in snippet" not in str(e).lower()


# ============================================================================
# Tests for validate_span_ordering
# ============================================================================


class TestValidateSpanOrdering:
    """Tests for validate_span_ordering validator."""

    def test_span_start_greater_than_end_fails(self):
        """span[0] >= span[1] should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_span_001",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_decay",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(0.1),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    "rate_constant": "k_decay",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    "state_variables": [
                        {
                            "name": "A",
                            "units": "dimensionless",
                            "initial_condition": {
                                "value": 1.0,
                                "rationale": "Initial condition for testing",
                            },
                        }
                    ],
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                        "span": [10.0, 5.0],  # Invalid: start > end
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "span" in str(exc_info.value).lower()

    def test_negative_span_fails(self):
        """Negative span start should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_span_002",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_decay",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(0.1),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    "rate_constant": "k_decay",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    "state_variables": [
                        {
                            "name": "A",
                            "units": "dimensionless",
                            "initial_condition": {
                                "value": 1.0,
                                "rationale": "Initial condition for testing",
                            },
                        }
                    ],
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                        "span": [-5.0, 10.0],  # Invalid: negative start
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value).lower()
        assert "span" in error_str or "non-negative" in error_str


# ============================================================================
# Tests for validate_no_invisible_characters
# ============================================================================


class TestValidateNoInvisibleCharacters:
    """Tests for validate_no_invisible_characters validator."""

    def test_invisible_unicode_in_text_fails(self):
        """Invisible unicode characters in string fields should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Add zero-width space in a text field
        data["study_interpretation"] = "Test interpretation\u200Bwith invisible character"

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "invisible" in str(exc_info.value).lower() or "character" in str(exc_info.value).lower()

    def test_normal_text_passes(self):
        """Normal text without invisible characters should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )

        # Should pass - no invisible characters
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "invisible" not in str(e).lower()


# ============================================================================
# Tests for validate_units_are_valid_pint
# ============================================================================


class TestValidateUnitsAreValidPint:
    """Tests for validate_units_are_valid_pint validator."""

    def test_invalid_unit_string_fails(self):
        """Invalid Pint unit string should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Use an invalid unit string
        data["inputs"][0]["units"] = "invalid_unit_xyz"

        # Should fail - either in unit validation or when trying to use the unit
        with pytest.raises((ValidationError, Exception)) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value).lower()
        assert "invalid_unit_xyz" in error_str or "not defined" in error_str or "not a valid" in error_str

    def test_valid_pint_units_passes(self):
        """Valid Pint unit strings should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Default units are "1/day" which is valid

        # Should pass unit validation
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "not a valid pint unit" not in str(e).lower()


# ============================================================================
# Tests for validate_ode_model_requirements
# ============================================================================


class TestValidateODEModelRequirements:
    """Tests for validate_ode_model_requirements validator."""

    def test_first_order_decay_without_state_variables_fails(self):
        """first_order_decay model without state_variables should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_ode_001",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_decay",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(0.1),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    "rate_constant": "k_decay",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    # No state_variables - should fail
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                        "span": [0.0, 10.0],
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "state_variables" in str(exc_info.value).lower()

    def test_first_order_decay_without_span_fails(self):
        """first_order_decay model without independent_variable span should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_ode_002",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_decay",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(0.1),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    "rate_constant": "k_decay",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    "state_variables": [
                        {
                            "name": "A",
                            "units": "dimensionless",
                            "initial_condition": {
                                "value": 1.0,
                                "rationale": "Initial condition for testing",
                            },
                        }
                    ],
                    # No span in independent_variable - should fail
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "span" in str(exc_info.value).lower()

    def test_algebraic_without_state_variables_passes(self):
        """algebraic model without state_variables should pass (non-ODE)."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Algebraic models don't require state_variables

        # Should pass
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "state_variables" not in str(e).lower()

    def test_valid_ode_model_passes(self):
        """Valid ODE model with state_variables and span should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = {
            "target_id": "test_ode_003",
            "inputs": [
                {
                    "name": "test_value",
                    "value": 10.0,
                    "units": "1/day",
                    "input_type": "direct_measurement",
                    "role": "target",
                    "source_ref": "Test2023",
                    "source_location": "Table 1",
                    "extraction_method": "manual",
                }
            ],
            "calibration": {
                "parameters": [
                    {
                        "name": "k_decay",
                        "units": "1/day",
                        "prior": {
                            "distribution": "lognormal",
                            "mu": math.log(0.1),
                            "sigma": 0.5,
                        },
                    }
                ],
                "forward_model": {
                    "type": "first_order_decay",
                    "rate_constant": "k_decay",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                    "state_variables": [
                        {
                            "name": "A",
                            "units": "dimensionless",
                            "initial_condition": {
                                "value": 1.0,
                                "rationale": "Initial condition for testing",
                            },
                        }
                    ],
                    "independent_variable": {
                        "name": "time",
                        "units": "day",
                        "span": [0.0, 10.0],
                    },
                },
                "error_model": [
                    {
                        "name": "test_measurement",
                        "units": "1/day",
                        "uses_inputs": ["test_value"],
                        "evaluation_points": [10.0],
                        "observation_code": measurement_error_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Test notes",
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
            },
            "source_relevance": {
                "indication_match": "exact",
                "indication_match_justification": "Test justification for source relevance with exact indication match.",
                "species_source": "human",
                "species_target": "human",
                "source_quality": "primary_human_in_vitro",
                "perturbation_type": "physiological_baseline",
                "estimated_translation_uncertainty_fold": 1.0,
            },
            "study_interpretation": "Test interpretation",
            "key_assumptions": ["Test assumption"],
            "primary_data_source": {
                "doi": "10.1234/test",
                "title": "Test Paper",
                "source_tag": "Test2023",
            },
            "secondary_data_sources": [],
        }

        # Should pass all ODE requirements
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            # Check it's not failing on ODE requirements
            error_str = str(e).lower()
            assert "state_variables" not in error_str or "require" not in error_str


# ============================================================================
# Tests for validate_cross_species_uncertainty
# ============================================================================


class TestValidateCrossSpeciesUncertainty:
    """Tests for validate_cross_species_uncertainty validator."""

    def test_cross_species_without_sufficient_uncertainty_fails(self):
        """Cross-species extrapolation with low uncertainty should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Set cross-species with low uncertainty
        data["source_relevance"]["species_source"] = "mouse"
        data["source_relevance"]["species_target"] = "human"
        data["source_relevance"]["estimated_translation_uncertainty_fold"] = 1.5  # Too low

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value).lower()
        assert "cross-species" in error_str or "uncertainty" in error_str

    def test_cross_species_with_sufficient_uncertainty_passes(self):
        """Cross-species extrapolation with adequate uncertainty should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Set cross-species with adequate uncertainty
        data["source_relevance"]["species_source"] = "mouse"
        data["source_relevance"]["species_target"] = "human"
        data["source_relevance"]["estimated_translation_uncertainty_fold"] = 3.0  # Adequate

        # Should pass
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "cross-species" not in str(e).lower()


# ============================================================================
# Tests for validate_cross_indication_uncertainty
# ============================================================================


class TestValidateCrossIndicationUncertainty:
    """Tests for validate_cross_indication_uncertainty validator."""

    def test_cross_indication_proxy_without_sufficient_uncertainty_fails(self):
        """Cross-indication proxy with low uncertainty should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Set proxy indication with low uncertainty
        data["source_relevance"]["indication_match"] = "proxy"
        data["source_relevance"]["estimated_translation_uncertainty_fold"] = 2.0  # Too low for proxy

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value).lower()
        assert "cross-indication" in error_str or "proxy" in error_str

    def test_cross_indication_with_sufficient_uncertainty_passes(self):
        """Cross-indication extrapolation with adequate uncertainty should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Set proxy indication with adequate uncertainty
        data["source_relevance"]["indication_match"] = "proxy"
        data["source_relevance"]["indication_match_justification"] = (
            "This is a proxy indication that requires adequate justification text for testing purposes."
        )
        data["source_relevance"]["estimated_translation_uncertainty_fold"] = 5.0  # Adequate for proxy

        # Should pass
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "cross-indication" not in str(e).lower()


# ============================================================================
# Tests for validate_pharmacological_perturbation_justification
# ============================================================================


class TestValidatePerturbationJustification:
    """Tests for validate_pharmacological_perturbation_justification validator."""

    def test_pharmacological_perturbation_without_justification_fails(self):
        """Pharmacological perturbation without perturbation_relevance should fail."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Set pharmacological perturbation without justification
        data["source_relevance"]["perturbation_type"] = "pharmacological"
        # perturbation_relevance not provided

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value).lower()
        assert "pharmacological" in error_str or "perturbation_relevance" in error_str

    def test_pharmacological_perturbation_with_justification_passes(self):
        """Pharmacological perturbation with perturbation_relevance should pass."""
        measurement_error_code = """
def derive_observation(inputs, sample_size):
    return {'value': inputs['test_value'], 'sd': 1.0}
"""
        data = make_algebraic_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            measurement_error_code=measurement_error_code,
        )
        # Set pharmacological perturbation with justification
        data["source_relevance"]["perturbation_type"] = "pharmacological"
        data["source_relevance"]["perturbation_relevance"] = (
            "The drug-induced response reflects the same biological pathway being modeled, "
            "as the mechanism of action directly targets the parameter being estimated."
        )

        # Should pass
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            assert "perturbation_relevance" not in str(e).lower()
