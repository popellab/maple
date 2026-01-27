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
    """Minimal valid measurement without distribution_code."""
    return Measurement(
        name="test_measurement",
        units="1/day",
        uses_inputs=["test_value"],
        evaluation_points=[0.0],
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


def make_direct_conversion_target(
    input_value: float,
    prior_mu: float,
    formula: str = "k = value",
    distribution_code: str = None,
):
    """Helper to create a direct_conversion SubmodelTarget for testing."""
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
            "model": {
                "type": "direct_conversion",
                "formula": formula,
                "data_rationale": "Test",
                "submodel_rationale": "Test",
            },
            "measurements": [
                {
                    "name": "test_measurement",
                    "units": "1/day",
                    "uses_inputs": ["test_value"],
                    "evaluation_points": [0.0],
                    "distribution_code": distribution_code,
                    "likelihood": {"distribution": "lognormal"},
                }
            ],
            "identifiability_notes": "Test notes",
        },
        "experimental_context": {
            "species": "human",
            "system": "in_vitro",
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


class TestDistributionCodeRequiredWithFormula:
    """Tests for validate_distribution_code_required_with_formula."""

    def test_direct_conversion_without_distribution_code_fails(self):
        """Direct conversion with formula but no distribution_code should fail."""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value / 2",
            distribution_code=None,
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "distribution_code" in str(exc_info.value)

    def test_direct_conversion_with_distribution_code_passes(self):
        """Direct conversion with distribution_code should pass."""
        distribution_code = """
def derive_distribution(inputs, ureg):
    import numpy as np
    value = inputs['test_value'].magnitude
    return {
        'median': [value] * ureg('1/day'),
        'ci95': [[value * 0.5, value * 2.0]] * ureg('1/day'),
        'units': '1/day',
    }
"""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            distribution_code=distribution_code,
        )

        # Should not raise - but may fail other validators
        # We're testing this specific validator
        try:
            SubmodelTarget(**data)
        except ValidationError as e:
            # Check it's not failing on distribution_code_required
            errors = str(e)
            assert "But no measurement has distribution_code" not in errors


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
        distribution_code = """
def derive_distribution(inputs, ureg):
    value = inputs['test_value'].magnitude
    return {
        'median': [value] * ureg('1/day'),
        'ci95': [[value * 0.5, value * 2.0]],
        'units': '1/day',
    }
"""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=-20.0,  # exp(-20) ≈ 2e-9
            formula="k = value",
            distribution_code=distribution_code,
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        error_str = str(exc_info.value)
        assert "Prior predictive check failed" in error_str or "orders of magnitude" in error_str

    def test_matching_scale_passes(self):
        """Prior and observation on same scale should pass."""
        distribution_code = """
def derive_distribution(inputs, ureg):
    value = inputs['test_value'].magnitude
    return {
        'median': [value] * ureg('1/day'),
        'ci95': [[value * 0.5, value * 2.0]],
        'units': '1/day',
    }
"""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=math.log(10.0),  # median = 10
            formula="k = value",
            distribution_code=distribution_code,
        )

        # Should pass - prior median ≈ observation
        try:
            target = SubmodelTarget(**data)
            assert target is not None
        except ValidationError as e:
            # If it fails, should not be due to scale mismatch
            assert "orders of magnitude" not in str(e)

    def test_distribution_code_error_raises(self):
        """Error in distribution_code execution should raise."""
        distribution_code = """
def derive_distribution(inputs, ureg):
    # This will raise KeyError
    value = inputs['nonexistent_input']
    return {'median': [value]}
"""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            distribution_code=distribution_code,
        )

        with pytest.raises(ValidationError) as exc_info:
            SubmodelTarget(**data)

        assert "distribution_code execution error" in str(exc_info.value)


# ============================================================================
# Tests for validate_clipping_suggests_lognormal
# ============================================================================


class TestClippingSuggestsLognormal:
    """Tests for validate_clipping_suggests_lognormal."""

    def test_clipping_in_distribution_code_warns(self):
        """Using np.clip in distribution_code should warn."""
        distribution_code = """
def derive_distribution(inputs, ureg):
    import numpy as np
    value = inputs['test_value'].magnitude
    # Clipping to avoid negatives
    samples = np.clip(np.random.normal(value, 1, 1000), 0, None)
    return {
        'median': [np.median(samples)] * ureg('1/day'),
        'ci95': [[np.percentile(samples, 2.5), np.percentile(samples, 97.5)]],
        'units': '1/day',
    }
"""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            distribution_code=distribution_code,
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
        """Using np.maximum in distribution_code should warn."""
        distribution_code = """
def derive_distribution(inputs, ureg):
    import numpy as np
    value = inputs['test_value'].magnitude
    samples = np.maximum(np.random.normal(value, 1, 1000), 0)
    return {
        'median': [np.median(samples)] * ureg('1/day'),
        'ci95': [[np.percentile(samples, 2.5), np.percentile(samples, 97.5)]],
        'units': '1/day',
    }
"""
        data = make_direct_conversion_target(
            input_value=10.0,
            prior_mu=math.log(10.0),
            formula="k = value",
            distribution_code=distribution_code,
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
        distribution_code = """
def derive_distribution(inputs, ureg):
    value = inputs['mean_value'].magnitude
    return {
        'median': [value] * ureg('1/day'),
        'ci95': [[value * 0.5, value * 2.0]],
        'units': '1/day',
    }
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
                "model": {
                    "type": "direct_conversion",
                    "formula": "k = mean_value",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                },
                "measurements": [
                    {
                        "name": "test",
                        "units": "1/day",
                        "uses_inputs": ["mean_value"],
                        "evaluation_points": [0.0],
                        "distribution_code": distribution_code,
                        "likelihood": {"distribution": "lognormal"},
                    }
                ],
                "identifiability_notes": "Parameter is identifiable from single observation.",  # No variance keywords
            },
            "experimental_context": {
                "species": "human",
                "system": "in_vitro",
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
        distribution_code = """
def derive_distribution(inputs, ureg):
    value = inputs['mean_value'].magnitude
    return {
        'median': [value] * ureg('1/day'),
        'ci95': [[value * 0.5, value * 2.0]],
        'units': '1/day',
    }
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
                "model": {
                    "type": "direct_conversion",
                    "formula": "k = mean_value",
                    "data_rationale": "Test",
                    "submodel_rationale": "Test",
                },
                "measurements": [
                    {
                        "name": "test",
                        "units": "1/day",
                        "uses_inputs": ["mean_value"],
                        "evaluation_points": [0.0],
                        "distribution_code": distribution_code,
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
