#!/usr/bin/env python3
"""
Tests for the shared quantile-anchor variability layer.

Covers:
- QuantileAnchor / ObservedDistribution validators (probability range, non-crossing
  quantile function, population-spread requires a scale, technical units cannot be a
  population spread)
- ObservedDistribution derivations (median, quantile interpolation, IQR)
- SpreadSource / POPULATION_SPREAD_SOURCES routing semantics
- Wiring into CalibrationTargetEstimates (resolved_spread_source fallback + consistency)
  and ErrorModel (additive, backwards-compatible)
"""

import pytest
from pydantic import ValidationError

from maple.core.calibration.shared_models import (
    ExperimentalUnitType,
    ObservedDistribution,
    POPULATION_SPREAD_SOURCES,
    QuantileAnchor,
    SourceRelevanceAssessment,
    SpreadSource,
)
from maple.core.calibration.enums import HeterogeneityTransfer
from maple.core.calibration.calibration_target_models import CalibrationTargetEstimates
from maple.core.calibration.submodel_target import ErrorModel


_SOURCE_RELEVANCE = dict(
    indication_match="exact",
    indication_match_justification="exact PDAC match",
    species_source="human",
    source_quality="primary_human_in_vitro",
    perturbation_type="physiological_baseline",
    perturbation_relevance="baseline",
    tme_compatibility="high",
    tme_compatibility_notes="recapitulates target biology",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_DEFAULT_OBS_CODE = (
    "def derive_observation(inputs, sample_size, rng, n_bootstrap):\n"
    "    return rng.normal(0.0, 1.0, n_bootstrap)"
)


def _median_iqr(q25, q50, q75, **kwargs):
    # Inject the biological provenance a population spread now requires, so tests
    # that aren't specifically about that requirement stay focused.
    if kwargs.get("spread_source") in POPULATION_SPREAD_SOURCES:
        kwargs.setdefault("n_biological", 42)
        kwargs.setdefault("experimental_unit_type", ExperimentalUnitType.BIOLOGICAL)
    return ObservedDistribution(
        quantiles=[
            QuantileAnchor(p=0.25, value=q25),
            QuantileAnchor(p=0.5, value=q50),
            QuantileAnchor(p=0.75, value=q75),
        ],
        **kwargs,
    )


def _cal_estimates(**overrides):
    base = dict(
        median=[15.0],
        ci95=[[10.0, 25.0]],
        units="cell/mm^2",
        sample_size=42,
        sample_size_rationale="n=42 in Methods",
        inputs=[],
        distribution_code="def derive_distribution(inputs, ureg): return {}",
    )
    base.update(overrides)
    return CalibrationTargetEstimates(**base)


# ---------------------------------------------------------------------------
# QuantileAnchor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.5])
def test_quantile_anchor_p_must_be_open_unit_interval(bad_p):
    with pytest.raises(ValidationError):
        QuantileAnchor(p=bad_p, value=1.0)


def test_quantile_anchor_valid():
    a = QuantileAnchor(p=0.5, value=3.0)
    assert a.p == 0.5 and a.value == 3.0


# ---------------------------------------------------------------------------
# ObservedDistribution validators
# ---------------------------------------------------------------------------


def test_empty_quantiles_rejected():
    with pytest.raises(ValidationError):
        ObservedDistribution(quantiles=[], spread_source=SpreadSource.CENTER_ONLY)


def test_duplicate_probability_levels_rejected():
    with pytest.raises(ValidationError, match="duplicate probability"):
        ObservedDistribution(
            quantiles=[QuantileAnchor(p=0.5, value=1.0), QuantileAnchor(p=0.5, value=2.0)],
            spread_source=SpreadSource.CENTER_ONLY,
        )


def test_crossing_quantile_function_rejected():
    # value must be non-decreasing in p
    with pytest.raises(ValidationError, match="non-decreasing"):
        ObservedDistribution(
            quantiles=[QuantileAnchor(p=0.25, value=20.0), QuantileAnchor(p=0.75, value=10.0)],
            spread_source=SpreadSource.ACROSS_PATIENT,
        )


def test_population_spread_requires_a_scale():
    # single anchor cannot be a population spread (no width)
    with pytest.raises(ValidationError, match="only a single quantile anchor"):
        ObservedDistribution(
            quantiles=[QuantileAnchor(p=0.5, value=1.0)],
            spread_source=SpreadSource.ACROSS_PATIENT,
        )


def test_center_only_single_anchor_allowed():
    d = ObservedDistribution(
        quantiles=[QuantileAnchor(p=0.5, value=3.0)], spread_source=SpreadSource.CENTER_ONLY
    )
    assert d.feeds_population_spread is False
    assert d.iqr() is None


@pytest.mark.parametrize("unit", [ExperimentalUnitType.TECHNICAL, ExperimentalUnitType.CLONAL])
def test_technical_unit_cannot_be_population_spread(unit):
    with pytest.raises(ValidationError, match="not population variability"):
        _median_iqr(
            10.0,
            15.0,
            20.0,
            spread_source=SpreadSource.BIOLOGICAL_EXPERIMENTAL,
            experimental_unit_type=unit,
        )


def test_anchors_stored_in_probability_order():
    d = ObservedDistribution(
        quantiles=[
            QuantileAnchor(p=0.75, value=20.0),
            QuantileAnchor(p=0.25, value=10.0),
            QuantileAnchor(p=0.5, value=15.0),
        ],
        spread_source=SpreadSource.ACROSS_PATIENT,
        n_biological=30,
        experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
    )
    assert [q.p for q in d.quantiles] == [0.25, 0.5, 0.75]


def test_population_spread_requires_n_biological():
    with pytest.raises(ValidationError, match="n_biological is not set"):
        ObservedDistribution(
            quantiles=[QuantileAnchor(p=0.25, value=10.0), QuantileAnchor(p=0.75, value=20.0)],
            spread_source=SpreadSource.BIOLOGICAL_EXPERIMENTAL,
            experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
        )


def test_population_spread_requires_experimental_unit_type():
    with pytest.raises(ValidationError, match="experimental_unit_type is not set"):
        ObservedDistribution(
            quantiles=[QuantileAnchor(p=0.25, value=10.0), QuantileAnchor(p=0.75, value=20.0)],
            spread_source=SpreadSource.ACROSS_PATIENT,
            n_biological=30,
        )


def test_center_only_exempt_from_biological_provenance():
    d = ObservedDistribution(
        quantiles=[QuantileAnchor(p=0.5, value=3.0)], spread_source=SpreadSource.CENTER_ONLY
    )
    assert d.n_biological is None and d.experimental_unit_type is None


# ---------------------------------------------------------------------------
# ObservedDistribution derivations
# ---------------------------------------------------------------------------


def test_median_and_iqr():
    d = _median_iqr(10.0, 15.0, 25.0, spread_source=SpreadSource.ACROSS_PATIENT, n_biological=42)
    assert d.median() == 15.0
    assert d.iqr() == 15.0


def test_quantile_interpolation_and_clamping():
    d = _median_iqr(10.0, 15.0, 25.0, spread_source=SpreadSource.ACROSS_PATIENT)
    # midway between p=0.25 (10) and p=0.5 (15) at p=0.375 -> 12.5
    assert d.quantile(0.375) == pytest.approx(12.5)
    # clamp below/above the anchor range
    assert d.quantile(0.01) == 10.0
    assert d.quantile(0.99) == 25.0


def test_iqr_none_when_range_not_spanned():
    d = ObservedDistribution(
        quantiles=[QuantileAnchor(p=0.4, value=1.0), QuantileAnchor(p=0.6, value=2.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        n_biological=30,
        experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
    )
    assert d.iqr() is None


# ---------------------------------------------------------------------------
# SpreadSource routing
# ---------------------------------------------------------------------------


def test_population_spread_sources_membership():
    assert SpreadSource.ACROSS_PATIENT in POPULATION_SPREAD_SOURCES
    assert SpreadSource.BIOLOGICAL_EXPERIMENTAL in POPULATION_SPREAD_SOURCES
    for s in (
        SpreadSource.TECHNICAL,
        SpreadSource.CENTER_ONLY,
        SpreadSource.TRANSLATION,
        SpreadSource.ASSUMED,
    ):
        assert s not in POPULATION_SPREAD_SOURCES


# ---------------------------------------------------------------------------
# CalibrationTargetEstimates wiring
# ---------------------------------------------------------------------------


def test_cal_resolved_spread_source_legacy_fallback():
    # No observed_distribution -> maps from legacy population_spread
    assert _cal_estimates().resolved_spread_source == SpreadSource.CENTER_ONLY
    assert (
        _cal_estimates(population_spread="across_patient").resolved_spread_source
        == SpreadSource.ACROSS_PATIENT
    )


def test_cal_resolved_spread_source_prefers_observed_distribution():
    od = _median_iqr(10.0, 15.0, 25.0, spread_source=SpreadSource.ACROSS_PATIENT, n_biological=42)
    e = _cal_estimates(population_spread="across_patient", observed_distribution=od)
    assert e.resolved_spread_source == SpreadSource.ACROSS_PATIENT
    assert e.observed_distribution.median() == 15.0


def test_cal_observed_distribution_contradiction_rejected():
    od = _median_iqr(10.0, 15.0, 25.0, spread_source=SpreadSource.ACROSS_PATIENT)
    with pytest.raises(ValidationError, match="contradicts"):
        _cal_estimates(population_spread="center_only", observed_distribution=od)


def test_cal_observed_distribution_center_only_consistent():
    od = ObservedDistribution(
        quantiles=[QuantileAnchor(p=0.5, value=15.0)], spread_source=SpreadSource.CENTER_ONLY
    )
    e = _cal_estimates(population_spread="center_only", observed_distribution=od)
    assert e.resolved_spread_source == SpreadSource.CENTER_ONLY


# ---------------------------------------------------------------------------
# ErrorModel (submodel) wiring — additive / backwards compatible
# ---------------------------------------------------------------------------


def test_submodel_observed_distribution_defaults_none():
    em = ErrorModel(name="x", units="nM", sample_size_input="n", observation_code=_DEFAULT_OBS_CODE)
    assert em.observed_distribution is None


def test_submodel_observed_distribution_set():
    od = ObservedDistribution(
        quantiles=[QuantileAnchor(p=0.25, value=1.0), QuantileAnchor(p=0.75, value=3.0)],
        spread_source=SpreadSource.BIOLOGICAL_EXPERIMENTAL,
        n_biological=6,
        experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
    )
    em = ErrorModel(
        name="x",
        units="nM",
        sample_size_input="n",
        observation_code=_DEFAULT_OBS_CODE,
        observed_distribution=od,
    )
    assert em.observed_distribution.feeds_population_spread is True


# ---------------------------------------------------------------------------
# heterogeneity_transfer on SourceRelevanceAssessment
# ---------------------------------------------------------------------------


def test_heterogeneity_transfer_defaults_none():
    s = SourceRelevanceAssessment(**_SOURCE_RELEVANCE)
    assert s.heterogeneity_transfer is None
    assert s.heterogeneity_transfer_justification is None


def test_heterogeneity_transfer_with_justification_ok():
    s = SourceRelevanceAssessment(
        **_SOURCE_RELEVANCE,
        heterogeneity_transfer=HeterogeneityTransfer.MODERATE,
        heterogeneity_transfer_justification="healthy donors; no disease/TME axis captured",
    )
    assert s.heterogeneity_transfer == HeterogeneityTransfer.MODERATE


def test_heterogeneity_transfer_requires_justification():
    with pytest.raises(ValidationError, match="heterogeneity_transfer_justification is required"):
        SourceRelevanceAssessment(
            **_SOURCE_RELEVANCE, heterogeneity_transfer=HeterogeneityTransfer.LOW
        )
