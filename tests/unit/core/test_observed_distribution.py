#!/usr/bin/env python3
"""Tests for the reported-statistics variability layer.

Covers ReportedStatistic / ObservedDistribution validation, the on-demand
quantile derivations, unit provenance, SpreadSource routing, and the wiring into
CalibrationTargetEstimates and ErrorModel.
"""

import pytest
from pydantic import ValidationError

from maple.core.calibration.shared_models import (
    QuantileConvention,
    DistributionShape,
    ExperimentalUnitType,
    ObservedDistribution,
    POPULATION_SPREAD_SOURCES,
    ReportedStatistic,
    SourceRelevanceAssessment,
    SpreadSource,
    StatKind,
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

_DEFAULT_OBS_CODE = (
    "def derive_observation(inputs, sample_size, rng, n_bootstrap):\n"
    "    return rng.normal(0.0, 1.0, n_bootstrap)"
)

_Z_Q = 0.6744897501960817
_Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(p, value):
    return ReportedStatistic(stat=StatKind.QUANTILE, p=p, value=value)


def _s(stat, value):
    return ReportedStatistic(stat=stat, value=value)


def _median_iqr(q25, q50, q75, **kwargs):
    """A median with quartiles. Injects unit provenance for population sources."""
    if kwargs.get("spread_source") in POPULATION_SPREAD_SOURCES:
        kwargs.setdefault("n_biological", 42)
        kwargs.setdefault("experimental_unit_type", ExperimentalUnitType.BIOLOGICAL)
    return ObservedDistribution(statistics=[_q(0.25, q25), _q(0.5, q50), _q(0.75, q75)], **kwargs)


def _center_scale(center, scale, shape, stat=StatKind.SD, center_stat="mean", **kwargs):
    stats = [
        _q(0.5, center) if center_stat == "median" else _s(StatKind.MEAN, center),
        _s(stat, scale),
    ]
    return ObservedDistribution(statistics=stats, shape=shape, **kwargs)


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
# ReportedStatistic
# ---------------------------------------------------------------------------


def test_quantile_needs_p():
    with pytest.raises(ValidationError, match="needs a probability level"):
        ReportedStatistic(stat=StatKind.QUANTILE, value=1.0)


@pytest.mark.parametrize("bad_p", [0.0, 1.0, -0.1, 1.5])
def test_quantile_p_must_be_in_open_unit_interval(bad_p):
    with pytest.raises(ValidationError, match="p must be in"):
        _q(bad_p, 1.0)


def test_non_quantile_must_not_set_p():
    with pytest.raises(ValidationError, match="must not set p"):
        ReportedStatistic(stat=StatKind.SD, value=1.0, p=0.5)


def test_negative_width_rejected():
    with pytest.raises(ValidationError, match="cannot be negative"):
        _s(StatKind.IQR, -1.0)


def test_negative_location_is_fine():
    """A mean may legitimately be negative; only widths may not."""
    assert _s(StatKind.MEAN, -3.0).value == -3.0


# ---------------------------------------------------------------------------
# ObservedDistribution validation
# ---------------------------------------------------------------------------


def test_empty_statistics_rejected():
    with pytest.raises(ValidationError, match="at least one entry"):
        ObservedDistribution(statistics=[], spread_source=SpreadSource.CENTER_ONLY)


def test_duplicate_statistic_rejected():
    with pytest.raises(ValidationError, match="repeats statistic"):
        ObservedDistribution(
            statistics=[_s(StatKind.SD, 1.0), _s(StatKind.SD, 2.0)],
            spread_source=SpreadSource.CENTER_ONLY,
        )


def test_duplicate_quantile_level_rejected():
    with pytest.raises(ValidationError, match="repeats statistic"):
        ObservedDistribution(
            statistics=[_q(0.5, 1.0), _q(0.5, 2.0)], spread_source=SpreadSource.CENTER_ONLY
        )


def test_same_stat_at_different_p_is_fine():
    d = ObservedDistribution(
        statistics=[_q(0.25, 1.0), _q(0.75, 3.0)], spread_source=SpreadSource.CENTER_ONLY
    )
    assert len(d.statistics) == 2


def test_crossing_quantile_function_rejected():
    with pytest.raises(ValidationError, match="non-decreasing"):
        ObservedDistribution(
            statistics=[_q(0.25, 20.0), _q(0.75, 10.0)], spread_source=SpreadSource.CENTER_ONLY
        )


def test_population_spread_requires_a_width():
    with pytest.raises(ValidationError, match="no width statistic"):
        ObservedDistribution(
            statistics=[_q(0.5, 1.0)],
            spread_source=SpreadSource.ACROSS_PATIENT,
            n_biological=10,
            experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
        )


def test_scalar_width_satisfies_the_population_spread_requirement():
    """A median plus an SD is a width, even with no quartiles."""
    d = ObservedDistribution(
        statistics=[_q(0.5, 10.0), _s(StatKind.SD, 2.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        n_biological=10,
        experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
    )
    assert d.feeds_population_spread is True


def test_center_only_single_statistic_allowed():
    d = ObservedDistribution(statistics=[_q(0.5, 3.0)], spread_source=SpreadSource.CENTER_ONLY)
    assert d.median() == 3.0


def test_median_sd_and_se_together_are_representable():
    """The combination a single center+scale could not express."""
    d = ObservedDistribution(
        statistics=[_q(0.5, 10.0), _s(StatKind.SD, 2.0), _s(StatKind.SE, 0.6)],
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.get(StatKind.SD) == 2.0
    assert d.get(StatKind.SE) == 0.6
    assert d.get(StatKind.QUANTILE, 0.5) == 10.0


# ---------------------------------------------------------------------------
# Unit provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", [ExperimentalUnitType.TECHNICAL, ExperimentalUnitType.CLONAL])
def test_technical_unit_cannot_be_population_spread(unit):
    d = _median_iqr(
        10.0,
        15.0,
        25.0,
        spread_source=SpreadSource.BIOLOGICAL_EXPERIMENTAL,
        experimental_unit_type=unit,
    )
    with pytest.raises(ValueError, match="not population variability"):
        d.require_unit_provenance()


def test_population_spread_requires_n_biological():
    d = ObservedDistribution(
        statistics=[_q(0.25, 10.0), _q(0.75, 20.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
    )
    with pytest.raises(ValueError, match="n_biological is not set"):
        d.require_unit_provenance()


def test_population_spread_requires_experimental_unit_type():
    d = ObservedDistribution(
        statistics=[_q(0.25, 10.0), _q(0.75, 20.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        n_biological=8,
    )
    with pytest.raises(ValueError, match="experimental_unit_type is not set"):
        d.require_unit_provenance()


def test_center_only_exempt_from_unit_provenance():
    d = ObservedDistribution(statistics=[_q(0.5, 3.0)], spread_source=SpreadSource.CENTER_ONLY)
    d.require_unit_provenance()  # does not raise


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


def test_median_and_iqr_from_quantiles():
    d = _median_iqr(10.0, 15.0, 25.0, spread_source=SpreadSource.ACROSS_PATIENT)
    assert d.median() == 15.0
    assert d.iqr() == 15.0


def test_reported_iqr_wins_over_derivation():
    d = ObservedDistribution(
        statistics=[_q(0.5, 10.0), _s(StatKind.IQR, 4.0)],
        spread_source=SpreadSource.CENTER_ONLY,
        shape=DistributionShape.NORMAL,
    )
    assert d.iqr() == 4.0


def test_quantile_interpolation_and_clamping():
    d = _median_iqr(10.0, 15.0, 25.0, spread_source=SpreadSource.ACROSS_PATIENT)
    assert d.quantile(0.375) == pytest.approx(12.5)
    assert d.quantile(0.01) == 10.0  # clamped to the lowest anchor
    assert d.quantile(0.99) == 25.0  # clamped to the highest


def test_iqr_none_when_range_not_spanned():
    d = ObservedDistribution(
        statistics=[_q(0.4, 1.0), _q(0.6, 2.0)], spread_source=SpreadSource.CENTER_ONLY
    )
    assert d.iqr() is None


def test_center_prefers_median_over_mean():
    d = ObservedDistribution(
        statistics=[_q(0.5, 3.0), _s(StatKind.MEAN, 5.0)],
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.center() == 3.0


# ---------------------------------------------------------------------------
# Expansion from a center and a scale
# ---------------------------------------------------------------------------


def test_normal_expansion_is_symmetric():
    d = _center_scale(10.0, 2.0, DistributionShape.NORMAL, spread_source=SpreadSource.CENTER_ONLY)
    assert d.median() == pytest.approx(10.0)
    assert d.quantile(0.75) - d.median() == pytest.approx(d.median() - d.quantile(0.25))
    assert d.iqr() == pytest.approx(2 * _Z_Q * 2.0)


def test_lognormal_expansion_is_log_symmetric():
    d = _center_scale(
        132.3, 132.1, DistributionShape.LOGNORMAL, spread_source=SpreadSource.CENTER_ONLY
    )
    # q25 * q75 == median^2 for a lognormal.
    assert d.quantile(0.25) * d.quantile(0.75) == pytest.approx(d.median() ** 2)
    # A mean-anchored lognormal has its median below the mean.
    assert d.median() < 132.3


def test_lognormal_median_anchored_keeps_the_median():
    d = _center_scale(
        50.0,
        20.0,
        DistributionShape.LOGNORMAL,
        center_stat="median",
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.median() == pytest.approx(50.0)


def test_logit_normal_stays_in_the_unit_interval():
    d = _center_scale(
        0.95,
        0.10,
        DistributionShape.LOGIT_NORMAL,
        center_stat="median",
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert 0.0 < d.quantile(0.25) < d.quantile(0.75) < 1.0


def test_logit_normal_center_must_be_in_the_unit_interval():
    d = _center_scale(
        1.5,
        0.1,
        DistributionShape.LOGIT_NORMAL,
        center_stat="median",
        spread_source=SpreadSource.CENTER_ONLY,
    )
    with pytest.raises(ValueError, match="center in"):
        d.quantile(0.25)


def test_iqr_scale_expands_through_the_normal_equivalent_sd():
    d = ObservedDistribution(
        statistics=[_q(0.5, 15.0), _s(StatKind.IQR, 10.0)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.quantile(0.75) - d.quantile(0.25) == pytest.approx(10.0)


def test_cv_scale_uses_the_center():
    d = ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 100.0), _s(StatKind.CV, 0.5)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.iqr() == pytest.approx(2 * _Z_Q * 50.0)


def test_ci95_bounds_give_a_scale():
    d = ObservedDistribution(
        statistics=[
            _s(StatKind.MEAN, 10.0),
            _s(StatKind.CI95_LO, 6.0),
            _s(StatKind.CI95_HI, 14.0),
        ],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.iqr() == pytest.approx(2 * _Z_Q * (8.0 / (2 * _Z_95)))


def test_expansion_needs_a_shape():
    d = ObservedDistribution(
        statistics=[_q(0.5, 10.0), _s(StatKind.SD, 2.0)],
        spread_source=SpreadSource.CENTER_ONLY,
    )
    with pytest.raises(ValueError, match="needs `shape`"):
        d.quantile(0.25)


def test_expansion_needs_a_width():
    d = ObservedDistribution(
        statistics=[_q(0.5, 10.0)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    with pytest.raises(ValueError, match="no width"):
        d.quantile(0.25)


def test_expansion_needs_a_center():
    d = ObservedDistribution(
        statistics=[_s(StatKind.SD, 2.0)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    with pytest.raises(ValueError, match="no center"):
        d.quantile(0.25)


def test_reported_quantiles_win_over_expansion():
    """Explicit quartiles are used as printed, not re-derived from the SD."""
    d = ObservedDistribution(
        statistics=[_q(0.25, 1.0), _q(0.5, 2.0), _q(0.75, 9.0), _s(StatKind.SD, 0.1)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert d.iqr() == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# SpreadSource routing
# ---------------------------------------------------------------------------


def test_population_spread_sources_membership():
    for s in (SpreadSource.ACROSS_PATIENT, SpreadSource.BIOLOGICAL_EXPERIMENTAL):
        assert s in POPULATION_SPREAD_SOURCES
    for s in (
        SpreadSource.CENTER_ONLY,
        SpreadSource.TECHNICAL,
        SpreadSource.TRANSLATION,
        SpreadSource.ASSUMED,
    ):
        assert s not in POPULATION_SPREAD_SOURCES


# ---------------------------------------------------------------------------
# CalibrationTargetEstimates wiring
# ---------------------------------------------------------------------------


def test_cal_no_observed_distribution_is_center_only():
    e = _cal_estimates()
    assert e.resolved_spread_source == SpreadSource.CENTER_ONLY
    assert e.feeds_population_spread is False


def test_cal_spread_source_comes_from_observed_distribution():
    # The cohort carries the unit accounting on the calibration side, so none is set here.
    od = ObservedDistribution(
        statistics=[_q(0.25, 10.0), _q(0.5, 15.0), _q(0.75, 25.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
    )
    e = _cal_estimates(observed_distribution=od)
    assert e.resolved_spread_source == SpreadSource.ACROSS_PATIENT
    assert e.feeds_population_spread is True
    assert e.observed_distribution.median() == 15.0


def test_cal_center_only_observed_distribution_does_not_feed_omega():
    od = ObservedDistribution(statistics=[_q(0.5, 15.0)], spread_source=SpreadSource.CENTER_ONLY)
    e = _cal_estimates(observed_distribution=od)
    assert e.resolved_spread_source == SpreadSource.CENTER_ONLY
    assert e.feeds_population_spread is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_biological", 42),
        ("n_technical", 3),
        ("experimental_unit_type", ExperimentalUnitType.BIOLOGICAL),
        ("unit_group", "donors"),
        ("n_biological_is_floor", True),
    ],
)
def test_cal_rejects_unit_accounting_the_cohort_owns(field, value):
    od = ObservedDistribution(
        statistics=[_q(0.25, 10.0), _q(0.5, 15.0), _q(0.75, 25.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        **{field: value},
    )
    with pytest.raises(ValidationError, match="cohort"):
        _cal_estimates(observed_distribution=od)


# ---------------------------------------------------------------------------
# ErrorModel (submodel) wiring
# ---------------------------------------------------------------------------


def test_submodel_observed_distribution_defaults_none():
    em = ErrorModel(name="x", units="nM", sample_size_input="n", observation_code=_DEFAULT_OBS_CODE)
    assert em.observed_distribution is None


def test_submodel_observed_distribution_set():
    od = ObservedDistribution(
        statistics=[_q(0.25, 1.0), _q(0.75, 3.0)],
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


def test_submodel_population_spread_must_state_its_units():
    """The requirement moved off ObservedDistribution onto the owner that carries n."""
    from maple.core.calibration.submodel_target import Calibration

    od = ObservedDistribution(
        statistics=[_q(0.25, 1.0), _q(0.75, 3.0)],
        spread_source=SpreadSource.BIOLOGICAL_EXPERIMENTAL,
    )
    em = ErrorModel(
        name="d1",
        units="nM",
        sample_size_input="n",
        observation_code=_DEFAULT_OBS_CODE,
        observed_distribution=od,
    )
    cal = Calibration.model_construct(error_model=[em])
    with pytest.raises(ValueError, match="n_biological is not set"):
        cal._spread_source_states_its_units()


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


# ---------------------------------------------------------------------------
# logit_normal shape on a bounded observable
# ---------------------------------------------------------------------------


def test_bounded_units_reject_a_normal_shape():
    od = ObservedDistribution(
        statistics=[_q(0.5, 0.5), _s(StatKind.SD, 0.1)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    with pytest.raises(ValidationError, match="logit_normal"):
        _cal_estimates(units="percent", observed_distribution=od)


def test_bounded_units_accept_logit_normal():
    od = ObservedDistribution(
        statistics=[_q(0.5, 0.5), _s(StatKind.SD, 0.1)],
        shape=DistributionShape.LOGIT_NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert _cal_estimates(units="percent", observed_distribution=od) is not None


def test_no_declared_shape_is_exempt():
    """Reported quantiles are used as printed, so nothing is expanded."""
    od = ObservedDistribution(
        statistics=[_q(0.25, 0.4), _q(0.5, 0.5), _q(0.75, 0.6)],
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert _cal_estimates(units="percent", observed_distribution=od) is not None


# ---------------------------------------------------------------------------
# unit_group (shared-biological-unit panels, submodel side)
# ---------------------------------------------------------------------------


def _od(
    n_bio, group, unit=ExperimentalUnitType.BIOLOGICAL, src=SpreadSource.BIOLOGICAL_EXPERIMENTAL
):
    return ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 100.0), _s(StatKind.SD, 30.0)],
        shape=DistributionShape.NORMAL,
        spread_source=src,
        n_biological=n_bio,
        experimental_unit_type=unit,
        unit_group=group,
    )


def _em(name, od):
    return ErrorModel(
        name=name,
        units="pg/mL",
        sample_size_input="n",
        observation_code=_DEFAULT_OBS_CODE,
        observed_distribution=od,
    )


def _od_center(group):
    # Does NOT feed omega, so it is exempt from the conditional-presence rule.
    return ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 100.0), _s(StatKind.SD, 30.0)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
        unit_group=group,
    )


def _run_unit_group_check(ems):
    from maple.core.calibration.submodel_target import Calibration

    Calibration.model_construct(error_model=ems)._unit_groups_consistent()


def test_unit_group_defaults_none():
    od = ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 1.0), _s(StatKind.SD, 1.0)],
        shape=DistributionShape.NORMAL,
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert od.unit_group is None


def test_unit_group_consistent_members_ok():
    _run_unit_group_check([_em("d1", _od(13, "donors")), _em("d2", _od(13, "donors"))])


def test_unit_group_unbalanced_n_allowed():
    # A donor missing at some doses: one population viewed several times.
    _run_unit_group_check([_em("d1", _od(13, "donors")), _em("d2", _od(11, "donors"))])


def test_unit_group_separate_groups_may_differ():
    _run_unit_group_check([_em("d1", _od(13, "lineA")), _em("d2", _od(20, "lineB"))])


def test_unit_group_required_when_multiple_population_entries():
    with pytest.raises(ValueError, match="leave unit_group unset"):
        _run_unit_group_check([_em("d1", _od(13, None)), _em("d2", _od(13, None))])


def test_unit_group_multiple_population_tagged_same_ok():
    _run_unit_group_check([_em("d1", _od(13, "donors")), _em("d2", _od(13, "donors"))])


def test_unit_group_multiple_population_tagged_different_ok():
    _run_unit_group_check([_em("d1", _od(13, "lineA")), _em("d2", _od(20, "lineB"))])


def test_unit_group_single_population_entry_untagged_ok():
    _run_unit_group_check([_em("d1", _od(13, None)), _em("c1", _od_center(None))])


def test_unit_group_center_only_entries_exempt_from_presence():
    _run_unit_group_check([_em("c1", _od_center(None)), _em("c2", _od_center(None))])


def test_unit_group_mismatched_spread_source_rejected():
    with pytest.raises(ValueError, match="disagree on 'spread_source'"):
        _run_unit_group_check(
            [
                _em("d1", _od(13, "g")),
                _em(
                    "d2",
                    _od(13, "g", unit=ExperimentalUnitType.TECHNICAL, src=SpreadSource.TECHNICAL),
                ),
            ]
        )


# ---------------------------------------------------------------------------
# A standard error is a width per sqrt(n)
# ---------------------------------------------------------------------------


def test_se_satisfies_the_population_width_requirement():
    """An SE determines a sample width given n, so it is not a center-only spread."""
    d = ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 9.84), _s(StatKind.SE, 1.16)],
        spread_source=SpreadSource.ACROSS_PATIENT,
    )
    assert d.feeds_population_spread


def test_se_widens_to_a_sample_sd_only_with_n():
    d = ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 9.84), _s(StatKind.SE, 1.16)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        shape=DistributionShape.LOGNORMAL,
    )
    assert d.population_sd() is None
    assert d.population_sd(40) == pytest.approx(1.16 * 40**0.5)


def test_quantiles_from_an_se_need_n_and_say_so():
    d = ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 9.84), _s(StatKind.SE, 1.16)],
        spread_source=SpreadSource.ACROSS_PATIENT,
        shape=DistributionShape.LOGNORMAL,
    )
    with pytest.raises(ValueError, match="pass n"):
        d.median()
    # With n the SE widens and the lognormal stays log-symmetric about its median.
    assert d.quantile(0.25, 40) * d.quantile(0.75, 40) == pytest.approx(d.median(40) ** 2)


def test_a_reported_sd_is_preferred_over_an_se():
    """Both reported: the sample width is the printed one, not the widened SE."""
    d = ObservedDistribution(
        statistics=[_s(StatKind.MEAN, 10.0), _s(StatKind.SD, 4.0), _s(StatKind.SE, 1.0)],
        spread_source=SpreadSource.ACROSS_PATIENT,
    )
    assert d.population_sd(100) == 4.0


def test_negative_se_is_rejected():
    with pytest.raises(ValidationError, match="cannot be negative"):
        ReportedStatistic(stat=StatKind.SE, value=-1.0)


def test_quantile_convention_defaults_to_unrecorded():
    """Papers do not state it, so the schema must not invent one."""
    d = _median_iqr(1.0, 2.0, 3.0, spread_source=SpreadSource.ACROSS_PATIENT)
    assert d.quantile_convention is None


def test_quantile_convention_records_the_estimator():
    d = _median_iqr(
        1.0,
        2.0,
        3.0,
        spread_source=SpreadSource.ACROSS_PATIENT,
        quantile_convention=QuantileConvention.TYPE6,
    )
    assert d.quantile_convention is QuantileConvention.TYPE6


def test_quantile_convention_needs_a_quantile():
    with pytest.raises(ValidationError, match="no quantile was reported"):
        ObservedDistribution(
            statistics=[_s(StatKind.MEAN, 10.0), _s(StatKind.SD, 4.0)],
            spread_source=SpreadSource.ACROSS_PATIENT,
            quantile_convention=QuantileConvention.TYPE7,
        )


def test_quantile_convention_rejects_an_unknown_type():
    with pytest.raises(ValidationError):
        _median_iqr(
            1.0, 2.0, 3.0, spread_source=SpreadSource.ACROSS_PATIENT, quantile_convention="type99"
        )
