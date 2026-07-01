"""Unit tests for the population-sample primitives (maple.core.calibration.population).

These lock in the across-patient population model used by CalibrationTarget
``distribution_code`` to emit the ``samples`` array for hierarchical inference.
"""

import numpy as np
import pytest

from maple.core.calibration import population as pop
from maple.core.unit_registry import ureg


def _q(x, u="cell / millimeter**2"):
    return x * ureg(u)


def _pct(a):
    return np.median(a), np.percentile(a, 2.5), np.percentile(a, 97.5)


# --------------------------------------------------------------------------- #
# Marginals                                                                   #
# --------------------------------------------------------------------------- #
def test_lognormal_from_median_iqr_recovers_median_and_iqr():
    m = pop.LogNormal.from_median_iqr(_q(50.0), _q(25.0), _q(100.0))
    s = m.sample(np.random.default_rng(0), 200_000).magnitude
    med, _, _ = _pct(s)
    assert med == pytest.approx(50.0, rel=0.02)
    # IQR endpoints come back at the reported quartiles
    assert np.percentile(s, 25) == pytest.approx(25.0, rel=0.03)
    assert np.percentile(s, 75) == pytest.approx(100.0, rel=0.03)


def test_lognormal_from_mean_sd_matches_arithmetic_mean():
    m = pop.LogNormal.from_mean_sd(_q(20.85), _q(4.5))
    s = m.sample(np.random.default_rng(0), 200_000).magnitude
    assert s.mean() == pytest.approx(20.85, rel=0.02)
    assert s.std() == pytest.approx(4.5, rel=0.03)


def test_lognormal_from_median_iqr_width_recovers_iqr():
    # median + IQR *width* (Q3 - Q1); the sample's IQR width should come back.
    m = pop.LogNormal.from_median_iqr_width(_q(140.45), _q(286.84))
    s = m.sample(np.random.default_rng(0), 400_000).magnitude
    assert np.median(s) == pytest.approx(140.45, rel=0.02)
    width = np.percentile(s, 75) - np.percentile(s, 25)
    assert width == pytest.approx(286.84, rel=0.03)


def test_lognormal_from_median_ci95_recovers_ci():
    m = pop.LogNormal.from_median_ci95(_q(60.0), _q(30.0), _q(120.0))
    s = m.sample(np.random.default_rng(0), 400_000).magnitude
    assert np.median(s) == pytest.approx(60.0, rel=0.02)
    assert np.percentile(s, 2.5) == pytest.approx(30.0, rel=0.04)
    assert np.percentile(s, 97.5) == pytest.approx(120.0, rel=0.04)


def test_lognormal_rejects_nonpositive_quartile():
    with pytest.raises(ValueError):
        pop.LogNormal.from_median_iqr(_q(50.0), _q(0.0), _q(100.0))


def test_beta_from_mean_sd_bounds_and_moments():
    m = pop.Beta.from_mean_sd(0.3062 * ureg.dimensionless, 0.168 * ureg.dimensionless)
    s = m.sample(np.random.default_rng(0), 200_000).magnitude
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert s.mean() == pytest.approx(0.3062, rel=0.03)


# --------------------------------------------------------------------------- #
# Correlation model                                                           #
# --------------------------------------------------------------------------- #
def test_rho_fixed_bivariate_hits_target_correlation():
    z = pop.Rho.fixed(-0.25).draw_latents(2, np.random.default_rng(0), 500_000)
    assert np.corrcoef(z[0], z[1])[0, 1] == pytest.approx(-0.25, abs=0.01)


def test_rho_fixed_onefactor_hits_target_correlation():
    z = pop.Rho.fixed(0.5).draw_latents(3, np.random.default_rng(0), 500_000)
    c = np.corrcoef(np.vstack(z))
    off = c[np.triu_indices(3, 1)]
    assert np.allclose(off, 0.5, atol=0.01)


def test_rho_fixed_negative_rho_multiple_components_raises():
    with pytest.raises(ValueError):
        pop.Rho.fixed(-0.3).draw_latents(3, np.random.default_rng(0), 10)


def test_rho_beta_average_correlation_near_prior_mean():
    # Beta(2,2) has mean 0.5; the realised average pairwise correlation should sit near it.
    z = pop.Rho.beta(2, 2).draw_latents(3, np.random.default_rng(0), 500_000)
    c = np.corrcoef(np.vstack(z))
    off = c[np.triu_indices(3, 1)]
    assert off.mean() == pytest.approx(0.5, abs=0.05)


# --------------------------------------------------------------------------- #
# copula_combine                                                              #
# --------------------------------------------------------------------------- #
def _two_lognormals():
    return [
        pop.LogNormal.from_median_iqr(
            _q(30.6, "percent"), _q(24.2, "percent"), _q(41.1, "percent")
        ),
        pop.LogNormal.from_median_iqr(_q(12.1, "percent"), _q(8.2, "percent"), _q(17.4, "percent")),
    ]


def test_fraction_median_invariant_to_rho_but_spread_widens_when_negative():
    rng = np.random.default_rng(0)
    indep = pop.copula_combine(
        _two_lognormals(), "fraction", pop.Rho.independent(), rng=rng, n=300_000
    )
    rng = np.random.default_rng(0)
    neg = pop.copula_combine(
        _two_lognormals(), "fraction", pop.Rho.fixed(-0.25), rng=rng, n=300_000
    )
    # fraction median is set by the marginal medians, ~invariant to rho
    assert np.median(neg).magnitude == pytest.approx(np.median(indep).magnitude, rel=0.02)
    # negative within-component correlation widens the fraction vs independent
    assert np.std(neg.magnitude) > np.std(indep.magnitude)


def test_positive_rho_tightens_a_ratio():
    rng = np.random.default_rng(0)
    indep = pop.copula_combine(
        _two_lognormals(), "ratio", pop.Rho.independent(), rng=rng, n=300_000
    )
    rng = np.random.default_rng(0)
    pos = pop.copula_combine(_two_lognormals(), "ratio", pop.Rho.fixed(0.8), rng=rng, n=300_000)
    assert np.std(pos.magnitude) < np.std(indep.magnitude)


def test_sum_beta_hyperprior_avoids_comonotone_blowup():
    comps = [
        pop.LogNormal.from_median_iqr(_q(3.0), _q(1.0), _q(6.0)),
        pop.LogNormal.from_median_iqr(_q(7.0), _q(2.0), _q(18.0)),
        pop.LogNormal.from_median_iqr(_q(1.0), _q(1.0), _q(3.0)),
    ]
    rng = np.random.default_rng(0)
    beta = pop.copula_combine(comps, "sum", pop.Rho.beta(2, 2), rng=rng, n=300_000)
    rng = np.random.default_rng(0)
    como = pop.copula_combine(comps, "sum", pop.Rho.fixed(1.0), rng=rng, n=300_000)
    cv_beta = np.std(beta.magnitude) / np.mean(beta.magnitude)
    cv_como = np.std(como.magnitude) / np.mean(como.magnitude)
    # the degenerate comonotone corner inflates the sum's CV; the hyperprior stays tamer
    assert cv_beta < cv_como


def test_combine_callable():
    comps = _two_lognormals()
    rng = np.random.default_rng(0)
    out = pop.copula_combine(comps, lambda d: (d[0] - d[1]), pop.Rho.independent(), rng=rng, n=1000)
    assert out.units == ureg("percent")


# --------------------------------------------------------------------------- #
# cohort_mixture / empirical_population / helpers                             #
# --------------------------------------------------------------------------- #
def test_cohort_mixture_equal_weight_spans_cohorts():
    cohorts = [
        pop.LogNormal.from_median_ci95(_q(25.2), _q(3.77), _q(167.9)),
        pop.LogNormal.from_median_ci95(_q(62.9), _q(30.3), _q(131.9)),
        pop.LogNormal.from_median_ci95(_q(138.8), _q(20.3), _q(965.2)),
    ]
    s = pop.cohort_mixture(cohorts, "equal", rng=np.random.default_rng(45), n=300_000).magnitude
    # median near the geometric center of the cohort medians; spread far wider than any one
    assert np.median(s) == pytest.approx(61.0, rel=0.06)
    assert np.percentile(s, 97.5) > 400.0


def test_empirical_population_returns_raw_values():
    vals = [_q(x) for x in (5.0, 7.0, 9.0, 11.0)]
    out = pop.empirical_population(vals)
    assert np.allclose(out.magnitude, [5.0, 7.0, 9.0, 11.0])


def test_empirical_population_resamples_to_n():
    vals = [_q(x) for x in (5.0, 7.0, 9.0, 11.0)]
    out = pop.empirical_population(vals, rng=np.random.default_rng(0), n=1000)
    assert out.size == 1000
    assert set(np.unique(out.magnitude)).issubset({5.0, 7.0, 9.0, 11.0})


def test_midpoint():
    assert (
        pop.midpoint(_q(50.0, "milligram/milliliter"), _q(100.0, "milligram/milliliter")).magnitude
        == 75.0
    )


def test_summarize_1d_and_2d():
    r = pop.summarize(np.arange(1.0, 101.0) * ureg.dimensionless)
    assert r["median_obs"].magnitude == pytest.approx(50.5, rel=0.01)
    assert "samples" in r
    joint = pop.summarize(np.arange(200.0).reshape(100, 2) * ureg.dimensionless)
    assert np.ndim(joint["median_obs"].magnitude) == 1
    assert joint["median_obs"].magnitude.shape == (2,)


def test_bootstrap_median_ci_brackets_point_median():
    vals = [_q(x) for x in (10, 12, 15, 20, 25, 30, 40)]
    med, lo, hi = pop.bootstrap_median(vals, rng=np.random.default_rng(0), n_boot=5000)
    assert lo.magnitude <= med.magnitude <= hi.magnitude
