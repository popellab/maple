"""Population-sample primitives for CalibrationTarget ``distribution_code``.

These helpers build the *across-patient population sample* — the ``samples`` array a
target returns for hierarchical inference (the omega / population-variability signal),
as distinct from the center + measurement uncertainty carried by ``median_obs`` / ci95.

Which primitive a target calls announces its data quality and how much is assumed:

    per-patient data          -> empirical_population   (no assumption; measured)
    one summary               -> Marginal.sample        (distribution shape only)
    several marginals, no joint-> copula_combine         (a correlation assumption)
    several cohort summaries   -> cohort_mixture         (exchangeable-cohort)

The hidden correlation/hyperprior machinery lives here, written once, so a target body
reduces to naming components + picking a correlation model. Genuinely bespoke targets
keep writing free-form ``distribution_code`` and can still use ``Marginal`` / summarize.

All functions take and return Pint quantities (the ``ureg`` injected into
``distribution_code``); ``samples`` may be 1-D (a scalar observable) or 2-D
``(n_patients, k)`` (a joint / compositional / trajectory observable).
"""

from __future__ import annotations

import numpy as np

# Standard-normal quantiles used to back a lognormal sigma out of a reported spread.
_Z_Q = 0.67448975329236258  # 75th percentile  -> IQR half-width
_Z_95 = 1.959963984540054  # 97.5th percentile -> 95% CI half-width


# --------------------------------------------------------------------------- #
# Marginals                                                                    #
# --------------------------------------------------------------------------- #
class Marginal:
    """A per-patient marginal distribution.

    Subclasses implement :meth:`z_sample`, mapping standard-normal draws ``z`` to a
    Pint quantity. Working in ``z`` space is what lets :func:`copula_combine` couple
    components at a target correlation (and lets mixed families — e.g. a lognormal
    density with a beta fraction — share one interface).
    """

    def z_sample(self, z):  # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, rng, n=100_000):
        """Draw ``n`` independent patients from this marginal (a population sample)."""
        return self.z_sample(rng.standard_normal(n))


class LogNormal(Marginal):
    """Lognormal marginal, parameterised in log space (``mu``, ``sigma``)."""

    def __init__(self, mu, sigma, units):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.units = units

    @classmethod
    def from_median_iqr(cls, median, q1, q3):
        """Median-anchored: ``mu = ln(median)``, ``sigma`` from the reported IQR."""
        u = median.units
        m = median.to(u).magnitude
        lo = q1.to(u).magnitude
        hi = q3.to(u).magnitude
        if lo <= 0:
            raise ValueError(f"LogNormal.from_median_iqr needs q1 > 0 (got {lo})")
        sigma = np.log(hi / lo) / (2.0 * _Z_Q)
        return cls(np.log(m), sigma, u)

    @classmethod
    def from_median_ci95(cls, median, lo, hi):
        """Median-anchored: ``sigma`` from a (symmetric-in-log) 95% CI."""
        u = median.units
        m = median.to(u).magnitude
        a = lo.to(u).magnitude
        b = hi.to(u).magnitude
        if a <= 0:
            raise ValueError(f"LogNormal.from_median_ci95 needs ci_lower > 0 (got {a})")
        sigma = np.log(b / a) / (2.0 * _Z_95)
        return cls(np.log(m), sigma, u)

    @classmethod
    def from_mean_sd(cls, mean, sd):
        """Mean-anchored (moment-matched): median = mean * exp(-sigma^2 / 2)."""
        u = mean.units
        m = mean.to(u).magnitude
        s = sd.to(u).magnitude
        if m <= 0:
            raise ValueError(f"LogNormal.from_mean_sd needs mean > 0 (got {m})")
        sigma2 = np.log(1.0 + (s / m) ** 2)
        return cls(np.log(m) - 0.5 * sigma2, np.sqrt(sigma2), u)

    def z_sample(self, z):
        return np.exp(self.mu + self.sigma * np.asarray(z)) * self.units


class Beta(Marginal):
    """Beta marginal on [0, 1] for bounded fractions.

    ``z_sample`` uses the Gaussian copula transform ``beta.ppf(Phi(z))`` so a Beta
    component can be coupled to lognormal components in :func:`copula_combine`.
    """

    def __init__(self, a, b, units):
        self.a = float(a)
        self.b = float(b)
        self.units = units

    @classmethod
    def from_mean_sd(cls, mean, sd, units=None):
        """Method-of-moments Beta from a mean and SD on [0, 1]."""
        m = mean.to("dimensionless").magnitude if hasattr(mean, "to") else float(mean)
        s = sd.to("dimensionless").magnitude if hasattr(sd, "to") else float(sd)
        if not (0.0 < m < 1.0):
            raise ValueError(f"Beta.from_mean_sd needs 0 < mean < 1 (got {m})")
        var = s**2
        common = m * (1.0 - m) / var - 1.0
        a = max(m * common, 1e-3)
        b = max((1.0 - m) * common, 1e-3)
        if units is None:
            import pint

            units = pint.get_application_registry().dimensionless
        return cls(a, b, units)

    def z_sample(self, z):
        from scipy.stats import beta, norm

        u = norm.cdf(np.asarray(z))
        return beta.ppf(u, self.a, self.b) * self.units


# --------------------------------------------------------------------------- #
# Correlation model                                                           #
# --------------------------------------------------------------------------- #
class Rho:
    """Within-patient correlation model for combining several marginals.

    ``rho`` is an observation-model assumption (it shapes the observed conditioning
    value, never the simulator). Construct with :meth:`independent`, :meth:`fixed`
    (a measured or assumed point value), or :meth:`beta` (a Beta(a,b) hyperprior on
    [0, 1] marginalised out — the honest treatment of an unidentified nuisance, and its
    vanishing density at rho=1 keeps the degenerate comonotone corner out of sums).
    """

    def __init__(self, kind, *, value=None, a=None, b=None):
        self.kind = kind
        self.value = value
        self.a = a
        self.b = b

    @classmethod
    def independent(cls):
        return cls("independent")

    @classmethod
    def fixed(cls, value):
        if not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"Rho.fixed needs value in [-1, 1] (got {value})")
        return cls("fixed", value=float(value))

    @classmethod
    def beta(cls, a, b):
        return cls("beta", a=float(a), b=float(b))

    def draw_latents(self, k, rng, n):
        """Return ``k`` standard-normal vectors of length ``n`` with correlation rho.

        - independent: rho = 0.
        - fixed, k == 2: exact bivariate ``z_b = rho*z_a + sqrt(1-rho^2)*eps`` (any sign).
        - fixed, k > 2: one-factor ``z_i = sqrt(rho)*F + sqrt(1-rho)*eps_i`` (rho >= 0;
          negative rho among >2 components has no PSD one-factor form and raises).
        - beta: per-sample rho ~ Beta(a,b) on [0,1], one-factor.
        """
        if self.kind == "independent":
            return [rng.standard_normal(n) for _ in range(k)]

        if self.kind == "fixed":
            rho = self.value
            if k == 2:
                za = rng.standard_normal(n)
                zb = rho * za + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
                return [za, zb]
            if rho < 0:
                raise ValueError(
                    "Rho.fixed with negative rho is only defined for 2 components; "
                    f"got {k}. A general negative correlation needs a full matrix."
                )
            sr, sr1 = np.sqrt(rho), np.sqrt(1.0 - rho)
            F = rng.standard_normal(n)
            return [sr * F + sr1 * rng.standard_normal(n) for _ in range(k)]

        if self.kind == "beta":
            rho = rng.beta(self.a, self.b, n)
            sr, sr1 = np.sqrt(rho), np.sqrt(1.0 - rho)
            F = rng.standard_normal(n)
            return [sr * F + sr1 * rng.standard_normal(n) for _ in range(k)]

        raise ValueError(f"unknown Rho kind {self.kind!r}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Combine primitives                                                          #
# --------------------------------------------------------------------------- #
def copula_combine(components, combine, correlation, *, rng, n=100_000):
    """Combine several :class:`Marginal` components at a within-patient correlation.

    Parameters
    ----------
    components : list[Marginal]
    combine : {'sum', 'fraction', 'ratio'} or callable
        ``'sum'``      -> sum of all components
        ``'fraction'`` -> components[0] / sum(components)   (fraction of a total)
        ``'ratio'``    -> components[0] / components[1]      (two-pole ratio)
        callable       -> ``combine(list_of_draws)`` for a bespoke combination
    correlation : Rho
    """
    z = correlation.draw_latents(len(components), rng, n)
    draws = [c.z_sample(z[i]) for i, c in enumerate(components)]
    if callable(combine):
        return combine(draws)
    if combine == "sum":
        return sum(draws[1:], draws[0])
    if combine == "fraction":
        total = sum(draws[1:], draws[0])
        return (draws[0] / total).to("dimensionless")
    if combine == "ratio":
        return (draws[0] / draws[1]).to("dimensionless")
    raise ValueError(f"unknown combine {combine!r}")


def cohort_mixture(cohorts, weights="equal", *, rng, n=300_000):
    """Pool several cohort patient-distributions into one across-patient population.

    ``cohorts`` are :class:`Marginal` (each a within-cohort patient distribution).
    ``weights`` is ``'equal'`` (each cohort an exchangeable population sample — the
    default; inverse-variance weighting would suppress wide cohorts and n-weighting lets
    the largest cohort's center dominate) or an explicit sequence of mixing weights.
    """
    k = len(cohorts)
    if weights == "equal":
        p = np.full(k, 1.0 / k)
    else:
        p = np.asarray(weights, dtype=float)
        p = p / p.sum()
    comp = rng.choice(k, size=n, p=p)
    z = rng.standard_normal(n)
    units = cohorts[0].z_sample(np.zeros(1)).units
    out = np.empty(n)
    for i, coh in enumerate(cohorts):
        mask = comp == i
        out[mask] = coh.z_sample(z[mask]).to(units).magnitude
    return out * units


def empirical_population(values, *, rng=None, n=None):
    """The across-patient population *is* the per-patient data (no assumption).

    ``values`` is a list of Pint quantities (or a Pint array) of per-patient
    measurements. Returned as-is, or resampled with replacement to length ``n`` (needs
    ``rng``) when a fixed sample size is wanted.
    """
    arr = _as_pint_array(values)
    if n is None or n == len(arr):
        return arr
    if rng is None:
        raise ValueError("empirical_population needs rng to resample to n")
    idx = rng.integers(0, len(arr), n)
    return arr[idx]


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def midpoint(low, high):
    """Center of a nuisance range, held fixed so it contributes no population spread."""
    return 0.5 * (low + high)


def summarize(samples):
    """Reduce a population sample to the ``derive_distribution`` return dict.

    ``samples`` may be 1-D (scalar observable) or 2-D ``(n_patients, k)`` (joint); the
    median / ci95 reduce over the patient axis (axis 0).
    """
    axis = 0 if getattr(samples, "ndim", 1) > 1 else None
    return {
        "median_obs": np.median(samples, axis=axis),
        "ci95_lower": np.percentile(samples, 2.5, axis=axis),
        "ci95_upper": np.percentile(samples, 97.5, axis=axis),
        "samples": samples,
    }


def bootstrap_median(values, *, rng, n_boot=10_000):
    """Center + its CI for per-patient data: point median with a bootstrap-median CI.

    Returns ``(median_obs, ci95_lower, ci95_upper)`` — the center summary that pairs
    with :func:`empirical_population` (which supplies the population spread).
    """
    arr = _as_pint_array(values)
    m = arr.magnitude
    boots = np.array([np.median(rng.choice(m, size=len(m), replace=True)) for _ in range(n_boot)])
    return (
        np.median(arr),
        np.percentile(boots, 2.5) * arr.units,
        np.percentile(boots, 97.5) * arr.units,
    )


def _as_pint_array(values):
    """Coerce a list of Pint quantities (or an already-array quantity) to a Pint array."""
    if hasattr(values, "magnitude") and np.ndim(values.magnitude) >= 1:
        return values
    units = values[0].units
    return np.array([v.to(units).magnitude for v in values]) * units
