"""
Convert SubmodelTarget YAMLs to prior distributions for SBI.

Pipeline:
  1. Load YAML, run observation_code bootstrap -> parameter samples
  2. Fit candidate distributions (lognormal, gamma, inv-gamma), pick best by AIC
  3. Apply deterministic translation sigma inflation from source_relevance fields
  4. Output prior specification with chosen distribution

Translation sigma inflation rubric (8 axes, additive in quadrature, floor=0.15):
  indication_match:        exact=0, related=0.2, proxy=0.5, unrelated=1.0
  species mismatch:        same=0, different=0.3
  source_quality:          primary_human_clinical=0, primary_human_in_vitro=0.1,
                           primary_animal_in_vivo=0.3, primary_animal_in_vitro=0.4,
                           review_article=0.2, textbook=0.3, non_peer_reviewed=0.5
  perturbation_type:       pathological_state=0, physiological_baseline=0.1,
                           pharmacological=0.25, genetic_perturbation=0.4
  tme_compatibility:       high=0, moderate=0.15, low=0.5
  measurement_directness:  direct=0, single_inversion=0.15,
                           steady_state_inversion=0.3, proxy_observable=0.5
  temporal_resolution:     timecourse=0, endpoint_pair=0.1,
                           snapshot_or_equilibrium=0.2
  experimental_system:     clinical_in_vivo=0, animal_in_vivo=0.15, ex_vivo=0.1,
                           in_vitro_coculture=0.15, in_vitro_primary=0.2,
                           in_vitro_cell_line=0.3

These add in quadrature: sigma_total = max(sqrt(sigma_data^2 + sum(sigma_i^2)), 0.15)

Usage (as library)::

    from maple.core.calibration.yaml_to_prior import process_yaml
    result = process_yaml(Path("target.yaml"))

Usage (CLI)::

    python -m maple.core.calibration.yaml_to_prior target.yaml
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from maple.core.calibration.submodel_target import SubmodelTarget


# ============================================================================
# Translation sigma inflation rubric
# ============================================================================

FLOOR_SIGMA = 0.15  # irreducible model-abstraction uncertainty

INDICATION_SIGMA = {
    "exact": 0.0,
    "related": 0.2,
    "proxy": 0.5,
    "unrelated": 1.0,
}

SPECIES_MISMATCH_SIGMA = 0.3  # added when source != target

SOURCE_QUALITY_SIGMA = {
    "primary_human_clinical": 0.0,
    "primary_human_in_vitro": 0.1,
    "primary_animal_in_vivo": 0.3,
    "primary_animal_in_vitro": 0.4,
    "review_article": 0.2,
    "textbook": 0.3,
    "non_peer_reviewed": 0.5,
}

TME_SIGMA = {
    "high": 0.0,
    "moderate": 0.15,
    "low": 0.5,
}

PERTURBATION_SIGMA = {
    "pathological_state": 0.0,
    "physiological_baseline": 0.1,
    "pharmacological": 0.25,
    "genetic_perturbation": 0.4,
}

DIRECTNESS_SIGMA = {
    "direct": 0.0,
    "single_inversion": 0.15,
    "steady_state_inversion": 0.3,
    "proxy_observable": 0.5,
}

TEMPORAL_SIGMA = {
    "timecourse": 0.0,
    "endpoint_pair": 0.1,
    "snapshot_or_equilibrium": 0.2,
}

SYSTEM_SIGMA = {
    "clinical_in_vivo": 0.0,
    "animal_in_vivo": 0.15,
    "ex_vivo": 0.1,
    "in_vitro_coculture": 0.15,
    "in_vitro_primary": 0.2,
    "in_vitro_cell_line": 0.3,
}


def compute_translation_sigma(sr) -> tuple[float, dict[str, float]]:
    """Compute translation sigma inflation from source_relevance fields.

    Returns (total_sigma, breakdown_dict) where breakdown shows each component.
    Components add in quadrature with a floor of FLOOR_SIGMA.
    """
    components = {}

    components["indication"] = INDICATION_SIGMA.get(sr.indication_match.value, 0.5)

    if sr.species_source != sr.species_target:
        components["species"] = SPECIES_MISMATCH_SIGMA
    else:
        components["species"] = 0.0

    components["quality"] = SOURCE_QUALITY_SIGMA.get(sr.source_quality.value, 0.3)

    components["tme"] = TME_SIGMA.get(sr.tme_compatibility.value, 0.15)

    components["perturbation"] = PERTURBATION_SIGMA.get(sr.perturbation_type.value, 0.2)

    # New axes (None-safe for YAMLs that haven't been updated yet)
    if sr.measurement_directness is not None:
        components["directness"] = DIRECTNESS_SIGMA.get(sr.measurement_directness.value, 0.15)

    if sr.temporal_resolution is not None:
        components["temporal"] = TEMPORAL_SIGMA.get(sr.temporal_resolution.value, 0.1)

    if sr.experimental_system is not None:
        components["system"] = SYSTEM_SIGMA.get(sr.experimental_system.value, 0.15)

    raw = np.sqrt(sum(v**2 for v in components.values()))
    total = max(raw, FLOOR_SIGMA)
    return total, components


@dataclass
class DistFit:
    """Result of fitting a single distribution to samples."""

    name: str  # e.g. "lognormal", "gamma", "invgamma"
    params: dict  # distribution-specific parameters
    aic: float  # Akaike information criterion (lower = better)
    ad_stat: float  # Anderson-Darling statistic (absolute GOF)
    ad_crit_5pct: float  # AD critical value at 5% significance
    ad_pass: bool  # AD stat < critical value at 5%
    median: float  # fitted median
    cv: float  # fitted CV


def _ad_test_samples(
    samples: np.ndarray, cdf_func, dist_name: str = "unknown"
) -> tuple[float, float]:
    """Anderson-Darling test against a fitted CDF.

    Returns (ad_stat, critical_value_5pct).
    Uses subsample of 2000 to give reasonable power without being overpowered.
    Critical values from Stephens (1986) for parameters estimated from data.
    """
    rng = np.random.default_rng(123)
    if len(samples) > 2000:
        samples = rng.choice(samples, 2000, replace=False)
    n = len(samples)
    s = np.sort(samples)
    u = cdf_func(s)
    u = np.clip(u, 1e-15, 1 - 1e-15)
    i = np.arange(1, n + 1)
    ad = -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n
    crit_table = {"lognormal": 0.752, "gamma": 0.786, "invgamma": 0.786}
    crit_5 = crit_table.get(dist_name, 0.752)
    return ad, crit_5


def fit_distributions(samples: np.ndarray) -> list[DistFit]:
    """Fit lognormal, gamma, and inverse-gamma to positive samples.

    Returns list of DistFit sorted by AIC (best first).
    """
    positive = samples[samples > 0]
    if len(positive) < 100:
        return []

    fits = []

    # --- Lognormal ---
    log_s = np.log(positive)
    mu_ln = np.mean(log_s)
    sig_ln = np.std(log_s, ddof=1)
    if sig_ln > 0:
        ll = np.sum(stats.lognorm.logpdf(positive, s=sig_ln, scale=np.exp(mu_ln)))
        aic = -2 * ll + 2 * 2
        median = np.exp(mu_ln)
        cv = np.sqrt(np.exp(sig_ln**2) - 1)
        ad, crit = _ad_test_samples(
            positive,
            lambda x: stats.lognorm.cdf(x, s=sig_ln, scale=np.exp(mu_ln)),
            "lognormal",
        )
        fits.append(
            DistFit(
                name="lognormal",
                params={"mu": mu_ln, "sigma": sig_ln},
                aic=aic,
                ad_stat=ad,
                ad_crit_5pct=crit,
                ad_pass=ad < crit,
                median=median,
                cv=cv,
            )
        )

    # --- Gamma ---
    try:
        a_gam, _, scale_gam = stats.gamma.fit(positive, floc=0)
        if a_gam > 0 and scale_gam > 0:
            ll = np.sum(stats.gamma.logpdf(positive, a_gam, scale=scale_gam))
            aic = -2 * ll + 2 * 2
            median = stats.gamma.ppf(0.5, a_gam, scale=scale_gam)
            cv = 1.0 / np.sqrt(a_gam)
            ad, crit = _ad_test_samples(
                positive,
                lambda x: stats.gamma.cdf(x, a_gam, scale=scale_gam),
                "gamma",
            )
            fits.append(
                DistFit(
                    name="gamma",
                    params={"shape": a_gam, "scale": scale_gam},
                    aic=aic,
                    ad_stat=ad,
                    ad_crit_5pct=crit,
                    ad_pass=ad < crit,
                    median=median,
                    cv=cv,
                )
            )
    except Exception:
        pass

    # --- Inverse Gamma ---
    try:
        inv_s = 1.0 / positive
        a_ig, _, scale_ig = stats.gamma.fit(inv_s, floc=0)
        if a_ig > 0 and scale_ig > 0:
            ll = np.sum(stats.invgamma.logpdf(positive, a_ig, scale=1.0 / scale_ig))
            aic = -2 * ll + 2 * 2
            median = stats.invgamma.ppf(0.5, a_ig, scale=1.0 / scale_ig)
            cv_ig = np.sqrt(1.0 / (a_ig - 2)) if a_ig > 2 else float("inf")
            ad, crit = _ad_test_samples(
                positive,
                lambda x: stats.invgamma.cdf(x, a_ig, scale=1.0 / scale_ig),
                "invgamma",
            )
            fits.append(
                DistFit(
                    name="invgamma",
                    params={"shape": a_ig, "scale": 1.0 / scale_ig},
                    aic=aic,
                    ad_stat=ad,
                    ad_crit_5pct=crit,
                    ad_pass=ad < crit,
                    median=median,
                    cv=cv_ig,
                )
            )
    except Exception:
        pass

    fits.sort(key=lambda f: f.aic)
    return fits


def solve_parameter_samples(target: SubmodelTarget) -> Optional[np.ndarray]:
    """Run observation_code and invert through forward_model to get parameter samples.

    For algebraic models, probes the forward model at test values to detect
    the transform (identity, linear, or nonlinear) and inverts accordingly.

    Returns parameter samples or None if inversion fails.
    """
    rng = np.random.default_rng(42)

    # Build inputs dict
    inputs_dict = {inp.name: inp.value for inp in target.inputs}

    # Run observation_code to get bootstrap samples
    entry = target.calibration.error_model[0]
    local_scope = {"np": np, "numpy": np}
    exec(entry.observation_code, local_scope)
    derive_observation = local_scope["derive_observation"]

    sample_size = int(inputs_dict.get(entry.sample_size_input, 1))
    obs_samples = derive_observation(inputs_dict, sample_size, rng, entry.n_bootstrap)

    # Run forward_model.code to understand the transform
    model = target.calibration.forward_model
    exec(model.code, local_scope)
    compute = local_scope["compute"]

    param_name = target.calibration.parameters[0].name

    # Probe the forward model at a few values to detect the transform
    test_vals = [0.1, 1.0, 10.0]
    outputs = []
    for v in test_vals:
        params = {param_name: v}
        outputs.append(compute(params, inputs_dict))

    # Check if identity: output == param
    if all(abs(o - v) < 1e-10 for o, v in zip(outputs, test_vals)):
        return obs_samples

    # Check if linear: output = a * param + b
    a = (outputs[2] - outputs[0]) / (test_vals[2] - test_vals[0])
    b = outputs[0] - a * test_vals[0]
    predicted_1 = a * test_vals[1] + b
    if abs(predicted_1 - outputs[1]) < 1e-6 * abs(outputs[1] + 1e-30):
        if abs(a) > 1e-30:
            return (obs_samples - b) / a

    # Nonlinear — use numerical inversion via bisection
    from scipy.optimize import brentq

    param_samples = np.full_like(obs_samples, np.nan)
    n_failed = 0
    for i, obs in enumerate(obs_samples):
        try:

            def residual(p):
                return compute({param_name: p}, inputs_dict) - obs

            param_samples[i] = brentq(residual, 1e-10, 1e6, xtol=1e-12)
        except (ValueError, RuntimeError):
            n_failed += 1

    valid = param_samples[np.isfinite(param_samples)]
    if len(valid) < len(obs_samples) * 0.9:
        print(f"  WARNING: {n_failed}/{len(obs_samples)} inversions failed")
    return valid


def process_yaml(yaml_path: Path) -> dict:
    """Process a single SubmodelTarget YAML into a prior specification.

    Returns a dict with keys: name, units, target_id, best_dist, all_fits,
    param_samples, median_data, sigma_data, translation_sigma,
    translation_breakdown, median_prior, sigma_prior, mu_prior, cv_data, cv_prior.

    On failure, returns a dict with keys: name, error.
    """
    import warnings

    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target = SubmodelTarget(**data)

    param_name = target.calibration.parameters[0].name
    param_units = target.calibration.parameters[0].units

    # Get parameter samples
    param_samples = solve_parameter_samples(target)
    if param_samples is None or len(param_samples) == 0:
        return {"name": param_name, "error": "Could not solve for parameter"}

    # Fit candidate distributions
    fits = fit_distributions(param_samples)
    if not fits:
        return {"name": param_name, "error": "All distribution fits failed"}

    best = fits[0]

    # Compute translation sigma
    trans_sigma, breakdown = compute_translation_sigma(target.source_relevance)

    # Convert to equivalent lognormal sigma for the inflation step
    if best.name == "lognormal":
        sigma_data = best.params["sigma"]
        mu_data = best.params["mu"]
    else:
        mu_data = np.log(best.median)
        sigma_data = np.sqrt(np.log(1 + best.cv**2))

    # Total sigma (quadrature in log-space)
    sigma_total = np.sqrt(sigma_data**2 + trans_sigma**2)

    return {
        "name": param_name,
        "units": param_units,
        "target_id": target.target_id,
        "best_dist": best,
        "all_fits": fits,
        "param_samples": param_samples,
        "median_data": best.median,
        "sigma_data": sigma_data,
        "translation_sigma": trans_sigma,
        "translation_breakdown": breakdown,
        "median_prior": best.median,
        "sigma_prior": sigma_total,
        "mu_prior": mu_data,
        "cv_data": best.cv,
        "cv_prior": np.sqrt(np.exp(sigma_total**2) - 1),
    }


def format_report(result: dict) -> str:
    """Format a process_yaml result dict as a human-readable report string."""
    if "error" in result:
        return f"ERROR: {result['name']}: {result['error']}"

    best = result["best_dist"]
    all_fits = result["all_fits"]

    lines = [
        f"Parameter: {result['name']} ({result['units']})",
        "",
        f"Distribution fits (n={len(all_fits)}):",
        f"  {'dist':<12} {'AIC':>10} {'dAIC':>6} {'AD':>7} {'crit5%':>7} {'AD?':>5} "
        f"{'median':>10} {'CV':>6}",
    ]

    best_aic = all_fits[0].aic
    for f in all_fits:
        daic = f.aic - best_aic
        ad_flag = "PASS" if f.ad_pass else "FAIL"
        lines.append(
            f"  {f.name:<12} {f.aic:>10.1f} {daic:>+6.1f} {f.ad_stat:>7.3f} "
            f"{f.ad_crit_5pct:>7.3f} {ad_flag:>5} {f.median:>10.4g} {f.cv:>6.2f}"
        )

    lines.extend(
        [
            "",
            f"Best: {best.name} (median={best.median:.4g}, CV={best.cv:.2f})",
            f"Translation sigma: {result['translation_sigma']:.3f}",
        ]
    )
    for k, v in result["translation_breakdown"].items():
        if v > 0:
            lines.append(f"  {k}: +{v:.2f}")

    lines.append(
        f"Prior: median={result['median_prior']:.4g}, "
        f"sigma={result['sigma_prior']:.3f}, "
        f"CV={result['cv_prior']:.2f}"
    )

    return "\n".join(lines)
