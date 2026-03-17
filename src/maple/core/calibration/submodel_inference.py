"""
Joint Bayesian inference across SubmodelTargets using NumPyro.

Builds a single NumPyro model from all SubmodelTargets:
  - Priors from pdac_priors.csv (broad starting distributions)
  - Likelihoods from SubmodelTarget forward models + bootstrap statistics
  - Translation sigma applied in the likelihood (not post-hoc)

Shared parameters (same name across targets) are sampled once and reused.

Usage::

    from maple.core.calibration.submodel_inference import run_joint_inference, load_priors_from_csv

    prior_specs = load_priors_from_csv("pdac_priors.csv")
    targets = [SubmodelTarget(**yaml.safe_load(p.read_text())) for p in yaml_paths]
    samples = run_joint_inference(prior_specs, targets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from maple.core.calibration.submodel_target import SubmodelTarget
from maple.core.calibration.submodel_utils import (
    STRUCTURED_ALGEBRAIC_TYPES,
    _evaluate_structured_model,
)
from maple.core.calibration.yaml_to_prior import (
    DistFit,
    compute_translation_sigma,
    fit_distributions,
)


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class PriorSpec:
    """Prior specification for a single parameter, loaded from CSV."""

    name: str
    distribution: str  # "lognormal", "normal", "uniform"
    units: str
    mu: Optional[float] = None
    sigma: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass
class ErrorModelEntry:
    """One error model entry, ready for NumPyro.

    Produced by running bootstrap on observation_code and fitting the result.
    """

    forward_fn: object  # callable: params dict -> predicted scalar
    value: float  # observed value (median from bootstrap fit)
    sigma: float  # log-space sigma (lognormal) or linear SD (normal)
    family: str  # "lognormal" or "normal"
    fit: DistFit  # full fit result for diagnostics


@dataclass
class TargetLikelihood:
    """All error model entries for one SubmodelTarget."""

    target_id: str
    sigma_trans: float  # translation sigma from source_relevance
    sigma_breakdown: dict  # per-axis breakdown for provenance
    entries: list[ErrorModelEntry] = field(default_factory=list)


# =============================================================================
# CSV prior loader
# =============================================================================


def load_priors_from_csv(csv_path: Path) -> dict[str, PriorSpec]:
    """Load parameter priors from a CSV file.

    Expected columns: name, median, units, distribution, dist_param1, dist_param2

    For lognormal: dist_param1=mu (log-space), dist_param2=sigma (log-space)
    For normal: dist_param1=mu, dist_param2=sigma
    For uniform: dist_param1=lower, dist_param2=upper
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    specs = {}
    for _, row in df.iterrows():
        name = row["name"]
        dist = row["distribution"].lower()
        units = str(row.get("units", "dimensionless"))

        if dist == "lognormal":
            specs[name] = PriorSpec(
                name=name,
                distribution="lognormal",
                units=units,
                mu=float(row["dist_param1"]),
                sigma=float(row["dist_param2"]),
            )
        elif dist == "normal":
            specs[name] = PriorSpec(
                name=name,
                distribution="normal",
                units=units,
                mu=float(row["dist_param1"]),
                sigma=float(row["dist_param2"]),
            )
        elif dist == "uniform":
            specs[name] = PriorSpec(
                name=name,
                distribution="uniform",
                units=units,
                lower=float(row["dist_param1"]),
                upper=float(row["dist_param2"]),
            )
        else:
            raise ValueError(f"Unknown distribution '{dist}' for parameter '{name}'")

    return specs


# =============================================================================
# Bootstrap runner
# =============================================================================


def _run_bootstrap(entry, inputs_dict: dict[str, float]) -> DistFit:
    """Execute observation_code and fit distributions to bootstrap samples.

    Args:
        entry: ErrorModel entry with observation_code
        inputs_dict: {input_name: float_value} from target inputs

    Returns:
        Best-fitting DistFit (by AIC)

    Raises:
        RuntimeError: if bootstrap or fitting fails
    """
    rng = np.random.default_rng(42)

    local_scope = {"np": np, "numpy": np}
    exec(entry.observation_code, local_scope)  # noqa: S102
    derive_observation = local_scope["derive_observation"]

    sample_size = int(inputs_dict.get(entry.sample_size_input, 1))
    samples = derive_observation(inputs_dict, sample_size, rng, entry.n_bootstrap)

    if not isinstance(samples, np.ndarray):
        samples = np.asarray(samples)

    fits = fit_distributions(samples)
    if not fits:
        raise RuntimeError(
            f"Could not fit any distribution to bootstrap samples for '{entry.name}'. "
            f"Got {len(samples)} samples, {np.sum(samples > 0)} positive."
        )

    return fits[0]


# =============================================================================
# Forward function builder
# =============================================================================

# ODE types that would need diffrax — deferred
ODE_TYPES = {
    "first_order_decay",
    "exponential_growth",
    "two_state",
    "saturation",
    "logistic",
    "michaelis_menten",
    "custom_ode",
}


def _build_forward_fns(
    target: SubmodelTarget,
    reference_db: Optional[dict[str, float]] = None,
) -> list:
    """Build one callable per error model entry.

    Each callable: param_dict -> predicted scalar (JAX-traceable).

    For structured types, wraps _evaluate_structured_model.
    For algebraic types, execs model.code with jax.numpy.
    """
    import jax.numpy as jnp

    model = target.calibration.forward_model
    model_type = model.type
    base_inputs = {inp.name: inp.value for inp in target.inputs}
    fns = []

    for entry in target.calibration.error_model:
        if model_type in STRUCTURED_ALGEBRAIC_TYPES:
            # Resolve x_input to numeric value for direct_fit / power_law
            x_val = base_inputs[entry.x_input] if entry.x_input else None

            # Capture in closure
            _x = x_val
            _inputs = base_inputs.copy()
            _model = model
            _ref = reference_db

            def _make_structured_fn(_model, _inputs, _ref, _x):
                def forward(param_values):
                    return _evaluate_structured_model(
                        _model, param_values, _inputs, _ref, x_value=_x
                    )

                return forward

            fns.append(_make_structured_fn(_model, _inputs, _ref, _x))

        elif model_type == "algebraic":
            _inputs = base_inputs.copy()
            local = {"np": jnp, "numpy": jnp}
            exec(model.code, local)  # noqa: S102
            _compute = local["compute"]

            def _make_algebraic_fn(_compute, _inputs):
                def forward(param_values):
                    return _compute(param_values, _inputs)

                return forward

            fns.append(_make_algebraic_fn(_compute, _inputs))

        elif model_type in ODE_TYPES:
            raise NotImplementedError(
                f"ODE model type '{model_type}' not yet supported in inference pipeline. "
                f"Use structured or algebraic types."
            )

        else:
            raise ValueError(f"Unknown forward model type: {model_type}")

    return fns


# =============================================================================
# Target likelihood builder
# =============================================================================


def build_target_likelihoods(
    targets: list[SubmodelTarget],
    prior_specs: dict[str, PriorSpec],
    reference_db: Optional[dict[str, float]] = None,
) -> list[TargetLikelihood]:
    """Build likelihood specifications from targets.

    For each target:
      1. Compute translation sigma from source_relevance
      2. Build forward functions (one per error model entry)
      3. Run bootstrap on each error model entry
      4. Package into TargetLikelihood

    Validates that all parameter names exist in prior_specs.
    """
    likelihoods = []

    for target in targets:
        # Validate parameter coverage
        for param in target.calibration.parameters:
            if param.name not in prior_specs:
                raise ValueError(
                    f"Parameter '{param.name}' in target '{target.target_id}' "
                    f"not found in priors CSV. Available: {sorted(prior_specs.keys())}"
                )

        # Translation sigma
        sigma_trans, breakdown = compute_translation_sigma(target.source_relevance)

        # Forward functions
        forward_fns = _build_forward_fns(target, reference_db)

        # Inputs dict for bootstrap
        inputs_dict = {inp.name: inp.value for inp in target.inputs}

        # Build entries
        entries = []
        for j, entry in enumerate(target.calibration.error_model):
            dist_fit = _run_bootstrap(entry, inputs_dict)

            # Convert to log-space sigma for likelihood
            if dist_fit.name == "lognormal":
                sigma = dist_fit.params["sigma"]
                family = "lognormal"
            elif dist_fit.name in ("gamma", "invgamma"):
                # Approximate as lognormal: sigma_log = sqrt(log(1 + cv^2))
                sigma = np.sqrt(np.log(1 + dist_fit.cv**2))
                family = "lognormal"
            else:
                sigma = dist_fit.params.get("sigma", dist_fit.cv * dist_fit.median)
                family = "normal"

            entries.append(
                ErrorModelEntry(
                    forward_fn=forward_fns[j],
                    value=dist_fit.median,
                    sigma=sigma,
                    family=family,
                    fit=dist_fit,
                )
            )

        likelihoods.append(
            TargetLikelihood(
                target_id=target.target_id,
                sigma_trans=sigma_trans,
                sigma_breakdown=breakdown,
                entries=entries,
            )
        )

    return likelihoods


# =============================================================================
# NumPyro model
# =============================================================================


def submodel_joint_model(prior_specs, target_likelihoods):
    """NumPyro model for joint inference across SubmodelTargets.

    Args:
        prior_specs: dict of {param_name: PriorSpec}
        target_likelihoods: list of TargetLikelihood
    """
    import numpyro
    import numpyro.distributions as dist
    from jax import numpy as jnp

    # Sample each parameter from its CSV-specified prior
    params = {}
    for name, spec in prior_specs.items():
        if spec.distribution == "lognormal":
            params[name] = numpyro.sample(name, dist.LogNormal(spec.mu, spec.sigma))
        elif spec.distribution == "normal":
            params[name] = numpyro.sample(name, dist.Normal(spec.mu, spec.sigma))
        elif spec.distribution == "uniform":
            params[name] = numpyro.sample(name, dist.Uniform(spec.lower, spec.upper))
        else:
            raise ValueError(f"Unsupported prior distribution: {spec.distribution}")

    # Likelihood: loop over targets, then error model entries
    for tl in target_likelihoods:
        for j, entry in enumerate(tl.entries):
            predicted = entry.forward_fn(params)

            site_name = (
                f"obs_{tl.target_id}_{j}"
                if len(tl.entries) > 1
                else f"obs_{tl.target_id}"
            )

            if entry.family == "lognormal":
                sigma_total = jnp.sqrt(entry.sigma**2 + tl.sigma_trans**2)
                numpyro.sample(
                    site_name,
                    dist.LogNormal(jnp.log(predicted), sigma_total),
                    obs=entry.value,
                )
            else:  # normal
                sd_total = jnp.sqrt(
                    entry.sigma**2 + (entry.value * tl.sigma_trans) ** 2
                )
                numpyro.sample(
                    site_name,
                    dist.Normal(predicted, sd_total),
                    obs=entry.value,
                )


# =============================================================================
# MCMC runner
# =============================================================================


def run_joint_inference(
    prior_specs: dict[str, PriorSpec],
    targets: list[SubmodelTarget],
    reference_db: Optional[dict[str, float]] = None,
    num_warmup: int = 1000,
    num_samples: int = 5000,
    num_chains: int = 4,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run joint MCMC inference across all SubmodelTargets.

    Args:
        prior_specs: Prior specifications from CSV
        targets: List of SubmodelTarget objects
        reference_db: Optional reference values for ReferenceRef resolution
        num_warmup: NUTS warmup iterations per chain
        num_samples: Post-warmup samples per chain
        num_chains: Number of MCMC chains
        seed: Random seed

    Returns:
        dict of {param_name: np.ndarray} with shape (num_samples * num_chains,)
    """
    try:
        import jax
        import jax.random
        import numpyro
        from numpyro.infer import MCMC, NUTS
    except ImportError as e:
        raise ImportError(
            "JAX and NumPyro are required for inference. "
            "Install with: pip install maple[inference]"
        ) from e

    # Build target likelihoods (runs bootstrap, builds forward fns)
    target_likelihoods = build_target_likelihoods(targets, prior_specs, reference_db)

    n_likelihood_terms = sum(len(tl.entries) for tl in target_likelihoods)
    print(
        f"Built joint model: {len(prior_specs)} parameters, "
        f"{len(targets)} targets, {n_likelihood_terms} likelihood terms"
    )

    # Run MCMC
    kernel = NUTS(submodel_joint_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        prior_specs=prior_specs,
        target_likelihoods=target_likelihoods,
    )

    # Diagnostics
    mcmc.print_summary()

    samples = mcmc.get_samples()
    # Convert JAX arrays to numpy
    return {name: np.asarray(vals) for name, vals in samples.items()}