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

import re
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

# ODE types requiring integration (analytical or numerical)
ODE_TYPES = {
    "first_order_decay",
    "exponential_growth",
    "two_state",
    "saturation",
    "logistic",
    "michaelis_menten",
    "custom_ode",
}

ANALYTICAL_ODE_TYPES = {
    "exponential_growth",
    "first_order_decay",
    "saturation",
    "two_state",
    "logistic",
}

_NUMPY_IMPORT_RE = re.compile(r"^\s*import\s+numpy(?:\s+as\s+\w+)?\s*$", re.MULTILINE)


def _strip_numpy_imports(code: str) -> str:
    """Remove ``import numpy as np`` lines from forward model code.

    Forward model code is exec'd with ``{"np": jnp}`` so bare ``np.*`` calls
    map to JAX.  An explicit import inside the function body overrides that
    binding with real NumPy, causing ``__array__()`` errors on JAX tracers.
    """
    return _NUMPY_IMPORT_RE.sub("", code)


def _resolve_ic(sv, inputs_dict: dict[str, float]) -> float:
    """Extract initial condition value from a StateVariable."""
    ic = sv.initial_condition
    if hasattr(ic, "input_ref"):
        return float(inputs_dict[ic.input_ref])
    return float(ic.value)


def _resolve_role(role, param_values, inputs_dict, reference_db):
    """Resolve a ParameterRole to a JAX-traceable value.

    For parameters being estimated, returns the JAX tracer from param_values.
    For InputRef/ReferenceRef/numeric literals, returns a plain float.
    Reuses the logic from submodel_utils._resolve_parameter_role.
    """
    from maple.core.calibration.submodel_utils import _resolve_parameter_role

    return _resolve_parameter_role(role, param_values, inputs_dict, reference_db)


def _make_observable_fn(obs, sv_names):
    """Build a JAX-traceable observable transform.

    Args:
        obs: Observable model (type="identity" or type="custom")
        sv_names: list of state variable names in order

    Returns:
        callable: (t, y, y_start) -> scalar, where y is a JAX array
    """
    import jax.numpy as jnp

    if obs is None or obs.type == "identity":
        # Return the first (or only) state variable
        return lambda t, y, y_start: y[0]

    if obs.type == "custom":
        local = {"np": jnp, "numpy": jnp}
        code = _strip_numpy_imports(obs.code)
        exec(code, local)  # noqa: S102
        _compute = local["compute"]
        return _compute

    raise ValueError(f"Unknown observable type: {obs.type}")


def _make_analytical_ode_fn(
    model_type, model, inputs_dict, reference_db, y0_values, eval_point, obs_fn, sv_names
):
    """Build a JAX-traceable forward fn using closed-form ODE solution.

    Returns a closure: param_values -> scalar prediction at eval_point.
    """
    import jax.numpy as jnp

    _t = float(eval_point)
    _y0 = [float(v) for v in y0_values]
    t0 = float(model.independent_variable.span[0])
    dt = _t - t0

    def _factory(model_type, model, inputs_dict, reference_db, _y0, dt, obs_fn):
        def forward(param_values):
            def r(role):
                return _resolve_role(role, param_values, inputs_dict, reference_db)

            if model_type == "exponential_growth":
                k = r(model.rate_constant)
                y_t = jnp.array([_y0[0] * jnp.exp(k * dt)])

            elif model_type == "first_order_decay":
                k = r(model.rate_constant)
                y_t = jnp.array([_y0[0] * jnp.exp(-k * dt)])

            elif model_type == "saturation":
                k = r(model.rate_constant)
                y_t = jnp.array([1.0 - (1.0 - _y0[0]) * jnp.exp(-k * dt)])

            elif model_type == "two_state":
                k = r(model.forward_rate)
                A0, B0 = _y0[0], _y0[1]
                A_t = A0 * jnp.exp(-k * dt)
                B_t = A0 * (1.0 - jnp.exp(-k * dt)) + B0
                y_t = jnp.array([A_t, B_t])

            elif model_type == "logistic":
                k = r(model.rate_constant)
                K = r(model.carrying_capacity)
                y0_val = _y0[0]
                y_t = jnp.array(
                    [K * y0_val * jnp.exp(k * dt) / (K - y0_val + y0_val * jnp.exp(k * dt))]
                )

            else:
                raise ValueError(f"No analytical solution for {model_type}")

            y_start = jnp.array(_y0)
            return obs_fn(_t, y_t, y_start)

        return forward

    return _factory(model_type, model, inputs_dict, reference_db, _y0, dt, obs_fn)


def _make_diffrax_ode_fn(
    model_type, model, inputs_dict, reference_db, y0_values, t_span, eval_point, obs_fn, sv_names
):
    """Build a JAX-traceable forward fn using diffrax ODE solver.

    Used for michaelis_menten and custom_ode types.
    """
    import jax.numpy as jnp

    _t0 = float(t_span[0])
    _t1 = float(eval_point)
    _y0 = jnp.array([float(v) for v in y0_values])
    _y0_start = _y0  # for observable y_start argument

    if model_type == "custom_ode":
        # exec user code to get ode(t, y, params, inputs)
        local = {"np": jnp, "numpy": jnp}
        code = _strip_numpy_imports(model.code)
        exec(code, local)  # noqa: S102
        _user_ode = local["ode"]

        def _factory_custom(_user_ode, inputs_dict, _y0, _y0_start, _t0, _t1, obs_fn):
            def forward(param_values):
                import diffrax

                def rhs(t, y, args):
                    params, inputs = args
                    dy = _user_ode(t, y, params, inputs)
                    if isinstance(dy, (list, tuple)):
                        return jnp.array(dy)
                    return dy

                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(rhs),
                    diffrax.Tsit5(),
                    t0=_t0,
                    t1=_t1,
                    dt0=(_t1 - _t0) / 100.0,
                    y0=_y0,
                    args=(param_values, inputs_dict),
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
                    max_steps=16384,
                    throw=False,
                )
                y_final = sol.ys[-1]
                return obs_fn(_t1, y_final, _y0_start)

            return forward

        return _factory_custom(_user_ode, inputs_dict, _y0, _y0_start, _t0, _t1, obs_fn)

    elif model_type == "michaelis_menten":

        def _factory_mm(model, inputs_dict, reference_db, _y0, _y0_start, _t0, _t1, obs_fn):
            def forward(param_values):
                import diffrax

                def r(role):
                    return _resolve_role(role, param_values, inputs_dict, reference_db)

                vmax = r(model.vmax)
                km = r(model.km)

                def rhs(t, y, _args):
                    return jnp.array([-vmax * y[0] / (km + y[0])])

                sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(rhs),
                    diffrax.Tsit5(),
                    t0=_t0,
                    t1=_t1,
                    dt0=(_t1 - _t0) / 100.0,
                    y0=_y0,
                    args=None,
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
                    max_steps=16384,
                    throw=False,
                )
                y_final = sol.ys[-1]
                return obs_fn(_t1, y_final, _y0_start)

            return forward

        return _factory_mm(model, inputs_dict, reference_db, _y0, _y0_start, _t0, _t1, obs_fn)

    else:
        raise ValueError(f"_make_diffrax_ode_fn does not handle model type: {model_type}")


def _build_forward_fns(
    target: SubmodelTarget,
    reference_db: Optional[dict[str, float]] = None,
) -> list:
    """Build one callable per error model entry.

    Each callable: param_dict -> predicted scalar (JAX-traceable).

    For structured types, wraps _evaluate_structured_model.
    For algebraic types, execs model.code with jax.numpy.
    For ODE types, uses analytical solutions or diffrax.
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
            # Inject x_input as '_x' so compute() can use it for multi-point eval
            if entry.x_input:
                _inputs["_x"] = base_inputs[entry.x_input]
            local = {"np": jnp, "numpy": jnp}
            # Strip 'import numpy' lines from forward model code — the exec scope
            # injects jax.numpy as 'np'/'numpy', but an explicit import inside the
            # function body would override that with real numpy, causing JAX tracer
            # errors during MCMC.
            _code = _strip_numpy_imports(model.code)
            exec(_code, local)  # noqa: S102
            _compute = local["compute"]

            def _make_algebraic_fn(_compute, _inputs):
                def forward(param_values):
                    return _compute(param_values, _inputs)

                return forward

            fns.append(_make_algebraic_fn(_compute, _inputs))

        elif model_type in ODE_TYPES:
            sv_list = model.state_variables
            sv_names = [sv.name for sv in sv_list]
            y0_values = [_resolve_ic(sv, base_inputs) for sv in sv_list]
            t_span = model.independent_variable.span
            eval_pt = entry.evaluation_points[0] if entry.evaluation_points else t_span[1]
            obs_fn = _make_observable_fn(entry.observable, sv_names)

            if model_type in ANALYTICAL_ODE_TYPES:
                fn = _make_analytical_ode_fn(
                    model_type,
                    model,
                    base_inputs,
                    reference_db,
                    y0_values,
                    eval_pt,
                    obs_fn,
                    sv_names,
                )
            else:  # michaelis_menten, custom_ode
                fn = _make_diffrax_ode_fn(
                    model_type,
                    model,
                    base_inputs,
                    reference_db,
                    y0_values,
                    t_span,
                    eval_pt,
                    obs_fn,
                    sv_names,
                )
            fns.append(fn)

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

    Validates that all non-nuisance parameter names exist in prior_specs.
    Injects nuisance parameters' inline priors into prior_specs.
    """
    likelihoods = []

    for target in targets:
        # Validate parameter coverage and inject nuisance priors
        for param in target.calibration.parameters:
            if param.nuisance:
                # Inject inline prior into prior_specs (if not already present
                # from another target sharing the same nuisance parameter)
                if param.name not in prior_specs:
                    p = param.prior
                    prior_specs[param.name] = PriorSpec(
                        name=param.name,
                        distribution=p.distribution,
                        units=param.units,
                        mu=p.mu,
                        sigma=p.sigma,
                        lower=p.lower,
                        upper=p.upper,
                    )
            elif param.name not in prior_specs:
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

            site_name = f"obs_{tl.target_id}_{j}" if len(tl.entries) > 1 else f"obs_{tl.target_id}"

            # Guard against NaN/non-positive predictions from solver failures
            # (e.g. diffrax hitting max_steps with extreme parameter draws).
            # Replace with a safe dummy value and assign -inf log-probability
            # so NUTS rejects the sample.
            valid = jnp.isfinite(predicted) & (predicted > 0)
            safe_predicted = jnp.where(valid, predicted, 1.0)
            numpyro.factor(f"valid_{site_name}", jnp.where(valid, 0.0, -jnp.inf))

            if entry.family == "lognormal":
                sigma_total = jnp.sqrt(entry.sigma**2 + tl.sigma_trans**2)
                numpyro.sample(
                    site_name,
                    dist.LogNormal(jnp.log(safe_predicted), sigma_total),
                    obs=entry.value,
                )
            else:  # normal
                sd_total = jnp.sqrt(entry.sigma**2 + (entry.value * tl.sigma_trans) ** 2)
                numpyro.sample(
                    site_name,
                    dist.Normal(safe_predicted, sd_total),
                    obs=entry.value,
                )


# =============================================================================
# MCMC runner
# =============================================================================


def _compute_mcmc_diagnostics(
    mcmc,
    prior_specs: dict[str, "PriorSpec"],
) -> dict:
    """Extract MCMC diagnostics from a completed MCMC run.

    Returns a dict with:
      - num_divergences: int
      - per_param: dict of {param_name: {r_hat, n_eff, contraction, z_score}}

    Contraction = 1 - var(posterior) / var(prior).  Values near 1 mean the data
    strongly informed the parameter; near 0 means little learning.

    Z-score = |mean(posterior) - mean(prior)| / std(prior).  Large values (>2)
    indicate the data pulled the posterior away from the prior.
    """
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    samples = mcmc.get_samples(group_by_chain=True)
    flat_samples = mcmc.get_samples(group_by_chain=False)
    extra_fields = mcmc.get_extra_fields()

    # Divergences
    num_div = 0
    if "diverging" in extra_fields:
        num_div = int(np.asarray(extra_fields["diverging"]).sum())

    per_param = {}
    for name, spec in prior_specs.items():
        chain_arr = np.asarray(samples[name])  # (num_chains, num_samples)
        flat_arr = np.asarray(flat_samples[name])

        # r_hat and n_eff
        if chain_arr.ndim == 1:
            # Single chain: reshape to (1, N)
            chain_arr = chain_arr[np.newaxis, :]
        r_hat = float(split_gelman_rubin(chain_arr))
        n_eff = float(effective_sample_size(chain_arr))

        # Prior moments (in natural space)
        if spec.distribution == "lognormal":
            prior_mean = np.exp(spec.mu + spec.sigma**2 / 2)
            prior_var = (np.exp(spec.sigma**2) - 1) * np.exp(2 * spec.mu + spec.sigma**2)
        elif spec.distribution == "normal":
            prior_mean = spec.mu
            prior_var = spec.sigma**2
        elif spec.distribution == "uniform":
            prior_mean = (spec.lower + spec.upper) / 2
            prior_var = (spec.upper - spec.lower) ** 2 / 12
        else:
            prior_mean = float(np.mean(flat_arr))
            prior_var = float(np.var(flat_arr))

        post_mean = float(np.mean(flat_arr))
        post_var = float(np.var(flat_arr))
        prior_std = float(np.sqrt(prior_var)) if prior_var > 0 else 1e-30

        contraction = 1.0 - post_var / prior_var if prior_var > 0 else 0.0
        z_score = abs(post_mean - prior_mean) / prior_std

        per_param[name] = {
            "r_hat": r_hat,
            "n_eff": n_eff,
            "contraction": contraction,
            "z_score": z_score,
        }

    return {
        "num_divergences": num_div,
        "per_param": per_param,
    }


def run_joint_inference(
    prior_specs: dict[str, PriorSpec],
    targets: list[SubmodelTarget],
    reference_db: Optional[dict[str, float]] = None,
    num_warmup: int = 1000,
    num_samples: int = 5000,
    num_chains: int = 4,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
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
        Tuple of (samples_dict, diagnostics_dict).
        samples_dict: {param_name: np.ndarray} with shape (num_samples * num_chains,)
        diagnostics_dict: {num_divergences, per_param: {name: {r_hat, n_eff, contraction, z_score}}}
    """
    try:
        import jax
        import jax.random
        from numpyro.infer import MCMC, NUTS
    except ImportError as e:
        raise ImportError(
            "JAX and NumPyro are required for inference. "
            "Install with: pip install maple[inference]"
        ) from e

    # Ensure JAX can run parallel chains on CPU
    import numpyro

    numpyro.set_host_device_count(num_chains)

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
    diagnostics = _compute_mcmc_diagnostics(mcmc, prior_specs)

    samples = mcmc.get_samples()
    # Convert JAX arrays to numpy
    samples_np = {name: np.asarray(vals) for name, vals in samples.items()}
    return samples_np, diagnostics
