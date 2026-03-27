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

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from maple.core.calibration.parameter_groups import ParameterGroupsConfig
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

logger = logging.getLogger(__name__)


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
    Uses fixed-step Heun solver (no adaptive stepping) for fast JAX compilation.
    """
    import jax.numpy as jnp

    _t0 = float(t_span[0])
    _t1 = float(eval_point)
    _y0 = jnp.array([float(v) for v in y0_values], dtype=jnp.float32)
    _y0_start = _y0  # for observable y_start argument
    # Fixed step size: 200 steps across the integration interval
    _dt = (_t1 - _t0) / 200.0

    if model_type == "custom_ode":
        # exec user code to get ode(t, y, params, inputs)
        local = {"np": jnp, "numpy": jnp}
        code = _strip_numpy_imports(model.code)
        exec(code, local)  # noqa: S102
        _user_ode = local["ode"]

        def _factory_custom(_user_ode, inputs_dict, _y0, _y0_start, _t0, _t1, _dt, obs_fn):
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
                    diffrax.Heun(),
                    t0=_t0,
                    t1=_t1,
                    dt0=_dt,
                    y0=_y0,
                    args=(param_values, inputs_dict),
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.ConstantStepSize(),
                    max_steps=4096,
                    throw=False,
                    progress_meter=diffrax.NoProgressMeter(),
                )
                y_final = sol.ys[-1]
                return obs_fn(_t1, y_final, _y0_start)

            return forward

        return _factory_custom(_user_ode, inputs_dict, _y0, _y0_start, _t0, _t1, _dt, obs_fn)

    elif model_type == "michaelis_menten":

        def _factory_mm(model, inputs_dict, reference_db, _y0, _y0_start, _t0, _t1, _dt, obs_fn):
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
                    diffrax.Heun(),
                    t0=_t0,
                    t1=_t1,
                    dt0=_dt,
                    y0=_y0,
                    args=None,
                    saveat=diffrax.SaveAt(t1=True),
                    stepsize_controller=diffrax.ConstantStepSize(),
                    max_steps=4096,
                    throw=False,
                    progress_meter=diffrax.NoProgressMeter(),
                )
                y_final = sol.ys[-1]
                return obs_fn(_t1, y_final, _y0_start)

            return forward

        return _factory_mm(model, inputs_dict, reference_db, _y0, _y0_start, _t0, _t1, _dt, obs_fn)

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

            # Per-entry observable: if present, compute() returns a vector/dict
            # and the observable selects/transforms the relevant output.
            # Same pattern as ODE models. If no observable, return raw scalar.
            _obs_fn = None
            if entry.observable is not None:
                _obs_fn = _make_observable_fn(entry.observable, [])

            def _make_algebraic_fn(_compute, _inputs, _obs_fn):
                def forward(param_values):
                    raw = _compute(param_values, _inputs)
                    if _obs_fn is not None:
                        return _obs_fn(None, raw, None)
                    return raw

                return forward

            fns.append(_make_algebraic_fn(_compute, _inputs, _obs_fn))

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


def build_numpy_forward_fns(
    target: SubmodelTarget,
    reference_db: Optional[dict[str, float]] = None,
) -> list:
    """Build one callable per error model entry using numpy/scipy (no JAX).

    Same interface as _build_forward_fns but ~500x faster per-call because
    it avoids JAX dispatch overhead. Intended for simulation-based inference
    (NPE) where forward models are called thousands of times independently.

    Each callable: param_dict -> predicted float

    For structured types, uses _evaluate_structured_model (already numpy).
    For algebraic types, execs model.code with numpy.
    For ODE types, uses scipy.integrate.solve_ivp.
    """
    from scipy.integrate import solve_ivp

    model = target.calibration.forward_model
    model_type = model.type
    base_inputs = {inp.name: inp.value for inp in target.inputs}
    fns = []

    def _make_np_observable_fn(obs):
        """Build a numpy observable: (t, y_array, y0_array) -> scalar."""
        if obs is None or obs.type == "identity":
            return lambda t, y, y_start: float(y[0])
        if obs.type == "custom":
            local = {"np": np, "numpy": np}
            exec(obs.code, local)  # noqa: S102
            return local["compute"]
        raise ValueError(f"Unknown observable type: {obs.type}")

    for entry in target.calibration.error_model:
        if model_type in STRUCTURED_ALGEBRAIC_TYPES:
            x_val = base_inputs[entry.x_input] if entry.x_input else None
            _x = x_val
            _inputs = base_inputs.copy()
            _model = model
            _ref = reference_db

            def _make_structured_fn(_model, _inputs, _ref, _x):
                def forward(param_values):
                    return float(
                        _evaluate_structured_model(_model, param_values, _inputs, _ref, x_value=_x)
                    )

                return forward

            fns.append(_make_structured_fn(_model, _inputs, _ref, _x))

        elif model_type == "algebraic":
            _inputs = base_inputs.copy()
            if entry.x_input:
                _inputs["_x"] = base_inputs[entry.x_input]
            local = {"np": np, "numpy": np}
            exec(model.code, local)  # noqa: S102
            _compute = local["compute"]

            _obs_fn = None
            if entry.observable is not None:
                _obs_fn = _make_np_observable_fn(entry.observable)

            def _make_algebraic_fn(_compute, _inputs, _obs_fn):
                def forward(param_values):
                    raw = _compute(param_values, _inputs)
                    if _obs_fn is not None:
                        return float(_obs_fn(None, raw, None))
                    return float(raw)

                return forward

            fns.append(_make_algebraic_fn(_compute, _inputs, _obs_fn))

        elif model_type in ODE_TYPES:
            sv_list = model.state_variables
            y0_values = np.array([_resolve_ic(sv, base_inputs) for sv in sv_list])
            t_span = [
                float(model.independent_variable.span[0]),
                float(model.independent_variable.span[1]),
            ]
            eval_pt = entry.evaluation_points[0] if entry.evaluation_points else t_span[1]
            obs_fn = _make_np_observable_fn(entry.observable)

            if model_type in ANALYTICAL_ODE_TYPES:
                # Reuse analytical solutions (they're just math, work with numpy)
                _t = float(eval_pt)
                _y0 = list(y0_values)
                dt = _t - t_span[0]

                def _make_analytical_np(
                    model_type, model, base_inputs, reference_db, _y0, dt, obs_fn
                ):
                    def forward(param_values):
                        def r(role):
                            return _resolve_role(role, param_values, base_inputs, reference_db)

                        if model_type == "exponential_growth":
                            k = r(model.rate_constant)
                            y_t = np.array([_y0[0] * np.exp(k * dt)])
                        elif model_type == "first_order_decay":
                            k = r(model.rate_constant)
                            y_t = np.array([_y0[0] * np.exp(-k * dt)])
                        elif model_type == "saturation":
                            k = r(model.rate_constant)
                            y_t = np.array([1.0 - (1.0 - _y0[0]) * np.exp(-k * dt)])
                        elif model_type == "logistic":
                            k = r(model.rate_constant)
                            K = r(model.carrying_capacity)
                            y_t = np.array([K / (1 + ((K / _y0[0]) - 1) * np.exp(-k * dt))])
                        else:
                            y_t = np.array(_y0)
                        return float(obs_fn(_t, y_t, np.array(_y0)))

                    return forward

                fns.append(
                    _make_analytical_np(
                        model_type, model, base_inputs, reference_db, _y0, dt, obs_fn
                    )
                )

            elif model_type == "custom_ode":
                local = {"np": np, "numpy": np}
                exec(model.code, local)  # noqa: S102
                _user_ode = local["ode"]

                def _make_scipy_ode(_user_ode, base_inputs, y0_values, t_span, eval_pt, obs_fn):
                    def forward(param_values):
                        def rhs(t, y):
                            dy = _user_ode(t, y, param_values, base_inputs)
                            return np.array(dy) if isinstance(dy, (list, tuple)) else dy

                        sol = solve_ivp(
                            rhs,
                            t_span,
                            y0_values,
                            t_eval=[eval_pt],
                            method="RK23",
                            rtol=1e-4,
                            atol=1e-6,
                            max_step=1.0,
                        )
                        if not sol.success:
                            return float("nan")
                        return float(obs_fn(eval_pt, sol.y[:, -1], y0_values))

                    return forward

                fns.append(
                    _make_scipy_ode(_user_ode, base_inputs, y0_values, t_span, eval_pt, obs_fn)
                )

            elif model_type == "michaelis_menten":

                def _make_scipy_mm(
                    model, base_inputs, reference_db, y0_values, t_span, eval_pt, obs_fn
                ):
                    def forward(param_values):
                        def r(role):
                            return _resolve_role(role, param_values, base_inputs, reference_db)

                        vmax = r(model.vmax)
                        km = r(model.km)

                        def rhs(t, y):
                            return np.array([-vmax * y[0] / (km + y[0])])

                        sol = solve_ivp(
                            rhs,
                            t_span,
                            y0_values,
                            t_eval=[eval_pt],
                            method="RK23",
                            rtol=1e-4,
                            atol=1e-6,
                            max_step=1.0,
                        )
                        if not sol.success:
                            return float("nan")
                        return float(obs_fn(eval_pt, sol.y[:, -1], y0_values))

                    return forward

                fns.append(
                    _make_scipy_mm(
                        model, base_inputs, reference_db, y0_values, t_span, eval_pt, obs_fn
                    )
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

    Validates that all non-nuisance parameter names exist in prior_specs.
    Injects nuisance parameters' inline priors into prior_specs.
    """
    import time

    likelihoods = []

    for i, target in enumerate(targets):
        t0 = time.monotonic()
        model_type = target.calibration.forward_model.type
        print(
            f"  [{i+1}/{len(targets)}] {target.target_id} ({model_type})...",
            end="",
            flush=True,
        )
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
        dt = time.monotonic() - t0
        print(f" {len(entries)} entries, {dt:.1f}s", flush=True)

    return likelihoods


# =============================================================================
# NumPyro model
# =============================================================================


def submodel_joint_model(prior_specs, target_likelihoods, parameter_groups=None):
    """NumPyro model for joint inference across SubmodelTargets.

    Args:
        prior_specs: dict of {param_name: PriorSpec}
        target_likelihoods: list of TargetLikelihood
        parameter_groups: optional ParameterGroupsConfig for hierarchical sampling
    """
    import numpyro
    import numpyro.distributions as dist
    from jax import numpy as jnp

    # Identify which parameters are in hierarchical groups
    grouped_params: set[str] = set()
    if parameter_groups is not None:
        grouped_params = parameter_groups.all_grouped_params

    # Sample non-grouped parameters from their CSV-specified priors
    params = {}
    for name, spec in prior_specs.items():
        if name in grouped_params:
            continue  # handled below in hierarchical block
        if spec.distribution == "lognormal":
            params[name] = numpyro.sample(name, dist.LogNormal(spec.mu, spec.sigma))
        elif spec.distribution == "normal":
            params[name] = numpyro.sample(name, dist.Normal(spec.mu, spec.sigma))
        elif spec.distribution == "uniform":
            params[name] = numpyro.sample(name, dist.Uniform(spec.lower, spec.upper))
        else:
            raise ValueError(f"Unsupported prior distribution: {spec.distribution}")

    # Sample hierarchical groups
    if parameter_groups is not None:
        # Determine which grouped params are actually constrained by targets
        constrained_params: set[str] = set()
        for tl in target_likelihoods:
            for entry in tl.entries:
                # forward_fn closure captures param names; check prior_specs keys
                # that appear in the target's forward model
                pass
        # Simpler: a param is constrained if it appears in prior_specs
        # (prior_specs only includes params referenced by loaded targets)
        constrained_params = set(prior_specs.keys())

        for group in parameter_groups.groups:
            gid = group.group_id
            member_names = {m.name for m in group.members}

            # Skip groups where no member is constrained by any target
            n_constrained = len(member_names & constrained_params)
            if n_constrained == 0:
                logger.info("Skipping group %s: no members constrained by targets", gid)
                # Fall back to independent CSV priors for these params
                for member in group.members:
                    if member.name in prior_specs:
                        spec = prior_specs[member.name]
                        if spec.distribution == "lognormal":
                            params[member.name] = numpyro.sample(
                                member.name, dist.LogNormal(spec.mu, spec.sigma)
                            )
                        elif spec.distribution == "normal":
                            params[member.name] = numpyro.sample(
                                member.name, dist.Normal(spec.mu, spec.sigma)
                            )
                continue

            bp = group.resolve_base_prior(prior_specs)

            # Sample group base rate (log-space)
            if bp.distribution == "lognormal":
                k_base = numpyro.sample(f"{gid}__base", dist.LogNormal(bp.mu, bp.sigma))
            else:  # normal
                k_base = numpyro.sample(f"{gid}__base", dist.Normal(bp.mu, bp.sigma))

            # For small groups (<=2 constrained members), fix tau to avoid
            # unidentifiable hyperparameter that harms sampling geometry.
            # For larger groups, sample tau with non-centered parameterization.
            tau_prior = group.between_member_sd
            if n_constrained <= 2:
                tau = tau_prior.sigma  # fixed, not sampled
            else:
                tau = numpyro.sample(f"{gid}__tau", dist.HalfNormal(tau_prior.sigma))

            # Non-centered parameterization: sample z ~ N(0,1), compute delta = tau * z
            # This avoids the "funnel" geometry where tau and delta are correlated.
            for member in group.members:
                dp = member.delta_prior
                if dp is not None:
                    # Informative delta prior — still use non-centered
                    z = numpyro.sample(
                        f"{member.name}__z",
                        dist.Normal(0.0, 1.0),
                    )
                    delta = dp.mu + dp.sigma * z
                else:
                    # Default: non-centered on group mean
                    z = numpyro.sample(
                        f"{member.name}__z",
                        dist.Normal(0.0, 1.0),
                    )
                    delta = tau * z

                # Final parameter value: k_i = k_base * exp(delta_i)
                params[member.name] = numpyro.deterministic(member.name, k_base * jnp.exp(delta))

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
    parameter_groups: Optional[ParameterGroupsConfig] = None,
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
        parameter_groups: Optional hierarchical parameter groups for partial pooling
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

    # Enable persistent compilation cache — first run compiles, subsequent
    # runs with the same model structure load from disk instantly.
    _cache_dir = Path.home() / ".cache" / "maple" / "jax_compilation_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(_cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_enable_x64", False)

    import time as _time

    # Build target likelihoods (runs bootstrap, builds forward fns)
    t0 = _time.perf_counter()
    target_likelihoods = build_target_likelihoods(targets, prior_specs, reference_db)
    t_build = _time.perf_counter() - t0

    n_likelihood_terms = sum(len(tl.entries) for tl in target_likelihoods)
    n_groups = len(parameter_groups.groups) if parameter_groups else 0
    n_grouped = len(parameter_groups.all_grouped_params) if parameter_groups else 0
    print(
        f"Built joint model in {t_build:.1f}s: {len(prior_specs)} parameters "
        f"({n_grouped} in {n_groups} hierarchical groups), "
        f"{len(targets)} targets, {n_likelihood_terms} likelihood terms"
    )

    # Run MCMC — split warmup(1) for compilation timing, then the rest
    kernel = NUTS(submodel_joint_model, dense_mass=True)

    # Phase A: single warmup step to trigger JAX compilation
    print("Compiling model (first MCMC step)...", flush=True)
    mcmc_compile = MCMC(kernel, num_warmup=1, num_samples=1, num_chains=1)
    t0 = _time.perf_counter()
    mcmc_compile.run(
        jax.random.PRNGKey(seed + 99),
        prior_specs=prior_specs,
        target_likelihoods=target_likelihoods,
        parameter_groups=parameter_groups,
    )
    t_compile = _time.perf_counter() - t0
    print(f"Compilation: {t_compile:.1f}s", flush=True)

    # Phase B: actual sampling (compilation cached, should be fast)
    print(
        f"Sampling ({num_warmup} warmup + {num_samples} samples, " f"{num_chains} chains)...",
        flush=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    t0 = _time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(seed),
        prior_specs=prior_specs,
        target_likelihoods=target_likelihoods,
        parameter_groups=parameter_groups,
    )
    t_sample = _time.perf_counter() - t0
    print(f"Sampling: {t_sample:.1f}s")

    # Diagnostics
    mcmc.print_summary()
    diagnostics = _compute_mcmc_diagnostics(mcmc, prior_specs)

    samples = mcmc.get_samples()
    # Convert JAX arrays to numpy
    samples_np = {name: np.asarray(vals) for name, vals in samples.items()}
    return samples_np, diagnostics


def run_joint_inference_vi(
    prior_specs: dict[str, PriorSpec],
    targets: list[SubmodelTarget],
    reference_db: Optional[dict[str, float]] = None,
    parameter_groups: Optional[ParameterGroupsConfig] = None,
    num_samples: int = 4000,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run joint Laplace approximation across all SubmodelTargets.

    Finds the MAP estimate, computes the Hessian to get a Gaussian
    approximation, then draws samples. Much faster than NUTS or SVI
    because it requires only one optimization pass (no iterative
    gradient steps through the guide).

    Args:
        prior_specs: Prior specifications from CSV
        targets: List of SubmodelTarget objects
        reference_db: Optional reference values
        parameter_groups: Optional hierarchical parameter groups
        num_samples: Number of posterior samples to draw from Laplace approx
        seed: Random seed

    Returns:
        Same interface as run_joint_inference: (samples_dict, diagnostics_dict)
    """
    try:
        import jax
        import jax.random
        import numpyro
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoLaplaceApproximation
        from numpyro.optim import Adam
    except ImportError as e:
        raise ImportError(
            "JAX and NumPyro are required for inference. "
            "Install with: pip install maple[inference]"
        ) from e

    _cache_dir = Path.home() / ".cache" / "maple" / "jax_compilation_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(_cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_enable_x64", False)

    import time as _time

    t0 = _time.perf_counter()
    target_likelihoods = build_target_likelihoods(targets, prior_specs, reference_db)
    t_build = _time.perf_counter() - t0

    n_likelihood_terms = sum(len(tl.entries) for tl in target_likelihoods)
    n_groups = len(parameter_groups.groups) if parameter_groups else 0
    n_grouped = len(parameter_groups.all_grouped_params) if parameter_groups else 0
    print(
        f"Built joint model in {t_build:.1f}s: {len(prior_specs)} parameters "
        f"({n_grouped} in {n_groups} hierarchical groups), "
        f"{len(targets)} targets, {n_likelihood_terms} likelihood terms"
    )

    # Build model function with args bound
    def model():
        submodel_joint_model(
            prior_specs=prior_specs,
            target_likelihoods=target_likelihoods,
            parameter_groups=parameter_groups,
        )

    guide = AutoLaplaceApproximation(model)
    svi = SVI(model, guide, Adam(0.01), loss=Trace_ELBO())

    print("Initializing Laplace (JAX compilation)...", flush=True)
    rng_key = jax.random.PRNGKey(seed)
    t0 = _time.perf_counter()
    svi_state = svi.init(rng_key)
    t_init = _time.perf_counter() - t0
    print(f"Compilation: {t_init:.1f}s", flush=True)

    # Optimize to find MAP
    n_steps = 2000
    try:
        from tqdm import trange

        pbar = trange(n_steps, desc="MAP", unit="step")
    except ImportError:
        pbar = range(n_steps)

    t0 = _time.perf_counter()
    losses = []
    for step in pbar:
        svi_state, loss = svi.update(svi_state)
        losses.append(float(loss))
        if hasattr(pbar, "set_postfix") and (step + 1) % 50 == 0:
            pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.1f}")

    t_opt = _time.perf_counter() - t0
    params = svi.get_params(svi_state)
    final_loss = np.mean(losses[-50:])
    print(f"MAP done in {t_opt:.1f}s. Final loss: {final_loss:.1f}")

    # Draw samples from the Laplace approximation (MAP + Hessian)
    print(f"Drawing {num_samples} samples from Laplace approximation...", flush=True)
    rng_key = jax.random.PRNGKey(seed + 1)
    t0 = _time.perf_counter()
    predictive = numpyro.infer.Predictive(
        model, guide=guide, params=params, num_samples=num_samples
    )
    all_samples = predictive(rng_key)
    t_sample = _time.perf_counter() - t0
    print(f"Sampling: {t_sample:.1f}s")

    samples_np = {name: np.asarray(vals) for name, vals in all_samples.items()}

    diagnostics = {
        "num_divergences": 0,
        "method": "laplace",
        "final_loss": float(final_loss),
        "per_param": {},
    }

    for name in prior_specs:
        if name in samples_np:
            diagnostics["per_param"][name] = {
                "r_hat": 1.0,
                "n_eff": float(num_samples),
                "contraction": 0.0,
                "z_score": 0.0,
            }

    return samples_np, diagnostics


def run_component_npe(
    prior_specs: dict[str, PriorSpec],
    targets: list[SubmodelTarget],
    reference_db: Optional[dict[str, float]] = None,
    parameter_groups: Optional[ParameterGroupsConfig] = None,
    num_simulations: int = 10000,
    num_posterior_samples: int = 4000,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run simulation-based inference (NPE) for components with ODE targets.

    Uses scipy.integrate.solve_ivp for forward simulation (no JAX needed),
    then trains a neural posterior estimator via sbi. Much faster than
    MCMC/Laplace for components containing custom_ode forward models.

    Args:
        prior_specs: Prior specifications from CSV
        targets: List of SubmodelTarget objects in this component
        reference_db: Optional reference values
        parameter_groups: Optional hierarchical parameter groups
        num_simulations: Number of prior-predictive simulations
        num_posterior_samples: Number of posterior samples to draw
        seed: Random seed

    Returns:
        Same interface as run_joint_inference: (samples_dict, diagnostics_dict)
    """
    import time as _time

    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(seed)

    # Collect QSP param names
    all_param_names = set()
    for t in targets:
        for p in t.calibration.parameters:
            if not p.nuisance:
                all_param_names.add(p.name)
    if parameter_groups:
        all_param_names |= parameter_groups.all_grouped_params
    qsp_params = sorted(all_param_names)

    # Collect nuisance params per target (sampled fresh each simulation)
    nuisance_specs: dict[str, dict] = {}
    for t in targets:
        for p in t.calibration.parameters:
            if p.nuisance and p.prior and p.name not in nuisance_specs:
                nuisance_specs[p.name] = {"mu": p.prior.mu, "sigma": p.prior.sigma}

    print(
        f"NPE inference: {len(qsp_params)} params, {len(targets)} targets, "
        f"{num_simulations} sims"
    )

    # Build simulator definitions for each observable
    sim_defs = []
    obs_values = []

    for target in targets:
        fm = target.calibration.forward_model
        inputs_dict = {inp.name: inp.value for inp in target.inputs}

        for entry in target.calibration.error_model:
            obs_values.append(entry.fit.median)

            if fm.type == "custom_ode":
                local_ns = {"np": np, "numpy": np}
                exec(fm.code, local_ns)  # noqa: S102
                ode_fn = local_ns["ode"]

                y0 = []
                sv_names = [sv.name for sv in fm.state_variables]
                for sv in fm.state_variables:
                    ic = sv.initial_condition
                    if hasattr(ic, "input_ref") and ic.input_ref:
                        y0.append(float(inputs_dict[ic.input_ref]))
                    elif hasattr(ic, "value") and ic.value is not None:
                        y0.append(float(ic.value))
                    else:
                        y0.append(0.0)

                t_span = [
                    float(fm.independent_variable.span[0]),
                    float(fm.independent_variable.span[1]),
                ]
                eval_pt = entry.evaluation_points[0] if entry.evaluation_points else t_span[1]

                obs_code = entry.observable.code if entry.observable else None

                sim_defs.append(
                    {
                        "type": "ode",
                        "ode_fn": ode_fn,
                        "y0": np.array(y0),
                        "t_span": t_span,
                        "eval_pt": eval_pt,
                        "inputs": inputs_dict,
                        "sv_names": sv_names,
                        "obs_code": obs_code,
                    }
                )

            elif fm.type == "algebraic":
                local_ns = {"np": np, "numpy": np}
                exec(fm.code, local_ns)  # noqa: S102
                compute_fn = local_ns["compute"]
                obs_code = entry.observable.code if entry.observable else None

                sim_defs.append(
                    {
                        "type": "algebraic",
                        "compute_fn": compute_fn,
                        "inputs": inputs_dict,
                        "obs_code": obs_code,
                    }
                )

            else:
                # Structured types — evaluate directly
                from maple.core.calibration.submodel_utils import (
                    _evaluate_structured_model,
                )

                x_val = inputs_dict.get(entry.x_input) if entry.x_input else None
                sim_defs.append(
                    {
                        "type": "structured",
                        "model": fm,
                        "inputs": inputs_dict,
                        "reference_db": reference_db,
                        "x_val": x_val,
                    }
                )

    n_obs = len(sim_defs)
    print(f"  {n_obs} observables")

    # Sample prior draws
    t0 = _time.perf_counter()
    theta = np.zeros((num_simulations, len(qsp_params)))
    for j, pname in enumerate(qsp_params):
        spec = prior_specs.get(pname)
        if spec:
            theta[:, j] = rng.lognormal(spec.mu, spec.sigma, num_simulations)
        else:
            theta[:, j] = rng.lognormal(0, 1, num_simulations)

    # Forward simulate all observables
    x = np.full((num_simulations, n_obs), np.nan)
    for i in range(num_simulations):
        param_dict = {pname: theta[i, j] for j, pname in enumerate(qsp_params)}
        # Add nuisance draws
        for nname, nspec in nuisance_specs.items():
            param_dict[nname] = rng.lognormal(nspec["mu"], nspec["sigma"])

        for obs_idx, sd in enumerate(sim_defs):
            try:
                if sd["type"] == "ode":

                    def rhs(t, y, _ode_fn=sd["ode_fn"], _p=param_dict, _inp=sd["inputs"]):
                        dy = _ode_fn(t, y, _p, _inp)
                        return np.array(dy) if isinstance(dy, (list, tuple)) else dy

                    sol = solve_ivp(
                        rhs,
                        sd["t_span"],
                        sd["y0"],
                        t_eval=[sd["eval_pt"]],
                        method="RK23",
                        rtol=1e-4,
                        atol=1e-6,
                    )
                    if not sol.success:
                        continue
                    y_final = sol.y[:, -1]

                    if sd["obs_code"]:
                        obs_ns = {"np": np, "numpy": np}
                        exec(sd["obs_code"], obs_ns)  # noqa: S102
                        y_dict = dict(zip(sd["sv_names"], y_final))
                        x[i, obs_idx] = float(obs_ns["compute"](sd["eval_pt"], y_dict, sd["y0"]))
                    else:
                        x[i, obs_idx] = float(y_final[0])

                elif sd["type"] == "algebraic":
                    raw = sd["compute_fn"](param_dict, sd["inputs"])
                    if sd["obs_code"]:
                        obs_ns = {"np": np, "numpy": np}
                        exec(sd["obs_code"], obs_ns)  # noqa: S102
                        x[i, obs_idx] = float(obs_ns["compute"](None, raw, None))
                    else:
                        x[i, obs_idx] = float(raw)

                elif sd["type"] == "structured":
                    from maple.core.calibration.submodel_utils import (
                        _evaluate_structured_model,
                    )

                    x[i, obs_idx] = float(
                        _evaluate_structured_model(
                            sd["model"],
                            param_dict,
                            sd["inputs"],
                            sd["reference_db"],
                            x_value=sd["x_val"],
                        )
                    )
            except Exception:
                continue

    t_sim = _time.perf_counter() - t0

    # Filter valid rows
    valid = np.all(np.isfinite(x) & (x > 0), axis=1)
    n_valid = int(np.sum(valid))
    theta_valid = theta[valid]
    x_valid = x[valid]
    print(
        f"  Simulated in {t_sim:.1f}s ({n_valid}/{num_simulations} valid, "
        f"{t_sim / num_simulations * 1000:.2f} ms/sim)"
    )

    if n_valid < 100:
        raise RuntimeError(f"Only {n_valid}/{num_simulations} valid simulations")

    # Train NPE in log-space
    t0 = _time.perf_counter()

    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    log_theta = np.log(theta_valid).astype(np.float32)
    log_x = np.log(x_valid).astype(np.float32)

    # Standardize
    x_mean = np.mean(log_x, axis=0)
    x_std = np.std(log_x, axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_normed = (log_x - x_mean) / x_std

    theta_tensor = torch.as_tensor(log_theta)
    x_tensor = torch.as_tensor(x_normed)

    log_lo = torch.tensor(log_theta.min(axis=0) - 1.0, dtype=torch.float32)
    log_hi = torch.tensor(log_theta.max(axis=0) + 1.0, dtype=torch.float32)
    prior_box = BoxUniform(low=log_lo, high=log_hi)

    inference = NPE(prior=prior_box)
    inference.append_simulations(theta_tensor, x_tensor)
    density_estimator = inference.train(
        training_batch_size=min(256, n_valid // 4),
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator)
    t_train = _time.perf_counter() - t0
    print(f"  NPE trained in {t_train:.1f}s")

    # Condition on observed data
    log_x_obs = np.log(np.array(obs_values, dtype=np.float64)).astype(np.float32)
    x_obs_normed = (log_x_obs - x_mean) / x_std
    x_obs_tensor = torch.as_tensor(x_obs_normed)

    log_post = posterior.sample((num_posterior_samples,), x=x_obs_tensor).numpy()
    post_samples = np.exp(log_post)

    samples_dict = {pname: post_samples[:, j] for j, pname in enumerate(qsp_params)}

    diagnostics = {
        "num_divergences": 0,
        "method": "npe",
        "num_simulations": num_simulations,
        "num_valid": n_valid,
        "sim_time_s": float(t_sim),
        "train_time_s": float(t_train),
        "per_param": {
            pname: {
                "r_hat": 1.0,
                "n_eff": float(num_posterior_samples),
                "contraction": 0.0,
                "z_score": 0.0,
            }
            for pname in qsp_params
        },
    }

    return samples_dict, diagnostics
