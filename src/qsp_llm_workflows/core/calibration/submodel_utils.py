#!/usr/bin/env python3
"""
Utility functions for instantiating and solving submodels.

These functions convert the declarative model specifications (FirstOrderDecayModel,
TwoStateModel, etc.) into executable ODE functions that can be integrated.

This module is separated from the Pydantic models to enable unit testing.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp


def get_prior_median(prior) -> Optional[float]:
    """
    Extract the median value from a Prior specification.

    Args:
        prior: Prior object with distribution, mu, sigma, lower, upper fields

    Returns:
        Median value, or None if prior is invalid/incomplete
    """
    if prior is None:
        return None

    dist = prior.distribution.value if hasattr(prior.distribution, "value") else prior.distribution

    if dist == "lognormal":
        if prior.mu is None:
            return None
        return math.exp(prior.mu)
    elif dist == "normal":
        if prior.mu is None:
            return None
        return prior.mu
    elif dist == "uniform":
        if prior.lower is None or prior.upper is None:
            return None
        return (prior.lower + prior.upper) / 2
    elif dist == "half_normal":
        if prior.sigma is None:
            return None
        # Half-normal median ≈ 0.674 * sigma
        return 0.674 * prior.sigma

    return None


def build_ode_function(
    model,
    param_values: Dict[str, float],
    input_values: Dict[str, float],
) -> Optional[Callable[[float, List[float]], List[float]]]:
    """
    Build an ODE function from a model specification.

    Args:
        model: Model specification (FirstOrderDecayModel, TwoStateModel, etc.)
        param_values: Dict mapping parameter names to values
        input_values: Dict mapping input names to values

    Returns:
        ODE function with signature (t, y) -> dydt, or None if model type not supported
    """
    model_type = model.type if hasattr(model, "type") else None

    def get_value(role, default: float = 1.0) -> float:
        """Get value from parameter or input reference."""
        if isinstance(role, str):
            # It's a parameter name
            return param_values.get(role, default)
        elif hasattr(role, "input_ref"):
            # It's an InputRef
            return input_values.get(role.input_ref, default)
        return default

    if model_type == "first_order_decay":
        k = get_value(model.rate_constant)

        def ode(t: float, y: List[float]) -> List[float]:
            return [-k * y[0]]

        return ode

    elif model_type == "exponential_growth":
        k = get_value(model.rate_constant)

        def ode(t: float, y: List[float]) -> List[float]:
            return [k * y[0]]

        return ode

    elif model_type == "two_state":
        k = get_value(model.forward_rate)

        def ode(t: float, y: List[float]) -> List[float]:
            # dA/dt = -k*A, dB/dt = +k*A
            return [-k * y[0], k * y[0]]

        return ode

    elif model_type == "saturation":
        k = get_value(model.rate_constant)

        def ode(t: float, y: List[float]) -> List[float]:
            return [k * (1 - y[0])]

        return ode

    elif model_type == "logistic":
        k = get_value(model.rate_constant)
        K = get_value(model.carrying_capacity)

        def ode(t: float, y: List[float]) -> List[float]:
            return [k * y[0] * (1 - y[0] / K)]

        return ode

    elif model_type == "michaelis_menten":
        Vmax = get_value(model.vmax)
        Km = get_value(model.km)

        def ode(t: float, y: List[float]) -> List[float]:
            return [-Vmax * y[0] / (Km + y[0])]

        return ode

    elif model_type == "custom":
        # Execute custom code to get the ODE function
        try:
            local_scope: Dict = {}
            exec(model.code, local_scope)
            custom_ode = local_scope.get("ode")
            if custom_ode is None:
                return None

            # Wrap to match expected signature
            def ode(t: float, y: List[float]) -> List[float]:
                return custom_ode(t, y, param_values, input_values)

            return ode
        except Exception:
            return None

    # direct_conversion and direct_fit don't have ODEs
    return None


def solve_submodel(
    model,
    param_values: Dict[str, float],
    input_values: Dict[str, float],
    y0: List[float],
    t_span: Tuple[float, float],
    t_eval: Optional[List[float]] = None,
) -> Optional[Dict]:
    """
    Solve a submodel ODE system.

    Args:
        model: Model specification
        param_values: Dict mapping parameter names to values
        input_values: Dict mapping input names to values
        y0: Initial conditions
        t_span: (t_start, t_end) for integration
        t_eval: Optional time points to evaluate at

    Returns:
        Dict with 't' (time points) and 'y' (state values), or None if solving fails
    """
    ode_func = build_ode_function(model, param_values, input_values)

    if ode_func is None:
        return None

    try:
        sol = solve_ivp(
            ode_func,
            t_span,
            y0,
            method="RK45",
            t_eval=t_eval,
            dense_output=True,
        )

        if not sol.success:
            return None

        # Check for NaN or Inf
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            return None

        return {
            "t": sol.t,
            "y": sol.y,
            "sol": sol,  # Include full solution object for dense_output
        }

    except Exception:
        return None


def compute_observable(
    solution: Dict,
    observable,
    y0: List[float],
    state_variable_names: Optional[List[str]] = None,
    eval_point: Optional[float] = None,
) -> Optional[float]:
    """
    Compute an observable from an ODE solution.

    Args:
        solution: Dict from solve_submodel with 't', 'y', and optionally 'sol'
        observable: Observable specification with type and state_variables
        y0: Initial conditions (for fold_change, fraction_remaining)
        state_variable_names: List of state variable names (for indexing)
        eval_point: Time point to evaluate at (uses last point if None)

    Returns:
        Observable value, or None if computation fails
    """
    if solution is None:
        return None

    # Get state values at evaluation point
    if eval_point is not None and "sol" in solution:
        y_eval = solution["sol"].sol(eval_point)
    else:
        # Use last time point
        y_eval = solution["y"][:, -1]

    if observable is None:
        # Default: return first state variable
        return float(y_eval[0])

    obs_type = observable.type.value if hasattr(observable.type, "value") else observable.type

    # Determine which state variable to use
    sv_idx = 0
    if observable.state_variables and state_variable_names:
        target_sv = observable.state_variables[0]
        if target_sv in state_variable_names:
            sv_idx = state_variable_names.index(target_sv)

    if obs_type == "final_value":
        return float(y_eval[sv_idx])

    elif obs_type == "fraction_remaining":
        if y0[sv_idx] == 0:
            return None
        return float(y_eval[sv_idx] / y0[sv_idx])

    elif obs_type == "fold_change":
        if y0[sv_idx] == 0:
            return None
        return float(y_eval[sv_idx] / y0[sv_idx])

    elif obs_type == "half_life":
        # Find time when y = y0/2
        if "sol" not in solution:
            return None
        # Binary search for t where y(t) = y0/2
        target = y0[sv_idx] / 2
        t_start, t_end = solution["t"][0], solution["t"][-1]
        sol_func = solution["sol"].sol

        for _ in range(50):  # Max iterations
            t_mid = (t_start + t_end) / 2
            y_mid = sol_func(t_mid)[sv_idx]
            if abs(y_mid - target) < 1e-6 * abs(target):
                return float(t_mid)
            if y_mid > target:
                t_start = t_mid
            else:
                t_end = t_mid

        return float((t_start + t_end) / 2)

    elif obs_type == "auc":
        # Trapezoidal integration
        t = solution["t"]
        y = solution["y"][sv_idx, :]
        return float(np.trapezoid(y, t))

    elif obs_type == "max_value":
        return float(np.max(solution["y"][sv_idx, :]))

    elif obs_type == "time_to_max":
        idx = np.argmax(solution["y"][sv_idx, :])
        return float(solution["t"][idx])

    elif obs_type == "custom":
        # Execute custom observable code
        if observable.code is None:
            return None
        try:
            local_scope: Dict = {"np": np}
            exec(observable.code, local_scope)
            compute_fn = local_scope.get("compute")
            if compute_fn is None:
                raise ValueError(
                    "Custom observable code must define a 'compute' function. "
                    "Expected signature: def compute(t, y, y0) where t is time array, "
                    "y is state array [n_states, n_times], y0 is initial conditions."
                )
            return float(compute_fn(solution["t"], solution["y"], y0))
        except ValueError:
            raise  # Re-raise ValueError (our own errors)
        except Exception as e:
            raise ValueError(
                f"Custom observable code execution failed: {e}\n"
                f"Expected signature: def compute(t, y, y0) where t is time array, "
                f"y is state array [n_states, n_times], y0 is initial conditions."
            ) from e

    # Default fallback
    return float(y_eval[sv_idx])


class PriorPredictiveError(Exception):
    """Exception raised when prior predictive check fails."""

    pass


def run_prior_predictive(
    model,
    prior,
    param_name: str,
    state_variables,
    independent_variable,
    measurement,
    input_values: Dict[str, float],
) -> float:
    """
    Run a prior predictive check: sample from prior, solve model, compute observable.

    Args:
        model: Model specification
        prior: Prior specification for the parameter
        param_name: Name of the parameter being estimated
        state_variables: List of StateVariable objects
        independent_variable: IndependentVariable object
        measurement: Measurement object with observable and evaluation_points
        input_values: Dict mapping input names to values

    Returns:
        Predicted observable value using prior median

    Raises:
        PriorPredictiveError: If any step of the computation fails
    """
    # Get prior median
    prior_median = get_prior_median(prior)
    if prior_median is None:
        raise PriorPredictiveError(
            f"Could not extract prior median for parameter '{param_name}'. "
            f"Check that prior distribution has required parameters (mu for lognormal/normal, "
            f"lower/upper for uniform, sigma for half_normal)."
        )

    # For direct_conversion, parameter IS the prediction (legacy)
    model_type = model.type if hasattr(model, "type") else None
    if model_type == "direct_conversion":
        return prior_median

    # For algebraic models, execute the forward model code
    if model_type == "algebraic":
        if not hasattr(model, "code") or not model.code:
            raise PriorPredictiveError(
                f"AlgebraicModel requires 'code' field but none is defined."
            )

        try:
            import numpy as np
            from qsp_llm_workflows.core.unit_registry import ureg

            # Build params dict with prior medians
            # Note: For multi-param models, we'd need all prior medians
            params = {param_name: prior_median}

            # Build inputs dict with units
            inputs_pint = {}
            for name, value in input_values.items():
                # Try to preserve units if they exist, otherwise use raw value
                inputs_pint[name] = value

            # Execute forward model
            local_scope = {"np": np, "numpy": np, "ureg": ureg}
            exec(model.code, local_scope)
            compute_fn = local_scope.get("compute")

            if compute_fn is None:
                raise PriorPredictiveError(
                    f"AlgebraicModel.code must define 'compute' function."
                )

            result = compute_fn(params, inputs_pint, ureg)

            # Extract magnitude if it's a Pint Quantity
            if hasattr(result, "magnitude"):
                return result.magnitude
            return result

        except PriorPredictiveError:
            raise
        except Exception as e:
            raise PriorPredictiveError(
                f"AlgebraicModel forward model execution failed: {e}"
            ) from e

    # For ODE models, solve forward
    if state_variables is None:
        raise PriorPredictiveError(
            f"ODE model '{model_type}' requires state_variables but none are defined. "
            f"Add a state_variables section to the calibration."
        )

    if independent_variable is None:
        raise PriorPredictiveError(
            f"ODE model '{model_type}' requires independent_variable but none is defined. "
            f"Add an independent_variable section with name, units, and span."
        )

    # Get initial conditions
    y0 = []
    for sv in state_variables:
        ic = sv.initial_condition
        if hasattr(ic, "value"):
            y0.append(ic.value)
        elif hasattr(ic, "input_ref"):
            y0.append(input_values.get(ic.input_ref, 1.0))
        else:
            y0.append(1.0)

    if not y0:
        raise PriorPredictiveError(
            f"No initial conditions found for state variables. "
            f"Each state variable needs an initial_condition with either a value or input_ref."
        )

    # Get time span
    if independent_variable.span is None:
        raise PriorPredictiveError(
            f"independent_variable.span is not defined. "
            f"Add span: [t_start, t_end] to the independent_variable section."
        )
    t_span = tuple(independent_variable.span)

    # Build params dict
    param_values = {param_name: prior_median}

    # Get evaluation points
    t_eval = None
    if measurement and measurement.evaluation_points:
        t_eval = measurement.evaluation_points

    # Solve ODE
    solution = solve_submodel(model, param_values, input_values, y0, t_span, t_eval)

    if solution is None:
        raise PriorPredictiveError(
            f"ODE integration failed for model '{model_type}'. "
            f"Parameter: {param_name} = {prior_median:.2e}, "
            f"y0 = {y0}, t_span = {t_span}. "
            f"Check for numerical instability or invalid parameter values."
        )

    # Compute observable
    observable = measurement.observable if measurement else None
    sv_names = [sv.name for sv in state_variables] if state_variables else None
    eval_point = t_eval[-1] if t_eval else None

    try:
        result = compute_observable(solution, observable, y0, sv_names, eval_point)
    except ValueError as e:
        raise PriorPredictiveError(str(e)) from e

    if result is None:
        obs_type = observable.type if observable else "default"
        if observable and obs_type == "custom" and observable.code is None:
            raise PriorPredictiveError(
                f"Observable type is 'custom' but no code is provided. "
                f"Either add code to compute the observable, or use a built-in type "
                f"like 'final_value', 'fraction_remaining', 'fold_change', 'auc', etc."
            )
        raise PriorPredictiveError(
            f"Observable computation returned None. "
            f"Observable type: {obs_type}. "
            f"Check that the observable is correctly defined."
        )

    return result
