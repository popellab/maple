#!/usr/bin/env python3
"""
Calibration toolkit for IsolatedSystemTarget.

Features:
1. Weighted least squares fitting using CI95 for uncertainty
2. Vector-valued observables (time-course data)
3. Profile likelihood for identifiability analysis
4. Bayesian inference with PyMC (optional)

Usage:
    # Basic calibration
    python scripts/calibrate_isolated_target.py path/to/target.yaml

    # With profile likelihood
    python scripts/calibrate_isolated_target.py path/to/target.yaml --profile

    # With Bayesian inference
    python scripts/calibrate_isolated_target.py path/to/target.yaml --bayesian
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution


def load_target(yaml_path: Path) -> dict:
    """Load IsolatedSystemTarget from YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def build_submodel_function(code: str) -> callable:
    """Execute submodel code and return the function."""
    local_scope = {}
    exec(code, local_scope)
    return local_scope["submodel"]


def build_observable_function(observable: dict) -> callable:
    """Build observable function from config, or use default (y[0])."""
    if observable.get("code"):
        local_scope = {}
        exec(observable["code"], local_scope)
        return local_scope["compute_observable"]
    else:
        # Default: return first state variable
        def default_observable(t, y, constants, ureg):
            return y[0]

        return default_observable


def estimate_sigma_from_ci95(ci95_lo: float, ci95_hi: float) -> float:
    """Estimate standard deviation from 95% CI assuming normal distribution."""
    # CI95 = mean ± 1.96 * sigma
    # width = 2 * 1.96 * sigma
    # sigma = width / (2 * 1.96)
    return (ci95_hi - ci95_lo) / (2 * 1.96)


def extract_calibration_data(target: dict) -> dict:
    """Extract all relevant data from the target for calibration."""
    submodel = target["submodel"]
    estimates = target["calibration_target_estimates"]

    # Build inputs dict
    inputs = {}
    for inp in estimates.get("inputs", []):
        val = inp["value"]
        if isinstance(val, list):
            val = val[0]
        inputs[inp["name"]] = val

    for assumption in estimates.get("assumptions", []):
        val = assumption["value"]
        if isinstance(val, list):
            val = val[0]
        inputs[assumption["name"]] = val

    # Build initial conditions
    y0 = []
    for sv in submodel["state_variables"]:
        init_input = sv["initial_value_input"]
        if init_input in inputs:
            y0.append(inputs[init_input])
        else:
            raise ValueError(f"Initial value input '{init_input}' not found")

    # Extract target data
    median = np.array(estimates["median"])
    ci95 = estimates.get("ci95", [])

    # Estimate sigma from CI95 if available
    if ci95:
        sigma = np.array([estimate_sigma_from_ci95(lo, hi) for lo, hi in ci95])
    else:
        # Default: assume 10% CV
        sigma = 0.1 * median

    # Handle index_values for time-course data
    index_values = estimates.get("index_values")
    if index_values is None:
        # Single endpoint: use t_span[1]
        obs_times = [submodel["t_span"][1]]
    else:
        obs_times = index_values

    return {
        "submodel_fn": build_submodel_function(submodel["code"]),
        "observable_fn": build_observable_function(submodel.get("observable", {})),
        "param_names": submodel["parameters"],
        "state_vars": submodel["state_variables"],
        "t_span": submodel["t_span"],
        "t_unit": submodel["t_unit"],
        "y0": np.array(y0),
        "inputs": inputs,
        "obs_times": np.array(obs_times),
        "target_median": median,
        "target_sigma": sigma,
        "target_ci95": ci95,
        "pattern": submodel.get("pattern"),
        "identifiability_notes": submodel.get("identifiability_notes"),
    }


def simulate(params: dict, data: dict) -> np.ndarray:
    """
    Simulate the submodel and return observable values at observation times.

    Args:
        params: Dict of {param_name: value}
        data: Extracted calibration data

    Returns:
        Array of observable values at each observation time
    """

    def ode(t, y):
        return data["submodel_fn"](t, list(y), params, data["inputs"])

    # Integrate over full span
    sol = solve_ivp(ode, data["t_span"], data["y0"], method="RK45", dense_output=True)

    if not sol.success:
        return np.full(len(data["obs_times"]), np.nan)

    # Evaluate at observation times
    obs_values = []
    for t_obs in data["obs_times"]:
        y_at_t = sol.sol(t_obs)
        obs = data["observable_fn"](t_obs, y_at_t, {}, None)
        obs_values.append(obs)

    return np.array(obs_values)


def negative_log_likelihood(x: np.ndarray, param_names: list, data: dict) -> float:
    """
    Compute negative log-likelihood assuming normal errors.

    NLL = sum[ 0.5 * ((y_obs - y_pred) / sigma)^2 + log(sigma) ]
    """
    params = dict(zip(param_names, x))

    try:
        y_pred = simulate(params, data)

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return 1e10

        # Weighted sum of squared residuals
        residuals = (data["target_median"] - y_pred) / data["target_sigma"]
        nll = 0.5 * np.sum(residuals**2) + np.sum(np.log(data["target_sigma"]))

        return nll

    except Exception:
        return 1e10


def run_weighted_least_squares(data: dict, method: str = "minimize") -> dict:
    """
    Run weighted least squares calibration.

    Uses CI95-derived sigma to weight residuals.
    """
    param_names = data["param_names"]
    n_params = len(param_names)

    # Parameter bounds (rates typically 0 to 10/day)
    bounds = [(1e-6, 10.0) for _ in range(n_params)]

    print(f"\n{'='*60}")
    print("WEIGHTED LEAST SQUARES CALIBRATION")
    print(f"{'='*60}")
    print(f"\nParameters: {param_names}")
    print(f"Observation times: {data['obs_times']} {data['t_unit']}")
    print(f"Target median: {data['target_median']}")
    print(f"Target sigma (from CI95): {data['target_sigma']}")

    if method == "minimize":
        # Multiple restarts for robustness
        best_result = None
        best_nll = np.inf

        for _ in range(5):
            x0 = np.random.uniform(0.1, 2.0, n_params)
            result = minimize(
                negative_log_likelihood,
                x0,
                args=(param_names, data),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result

        result = best_result

    else:  # differential_evolution
        result = differential_evolution(
            negative_log_likelihood,
            bounds,
            args=(param_names, data),
            seed=42,
            maxiter=500,
        )

    fitted_params = dict(zip(param_names, result.x))

    # Compute predictions with fitted parameters
    y_pred = simulate(fitted_params, data)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print("\nFitted parameters:")
    for name, value in fitted_params.items():
        print(f"  {name}: {value:.6f}")

    # Special case: birth-death model
    if "k_CD8_pro" in fitted_params and "k_CD8_death" in fitted_params:
        net_rate = fitted_params["k_CD8_pro"] - fitted_params["k_CD8_death"]
        print(f"\n  Net rate (k_pro - k_death): {net_rate:.6f} /{data['t_unit']}")
        if net_rate > 0:
            print(f"  Implied doubling time: {np.log(2)/net_rate:.2f} {data['t_unit']}")

    print("\nPredictions vs Targets:")
    for i, (pred, obs, sigma) in enumerate(
        zip(y_pred, data["target_median"], data["target_sigma"])
    ):
        z_score = (pred - obs) / sigma
        print(
            f"  [{i}] Predicted: {pred:.4f}, Target: {obs:.4f} ± {sigma:.4f}, z-score: {z_score:.2f}"
        )

    print(f"\nNegative log-likelihood: {result.fun:.4f}")
    print(f"Optimization success: {result.success}")

    return {
        "fitted_params": fitted_params,
        "predictions": y_pred,
        "nll": result.fun,
        "success": result.success,
        "data": data,
    }


def run_profile_likelihood(wls_result: dict, n_points: int = 50) -> dict:
    """
    Compute profile likelihood for each parameter.

    For each parameter, fix it at a range of values and optimize others.
    This reveals parameter identifiability.
    """
    data = wls_result["data"]
    fitted_params = wls_result["fitted_params"]
    param_names = data["param_names"]
    baseline_nll = wls_result["nll"]

    print(f"\n{'='*60}")
    print("PROFILE LIKELIHOOD ANALYSIS")
    print(f"{'='*60}")

    profiles = {}

    for profile_param in param_names:
        other_params = [p for p in param_names if p != profile_param]
        fitted_value = fitted_params[profile_param]

        # Range: 0.1x to 10x of fitted value
        param_range = np.linspace(max(1e-6, fitted_value * 0.1), fitted_value * 3.0, n_points)

        profile_nll = []

        for fixed_value in param_range:
            if len(other_params) == 0:
                # Only one parameter - just evaluate
                params = {profile_param: fixed_value}
                nll = negative_log_likelihood(np.array([fixed_value]), param_names, data)
            else:
                # Optimize other parameters with this one fixed
                def constrained_nll(x):
                    params = dict(zip(other_params, x))
                    params[profile_param] = fixed_value
                    x_full = [params[p] for p in param_names]
                    return negative_log_likelihood(np.array(x_full), param_names, data)

                x0 = [fitted_params[p] for p in other_params]
                bounds = [(1e-6, 10.0) for _ in other_params]

                result = minimize(
                    constrained_nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
                )
                nll = result.fun

            profile_nll.append(nll)

        profiles[profile_param] = {
            "values": param_range,
            "nll": np.array(profile_nll),
            "fitted_value": fitted_value,
        }

        # Analyze identifiability
        delta_nll = np.array(profile_nll) - baseline_nll
        # 95% CI corresponds to delta_nll < 1.92 (chi2 with 1 df)
        in_ci = delta_nll < 1.92

        if np.all(in_ci):
            print(f"\n{profile_param}: POORLY IDENTIFIABLE")
            print("  All tested values within 95% CI")
        else:
            ci_values = param_range[in_ci]
            if len(ci_values) > 0:
                print(f"\n{profile_param}: Identifiable")
                print(f"  Fitted: {fitted_value:.4f}")
                print(f"  95% CI: [{ci_values.min():.4f}, {ci_values.max():.4f}]")
            else:
                print(f"\n{profile_param}: Check results manually")

    # For birth-death model, also profile the net rate
    if "k_CD8_pro" in param_names and "k_CD8_death" in param_names:
        print("\n--- Net Rate Profile ---")
        print("Since k_pro and k_death are not individually identifiable,")
        print("profiling their DIFFERENCE (net rate):")

        net_fitted = fitted_params["k_CD8_pro"] - fitted_params["k_CD8_death"]
        net_range = np.linspace(net_fitted * 0.5, net_fitted * 1.5, n_points)

        net_nll = []
        for net_val in net_range:
            # Fix k_death = 0, set k_pro = net_val (one parameterization)
            params = {"k_CD8_pro": net_val, "k_CD8_death": 0.0}
            x_full = [params[p] for p in param_names]
            nll = negative_log_likelihood(np.array(x_full), param_names, data)
            net_nll.append(nll)

        profiles["net_rate"] = {
            "values": net_range,
            "nll": np.array(net_nll),
            "fitted_value": net_fitted,
        }

        delta_nll = np.array(net_nll) - min(net_nll)
        in_ci = delta_nll < 1.92
        ci_values = net_range[in_ci]
        print(f"  Fitted net rate: {net_fitted:.4f}")
        print(f"  95% CI: [{ci_values.min():.4f}, {ci_values.max():.4f}]")

    return profiles


def plot_profiles(profiles: dict, output_path: Optional[Path] = None):
    """Plot profile likelihood curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib not available for plotting.")
        return

    n_profiles = len(profiles)
    fig, axes = plt.subplots(1, n_profiles, figsize=(4 * n_profiles, 4))

    if n_profiles == 1:
        axes = [axes]

    for ax, (param_name, profile) in zip(axes, profiles.items()):
        values = profile["values"]
        nll = profile["nll"]
        fitted = profile["fitted_value"]

        # Plot as delta NLL from minimum
        delta_nll = nll - nll.min()

        ax.plot(values, delta_nll, "b-", linewidth=2)
        ax.axhline(y=1.92, color="r", linestyle="--", label="95% CI threshold")
        ax.axvline(x=fitted, color="g", linestyle=":", label=f"Fitted: {fitted:.3f}")

        ax.set_xlabel(param_name)
        ax.set_ylabel("Δ NLL")
        ax.set_title(f"Profile: {param_name}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(5, delta_nll.max()))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nProfile plot saved to: {output_path}")
    else:
        plt.show()


def run_bayesian_inference(data: dict, n_samples: int = 2000) -> dict:
    """
    Run Bayesian inference using PyMC.

    Returns posterior samples for all parameters.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("\nPyMC not installed. Install with: pip install pymc arviz")
        print("Skipping Bayesian inference.")
        return None

    param_names = data["param_names"]
    target_median = data["target_median"]
    target_sigma = data["target_sigma"]

    print(f"\n{'='*60}")
    print("BAYESIAN INFERENCE (PyMC)")
    print(f"{'='*60}")

    # For birth-death model, reparameterize to (net_rate, death_fraction)
    # This improves sampling when only net rate is identifiable
    is_birth_death = "k_CD8_pro" in param_names and "k_CD8_death" in param_names

    with pm.Model():
        if is_birth_death:
            # Reparameterization for identifiability
            # net_rate = k_pro - k_death (this is what's identifiable)
            # death_fraction = k_death / k_pro (weakly informed)
            net_rate = pm.Exponential("net_rate", lam=1.0)  # Prior: mean 1.0/day
            death_fraction = pm.Beta("death_fraction", alpha=1, beta=1)  # Uniform on [0,1]

            # Back-transform
            # k_pro - k_death = net_rate
            # k_death / k_pro = death_fraction
            # => k_death = death_fraction * k_pro
            # => k_pro - death_fraction * k_pro = net_rate
            # => k_pro * (1 - death_fraction) = net_rate
            # => k_pro = net_rate / (1 - death_fraction)
            k_pro = pm.Deterministic("k_CD8_pro", net_rate / (1 - death_fraction + 1e-10))
            k_death = pm.Deterministic("k_CD8_death", death_fraction * k_pro)

            param_vars = {"k_CD8_pro": k_pro, "k_CD8_death": k_death}
        else:
            # Generic: independent exponential priors
            param_vars = {}
            for pname in param_names:
                param_vars[pname] = pm.Exponential(pname, lam=1.0)

        # Likelihood: simulate and compare to data
        def simulate_theano(param_values):
            """Wrapper for simulation that works with PyMC."""
            # This is tricky - PyMC uses symbolic computation
            # For now, use a simpler approach: analytical solution for birth-death
            if is_birth_death:
                k_pro_val = param_values["k_CD8_pro"]
                k_death_val = param_values["k_CD8_death"]
                net = k_pro_val - k_death_val
                t_end = data["t_span"][1]
                y0 = data["y0"][0]
                # T(t) = T(0) * exp(net * t)
                return y0 * pm.math.exp(net * t_end)
            else:
                raise NotImplementedError("Generic Bayesian inference requires custom likelihood")

        # Predicted value
        mu = simulate_theano(param_vars)

        # Likelihood
        pm.Normal("obs", mu=mu, sigma=target_sigma[0], observed=target_median[0])

        # Sample
        print(f"\nSampling {n_samples} draws...")
        trace = pm.sample(
            n_samples, tune=1000, cores=1, progressbar=True, return_inferencedata=True
        )

    # Summary
    print(f"\n{'='*60}")
    print("POSTERIOR SUMMARY")
    print(f"{'='*60}")
    summary = az.summary(trace, var_names=param_names + (["net_rate"] if is_birth_death else []))
    print(summary)

    # Plot
    try:
        az.plot_posterior(trace, var_names=param_names + (["net_rate"] if is_birth_death else []))
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig("posterior.png", dpi=150)
        print("\nPosterior plot saved to: posterior.png")
    except Exception as e:
        print(f"\nCould not create posterior plot: {e}")

    return {"trace": trace, "summary": summary}


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate IsolatedSystemTarget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("yaml_path", type=Path, help="Path to IsolatedSystemTarget YAML")
    parser.add_argument(
        "--method",
        choices=["minimize", "differential_evolution"],
        default="minimize",
        help="Optimization method",
    )
    parser.add_argument("--profile", action="store_true", help="Run profile likelihood analysis")
    parser.add_argument("--bayesian", action="store_true", help="Run Bayesian inference with PyMC")
    parser.add_argument("--plot", type=Path, default=None, help="Save profile plot to this path")

    args = parser.parse_args()

    if not args.yaml_path.exists():
        print(f"Error: File not found: {args.yaml_path}")
        sys.exit(1)

    target = load_target(args.yaml_path)

    if not target.get("submodel"):
        print("Error: No submodel found (direct conversion mode).")
        print("Calibration requires a submodel with ODE dynamics.")
        sys.exit(1)

    # Extract data
    data = extract_calibration_data(target)

    # Run weighted least squares
    wls_result = run_weighted_least_squares(data, method=args.method)

    # Profile likelihood
    if args.profile:
        profiles = run_profile_likelihood(wls_result)
        plot_profiles(profiles, args.plot)

    # Bayesian inference
    if args.bayesian:
        run_bayesian_inference(data)

    # Print identifiability notes from the target
    if data.get("identifiability_notes"):
        print(f"\n{'='*60}")
        print("IDENTIFIABILITY NOTES (from extraction)")
        print(f"{'='*60}")
        print(data["identifiability_notes"])


if __name__ == "__main__":
    main()
