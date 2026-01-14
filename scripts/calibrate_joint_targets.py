#!/usr/bin/env python3
"""Joint Bayesian calibration for multiple IsolatedSystemTarget files.

This script performs joint inference across multiple calibration targets that
share parameters. Any parameters appearing in multiple targets are inferred
jointly, while target-specific parameters are inferred independently.

Usage:
    python scripts/calibrate_joint_targets.py target1.yaml target2.yaml ...
    python scripts/calibrate_joint_targets.py metadata-storage/to-review/*/*.yaml
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.integrate import solve_ivp


def load_target(yaml_path: Path) -> dict[str, Any] | None:
    """Load and parse an IsolatedSystemTarget YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Check for error marker
    if "error" in data and len(data) <= 2:
        print(f"  Skipping {yaml_path.name}: contains error marker")
        return None

    return data


def extract_submodel_and_data(data: dict[str, Any], target_name: str) -> dict[str, Any] | None:
    """Extract submodel code, parameters, and calibration data from target."""
    submodel = data.get("submodel")
    if submodel is None:
        print(f"  {target_name}: No submodel (direct conversion mode) - skipping")
        return None

    estimates = data.get("calibration_target_estimates", {})

    # Get median and CI95
    median = estimates.get("median", [])
    ci95 = estimates.get("ci95", [])
    units = estimates.get("units", "dimensionless")

    if not median:
        print(f"  {target_name}: No median data - skipping")
        return None

    # Extract inputs for initial conditions
    inputs = {}
    for inp in estimates.get("inputs", []):
        inputs[inp["name"]] = inp["value"]

    # Extract assumptions
    for assumption in estimates.get("assumptions", []):
        inputs[assumption["name"]] = assumption["value"]

    return {
        "name": target_name,
        "submodel_code": submodel.get("code", ""),
        "parameters": submodel.get("parameters", []),
        "state_variables": submodel.get("state_variables", []),
        "t_span": submodel.get("t_span", [0, 1]),
        "t_unit": submodel.get("t_unit", "day"),
        "observable": submodel.get("observable", {}),
        "median": np.array(median),
        "ci95": np.array(ci95) if ci95 else None,
        "units": units,
        "inputs": inputs,
    }


def compile_submodel(code: str, target_name: str):
    """Compile submodel code and return the function."""
    namespace = {"np": np, "numpy": np}
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"  {target_name}: Failed to compile submodel: {e}")
        return None

    if "submodel" not in namespace:
        print(f"  {target_name}: No 'submodel' function found in code")
        return None

    return namespace["submodel"]


def compile_observable(code: str, target_name: str):
    """Compile observable code and return the function."""
    if not code or not code.strip():
        return None

    namespace = {"np": np, "numpy": np}
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"  {target_name}: Failed to compile observable: {e}")
        return None

    if "compute_observable" not in namespace:
        print(f"  {target_name}: No 'compute_observable' function found")
        return None

    return namespace["compute_observable"]


def simulate_target(target: dict, params: dict[str, float]) -> float:
    """Simulate a target's submodel and return the observable value."""
    submodel_func = target["_submodel_func"]
    observable_func = target.get("_observable_func")

    # Get initial conditions from state variables
    y0 = []
    for sv in target["state_variables"]:
        init_input = sv.get("initial_value_input")
        if init_input and init_input in target["inputs"]:
            y0.append(target["inputs"][init_input])
        else:
            y0.append(1.0)  # default

    t_span = target["t_span"]
    t_eval = np.linspace(t_span[0], t_span[1], 100)

    # Solve ODE
    try:
        sol = solve_ivp(
            lambda t, y: submodel_func(t, y, params, target["inputs"]),
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
        )
    except Exception:
        return np.nan

    if not sol.success:
        return np.nan

    # Compute observable
    if observable_func is not None:

        class MockUreg:
            """Mock unit registry that returns dimensionless values."""

            def __call__(self, unit_str):
                return 1.0

            def __getattr__(self, name):
                return 1.0

        obs_constants = {}
        for const in target["observable"].get("constants", []):
            obs_constants[const["name"]] = const["value"]

        try:
            result = observable_func(sol.t[-1], sol.y[:, -1], obs_constants, MockUreg())
            return float(result)
        except Exception:
            return sol.y[0, -1]
    else:
        # Default: return first state variable at final time
        return sol.y[0, -1]


def negative_log_likelihood(
    param_values: np.ndarray, param_names: list[str], targets: list[dict]
) -> float:
    """Compute negative log-likelihood across all targets."""
    params = dict(zip(param_names, param_values))

    nll = 0.0
    for target in targets:
        predicted = simulate_target(target, params)
        if np.isnan(predicted):
            return 1e10

        observed = target["median"][0]
        sigma = target["sigma"]

        # Gaussian log-likelihood
        nll += 0.5 * ((predicted - observed) / sigma) ** 2

    return nll


def run_joint_bayesian_inference(
    targets: list[dict], all_params: list[str], n_samples: int = 2000, n_tune: int = 1000
):
    """Run joint Bayesian inference across all targets using PyMC."""
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        print("\nPyMC not installed. Install with: pip install pymc arviz")
        sys.exit(1)

    print(f"\nJoint inference over {len(all_params)} parameters: {all_params}")
    print(f"Using {len(targets)} calibration targets")

    # Estimate sigma from CI95 for each target
    for target in targets:
        if target["ci95"] is not None and len(target["ci95"]) > 0:
            ci = target["ci95"][0]
            target["sigma"] = (ci[1] - ci[0]) / (2 * 1.96)
        else:
            target["sigma"] = 0.1 * abs(target["median"][0])
        print(f"  {target['name']}: median={target['median'][0]:.4g}, sigma={target['sigma']:.4g}")

    # Find good starting point via optimization
    print("\nFinding MAP estimate...")
    from scipy.optimize import minimize

    x0 = np.ones(len(all_params)) * 0.5
    bounds = [(1e-6, 10.0) for _ in all_params]

    result = minimize(
        negative_log_likelihood,
        x0,
        args=(all_params, targets),
        method="L-BFGS-B",
        bounds=bounds,
    )

    map_estimate = dict(zip(all_params, result.x))
    print("MAP estimates:")
    for name, val in map_estimate.items():
        print(f"  {name}: {val:.4f}")

    # Build PyMC model using blackbox likelihood
    print("\nBuilding PyMC model...")

    with pm.Model():
        # Create parameter variables with weakly informative priors
        param_vars = {}
        for param in all_params:
            # Use MAP estimate to set prior scale
            map_val = map_estimate.get(param, 1.0)
            param_vars[param] = pm.HalfNormal(param, sigma=max(2.0, 3 * map_val))

        # Custom likelihood using pm.Potential
        def joint_logp(param_dict):
            """Compute log-probability for all targets."""
            total_logp = 0.0
            for target in targets:
                # Build params dict for this target
                target_params = {p: param_dict[p] for p in target["parameters"]}
                predicted = simulate_target(target, target_params)

                if np.isnan(predicted):
                    return -1e10

                observed = target["median"][0]
                sigma = target["sigma"]

                # Gaussian log-likelihood
                total_logp += -0.5 * ((predicted - observed) / sigma) ** 2

            return total_logp

        # Use pm.Potential with a custom Theano/PyTensor Op for the simulation
        # For simplicity, we use NUTS with numerical gradients via external sampling
        # This requires pytensor's `as_op` decorator

        try:
            import pytensor.tensor as pt
            from pytensor.graph import Op

            class SimulationOp(Op):
                """Custom Op for ODE simulation."""

                itypes = [pt.dvector]
                otypes = [pt.dscalar]

                def __init__(self, targets, param_names):
                    self.targets = targets
                    self.param_names = param_names

                def perform(self, node, inputs, outputs):
                    param_values = inputs[0]
                    params = dict(zip(self.param_names, param_values))

                    total_logp = 0.0
                    for target in self.targets:
                        target_params = {p: params[p] for p in target["parameters"] if p in params}
                        predicted = simulate_target(target, target_params)

                        if np.isnan(predicted):
                            outputs[0][0] = np.array(-1e10)
                            return

                        observed = target["median"][0]
                        sigma = target["sigma"]
                        total_logp += -0.5 * ((predicted - observed) / sigma) ** 2

                    outputs[0][0] = np.array(total_logp)

            sim_op = SimulationOp(targets, all_params)

            # Stack parameters into vector
            param_vector = pt.stack([param_vars[p] for p in all_params])
            logp = sim_op(param_vector)
            pm.Potential("likelihood", logp)

            # Sample using Metropolis (NUTS requires gradients)
            print(f"\nSampling {n_samples} draws with {n_tune} tuning steps...")
            print("(Using Metropolis sampler for ODE-based likelihood)")
            trace = pm.sample(
                n_samples,
                tune=n_tune,
                cores=1,
                random_seed=42,
                step=pm.Metropolis(),
                progressbar=True,
                return_inferencedata=True,
                initvals=map_estimate,
            )

        except Exception as e:
            print(f"\nFalling back to manual MCMC due to: {e}")
            trace = _manual_mcmc(targets, all_params, map_estimate, n_samples, n_tune)
            return trace

    # Print summary
    print("\n" + "=" * 60)
    print("POSTERIOR SUMMARY")
    print("=" * 60)
    summary = az.summary(trace, var_names=all_params)
    print(summary)

    # Print pairwise correlations for shared parameters
    if len(all_params) > 1:
        print("\nPosterior correlations:")
        samples = {p: trace.posterior[p].values.flatten() for p in all_params}
        for i, p1 in enumerate(all_params):
            for p2 in all_params[i + 1 :]:
                corr = np.corrcoef(samples[p1], samples[p2])[0, 1]
                print(f"  {p1} vs {p2}: r = {corr:.3f}")

    return trace


def _manual_mcmc(
    targets: list[dict], all_params: list[str], start: dict[str, float], n_samples: int, n_tune: int
):
    """Fallback manual Metropolis-Hastings MCMC."""
    import arviz as az

    print("\nRunning manual Metropolis-Hastings...")

    # Initialize
    current = np.array([start[p] for p in all_params])
    current_logp = -negative_log_likelihood(current, all_params, targets)

    # Adaptive proposal scale
    proposal_scale = 0.1 * current

    samples = []
    accepted = 0

    total_steps = n_tune + n_samples

    for i in range(total_steps):
        # Propose
        proposal = current + proposal_scale * np.random.randn(len(all_params))
        proposal = np.maximum(proposal, 1e-6)  # Keep positive

        proposal_logp = -negative_log_likelihood(proposal, all_params, targets)

        # Accept/reject
        log_alpha = proposal_logp - current_logp
        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            current_logp = proposal_logp
            if i >= n_tune:
                accepted += 1

        if i >= n_tune:
            samples.append(current.copy())

        # Adapt during tuning
        if i < n_tune and i > 0 and i % 100 == 0:
            recent_accept_rate = accepted / max(1, i - n_tune + 100)
            if recent_accept_rate < 0.2:
                proposal_scale *= 0.8
            elif recent_accept_rate > 0.5:
                proposal_scale *= 1.2

        if (i + 1) % 500 == 0:
            print(f"  Step {i + 1}/{total_steps}")

    samples = np.array(samples)
    accept_rate = accepted / n_samples
    print(f"  Acceptance rate: {accept_rate:.1%}")

    # Convert to ArviZ InferenceData
    posterior_dict = {p: samples[:, i].reshape(1, -1) for i, p in enumerate(all_params)}
    trace = az.from_dict(posterior=posterior_dict)

    return trace


def analyze_identifiability(targets: list[dict], all_params: list[str]):
    """Analyze parameter identifiability via profile likelihood."""
    print("\n" + "=" * 60)
    print("IDENTIFIABILITY ANALYSIS (Profile Likelihood)")
    print("=" * 60)

    from scipy.optimize import minimize

    # Estimate sigma from CI95 for each target (needed for likelihood)
    for target in targets:
        if "sigma" not in target:
            if target["ci95"] is not None and len(target["ci95"]) > 0:
                ci = target["ci95"][0]
                target["sigma"] = (ci[1] - ci[0]) / (2 * 1.96)
            else:
                target["sigma"] = 0.1 * abs(target["median"][0])

    print("\nTarget data:")
    for target in targets:
        print(f"  {target['name']}: median={target['median'][0]:.4g}, sigma={target['sigma']:.4g}")

    # First find global optimum
    x0 = np.ones(len(all_params)) * 0.5
    bounds = [(1e-6, 10.0) for _ in all_params]

    result = minimize(
        negative_log_likelihood,
        x0,
        args=(all_params, targets),
        method="L-BFGS-B",
        bounds=bounds,
    )

    opt_params = dict(zip(all_params, result.x))
    opt_nll = result.fun

    print(f"\nOptimal NLL: {opt_nll:.4f}")
    print("Optimal parameters:")
    for name, val in opt_params.items():
        print(f"  {name}: {val:.4f}")

    # Profile each parameter
    threshold = 1.92  # chi2(1) / 2 for 95% CI

    for param_idx, param_name in enumerate(all_params):
        print(f"\nProfiling {param_name}...")

        opt_val = result.x[param_idx]
        test_values = np.linspace(opt_val * 0.1, opt_val * 3, 50)
        profile_nll = []

        other_indices = [i for i in range(len(all_params)) if i != param_idx]

        for test_val in test_values:
            if len(other_indices) == 0:
                # Only one parameter
                nll = negative_log_likelihood(np.array([test_val]), all_params, targets)
            else:
                # Optimize other parameters
                def profile_obj(other_vals):
                    full_vals = np.zeros(len(all_params))
                    full_vals[param_idx] = test_val
                    for i, idx in enumerate(other_indices):
                        full_vals[idx] = other_vals[i]
                    return negative_log_likelihood(full_vals, all_params, targets)

                other_x0 = [result.x[i] for i in other_indices]
                other_bounds = [bounds[i] for i in other_indices]

                res = minimize(profile_obj, other_x0, method="L-BFGS-B", bounds=other_bounds)
                nll = res.fun

            profile_nll.append(nll)

        profile_nll = np.array(profile_nll)
        delta_nll = profile_nll - opt_nll

        # Find 95% CI
        in_ci = delta_nll < threshold
        if np.any(in_ci):
            ci_values = test_values[in_ci]
            ci_lo, ci_hi = ci_values.min(), ci_values.max()
            ci_width = ci_hi - ci_lo

            # Identifiability assessment
            if ci_width / opt_val > 2.0:
                status = "POORLY IDENTIFIABLE"
            elif ci_width / opt_val > 0.5:
                status = "MODERATELY IDENTIFIABLE"
            else:
                status = "WELL IDENTIFIABLE"

            print(f"  {param_name}: {opt_val:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]) - {status}")
        else:
            print(f"  {param_name}: {opt_val:.4f} - IDENTIFIABILITY UNCLEAR")


def main():
    parser = argparse.ArgumentParser(
        description="Joint Bayesian calibration for multiple IsolatedSystemTarget files"
    )
    parser.add_argument(
        "yaml_files",
        nargs="+",
        type=Path,
        help="YAML files containing IsolatedSystemTarget data",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of posterior samples (default: 2000)",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Number of tuning steps (default: 1000)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profile likelihood identifiability analysis",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("JOINT BAYESIAN CALIBRATION")
    print("=" * 60)

    # Load all targets
    targets = []
    for yaml_path in args.yaml_files:
        if not yaml_path.exists():
            print(f"  Warning: {yaml_path} not found, skipping")
            continue

        print(f"\nLoading {yaml_path.name}...")
        data = load_target(yaml_path)
        if data is None:
            continue

        target = extract_submodel_and_data(data, yaml_path.stem)
        if target is None:
            continue

        # Compile submodel
        submodel_func = compile_submodel(target["submodel_code"], target["name"])
        if submodel_func is None:
            continue
        target["_submodel_func"] = submodel_func

        # Compile observable if present
        obs_code = target["observable"].get("code", "")
        if obs_code:
            obs_func = compile_observable(obs_code, target["name"])
            target["_observable_func"] = obs_func

        targets.append(target)
        print(f"  Loaded: {target['name']}")
        print(f"    Parameters: {target['parameters']}")
        print(f"    t_span: {target['t_span']} {target['t_unit']}")
        print(f"    Median: {target['median']}")

    if not targets:
        print("\nNo valid targets found. Exiting.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(targets)} targets loaded")
    print("=" * 60)

    # Collect all unique parameters and count occurrences
    param_counts = Counter()
    for target in targets:
        param_counts.update(target["parameters"])

    all_params = sorted(param_counts.keys())

    print("\nParameters across targets:")
    for param, count in param_counts.most_common():
        shared = "(shared)" if count > 1 else "(single target)"
        print(f"  {param}: appears in {count} target(s) {shared}")

    # Run identifiability analysis if requested
    if args.profile:
        analyze_identifiability(targets, all_params)

    # Run joint Bayesian inference
    run_joint_bayesian_inference(targets, all_params, n_samples=args.samples, n_tune=args.tune)

    print("\nDone!")


if __name__ == "__main__":
    main()
