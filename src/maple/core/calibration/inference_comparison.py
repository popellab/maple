"""Compare single-target vs joint inference results for submodel targets.

Runs MCMC inference on each target individually, then jointly across all
targets, and produces a comparison report showing how joint inference
shifts parameter estimates relative to single-target posteriors and the
original CSV priors.

Can be run standalone::

    python -m maple.core.calibration.inference_comparison \\
        --priors-csv parameters/pdac_priors.csv \\
        --submodel-dir calibration_targets/submodel_targets/ \\
        --output results/inference_comparison/report.md

Or called programmatically::

    from maple.core.calibration.inference_comparison import run_comparison
    report = run_comparison(priors_csv, submodel_dir)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# MCMC result caching
# =============================================================================


def _compute_hash(*contents: str | bytes) -> str:
    """Compute a short SHA256 hash from concatenated contents."""
    h = hashlib.sha256()
    for c in contents:
        if isinstance(c, str):
            c = c.encode()
        h.update(c)
    return h.hexdigest()[:16]


def _cache_dir(submodel_dir: Path) -> Path:
    d = submodel_dir / ".compare_cache"
    d.mkdir(exist_ok=True)
    return d


def _save_cache(path: Path, data: dict) -> None:
    """Save MCMC results to JSON cache file."""

    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(path, "w") as f:
        json.dump(data, f, default=_convert)


def _load_cache(path: Path) -> dict | None:
    """Load cached MCMC results if they exist."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _contraction(prior_sigma: float, posterior_sigma: float) -> float:
    """Compute contraction: 1 - (posterior_sigma / prior_sigma)^2.

    +1 = fully constrained (posterior collapsed to a point).
     0 = no information gained.
    <0 = posterior wider than prior (data conflicts or model issues).
    """
    if prior_sigma <= 0:
        return 0.0
    return 1.0 - (posterior_sigma / prior_sigma) ** 2


def _z_score(prior_mu: float, posterior_mu: float, prior_sigma: float) -> float:
    """Shift in log-space medians, normalized by prior sigma."""
    if prior_sigma <= 0:
        return 0.0
    return abs(posterior_mu - prior_mu) / prior_sigma


def _build_structured_results(
    csv_priors: dict,
    joint_fits: dict,
    single_results: dict,
    joint_diag: dict,
    all_param_names: set,
    n_targets: int,
    num_samples: int,
) -> dict:
    """Build structured dict for YAML serialization."""
    from datetime import datetime, timezone

    parameters = {}
    for pname in sorted(all_param_names):
        prior_spec = csv_priors.get(pname)
        prior = None
        if prior_spec:
            median = float(np.exp(prior_spec.mu)) if prior_spec.mu is not None else None
            prior = {
                "median": median,
                "sigma": float(prior_spec.sigma) if prior_spec.sigma is not None else None,
            }

        joint = None
        if pname in joint_fits:
            jf = joint_fits[pname]
            joint = {
                "median": float(jf["median"]),
                "sigma": float(jf["sigma"]),
                "cv": float(jf["cv"]),
                "contraction": round(float(jf["contraction"]), 4),
                "z_score": round(float(jf["z_score"]), 4),
                "distribution": jf["dist"],
            }

        singles = []
        if pname in single_results:
            for entry in single_results[pname]:
                singles.append(
                    {
                        "target_id": entry["target_id"],
                        "median": float(entry["median"]),
                        "sigma": float(entry["sigma"]),
                        "cv": float(entry["cv"]),
                        "contraction": round(float(entry["contraction"]), 4),
                        "z_score": round(float(entry["z_score"]), 4),
                    }
                )

        # Flags
        flags = []
        if prior and joint and prior["sigma"] > 0:
            if joint["sigma"] / prior["sigma"] > 0.8:
                flags.append("weak_contraction")
        if joint and joint["z_score"] > 2.0:
            flags.append("high_z_score")

        # Tension: single-target medians disagree by >3x
        tension = False
        if len(singles) >= 2:
            medians = [s["median"] for s in singles]
            if min(medians) > 0:
                tension = max(medians) / min(medians) > 3.0

        diag = joint_diag.get("per_param", {}).get(pname)
        mcmc_diag = None
        if diag:
            mcmc_diag = {
                "n_eff": round(float(diag.get("n_eff", 0))),
                "r_hat": round(float(diag.get("r_hat", 0)), 4),
            }

        param_entry = {"prior": prior, "joint": joint}
        if singles:
            param_entry["single_targets"] = singles
        if tension:
            param_entry["tension"] = True
        if flags:
            param_entry["flags"] = flags
        if mcmc_diag:
            param_entry["mcmc_diagnostics"] = mcmc_diag

        # SBC results (NPE components only)
        sbc = joint_diag.get("sbc", {}).get(pname)
        if sbc:
            param_entry["sbc"] = sbc

        parameters[pname] = param_entry

    # Aggregate PPC stats
    ppc_covered = joint_diag.get("ppc_n_covered", 0)
    ppc_total = joint_diag.get("ppc_n_total", 0)

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_targets": n_targets,
            "n_parameters": len(all_param_names),
            "method": "component_wise",
            "num_samples": num_samples,
            "num_divergences": joint_diag.get("num_divergences", 0),
            "ppc_coverage": float(ppc_covered / ppc_total) if ppc_total else None,
            "ppc_n_covered": ppc_covered,
            "ppc_n_total": ppc_total,
        },
        "parameters": parameters,
    }


def _find_components(targets, param_groups):
    """Find connected components of targets linked by shared params or groups.

    Returns list of dicts: [{"params": set[str], "targets": list[SubmodelTarget]}]
    """
    from collections import defaultdict, deque

    # Build param -> target mapping (QSP params only)
    param_to_targets = defaultdict(list)
    target_to_params = {}
    for t in targets:
        t_params = set()
        for p in t.calibration.parameters:
            if not p.nuisance:
                t_params.add(p.name)
                param_to_targets[p.name].append(t)
        target_to_params[id(t)] = t_params

    # Build group membership edges
    group_edges = defaultdict(set)  # param -> set of group-linked params
    if param_groups:
        for g in param_groups.groups:
            members = {m.name for m in g.members}
            for m in members:
                group_edges[m] = group_edges[m] | members

    # All params (from targets + groups)
    all_params = set(param_to_targets.keys())
    if param_groups:
        all_params |= param_groups.all_grouped_params

    # BFS to find connected components
    visited = set()
    components = []
    for start_p in sorted(all_params):
        if start_p in visited:
            continue
        comp_params = set()
        comp_targets_set = set()
        queue = deque([start_p])
        while queue:
            p = queue.popleft()
            if p in visited:
                continue
            visited.add(p)
            comp_params.add(p)
            # Link via shared targets
            for t in param_to_targets.get(p, []):
                comp_targets_set.add(id(t))
                for p2 in target_to_params[id(t)]:
                    if p2 not in visited:
                        queue.append(p2)
            # Link via group membership
            for p2 in group_edges.get(p, set()):
                if p2 not in visited:
                    queue.append(p2)

        comp_targets = [t for t in targets if id(t) in comp_targets_set]
        components.append({"params": comp_params, "targets": comp_targets})

    return components


def run_comparison(
    priors_csv: str | Path,
    submodel_dir: str | Path,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
    num_samples: int = 4000,
    parameter_groups_path: str | Path | None = None,
) -> str:
    """Run component-wise NPE inference, return comparison report.

    Args:
        priors_csv: Path to priors CSV.
        submodel_dir: Directory containing SubmodelTarget YAMLs.
        glob_pattern: Glob for YAML files.
        num_samples: Number of posterior samples per component.
        parameter_groups_path: Optional path to parameter_groups.yaml.

    Returns:
        Markdown-formatted comparison report.
    """
    import yaml

    from maple.core.calibration.submodel_inference import (
        load_priors_from_csv,
    )
    from maple.core.calibration.submodel_target import SubmodelTarget
    from maple.core.calibration.yaml_to_prior import (
        fit_distributions,
    )

    from maple.core.calibration.parameter_groups import load_parameter_groups

    priors_csv = Path(priors_csv)
    submodel_dir = Path(submodel_dir)

    # Load parameter groups if provided (or auto-discover in submodel_dir)
    param_groups = None
    if parameter_groups_path is not None:
        param_groups = load_parameter_groups(Path(parameter_groups_path))
    else:
        auto_path = submodel_dir / "parameter_groups.yaml"
        if auto_path.exists():
            param_groups = load_parameter_groups(auto_path)
    if param_groups and param_groups.groups:
        logger.info(
            "Loaded %d parameter groups (%d params)",
            len(param_groups.groups),
            len(param_groups.all_grouped_params),
        )

    yaml_files = sorted(submodel_dir.glob(glob_pattern))
    # Exclude parameter_groups.yaml from target list
    yaml_files = [f for f in yaml_files if f.name != "parameter_groups.yaml"]
    if not yaml_files:
        return f"No YAML files found matching {glob_pattern} in {submodel_dir}"

    csv_priors = load_priors_from_csv(priors_csv)
    cache = _cache_dir(submodel_dir)

    # ── Load and parse all targets ──
    targets = []
    yaml_contents = {}  # {filename: raw_content} for cache hashing
    for yf in yaml_files:
        try:
            raw = yf.read_text()
            yaml_contents[yf.name] = raw
            data = yaml.safe_load(raw)
            target = SubmodelTarget.model_validate(data)
            targets.append(target)
        except Exception as e:
            logger.warning("Failed to load %s for joint: %s", yf.name, e)

    all_param_names = set()
    for t in targets:
        for p in t.calibration.parameters:
            if not p.nuisance:
                all_param_names.add(p.name)
    if param_groups:
        all_param_names |= param_groups.all_grouped_params

    priors_content = priors_csv.read_text()
    mcmc_config_str = f"npe:{num_samples}"
    method_str = "npe"

    # ── Phase 1: Component-wise joint inference ──
    # Find connected components (params linked by shared targets or group membership)
    components = _find_components(targets, param_groups)
    logger.info(
        "Phase 1: %d components (largest: %d params, %d targets)",
        len(components),
        max(len(c["params"]) for c in components) if components else 0,
        max(len(c["targets"]) for c in components) if components else 0,
    )

    joint_fits: dict[str, dict] = {}
    joint_diag: dict = {"num_divergences": 0, "per_param": {}}
    joint_samples_all: dict[str, list] = {}

    # Filter out components with no targets (e.g., group-only with no data)
    active_components = [c for c in components if c["targets"]]
    logger.info("Phase 1: %d active components", len(active_components))

    for ci, comp in enumerate(active_components):
        comp_targets = comp["targets"]
        comp_params = comp["params"]

        # Cache per component
        comp_content = "".join(
            yaml_contents.get(t.primary_data_source.source_tag + ".yaml", "") for t in comp_targets
        )
        # Fall back to target_id-based content hashing
        if not comp_content:
            comp_content = "|".join(sorted(t.target_id for t in comp_targets))
        comp_hash = _compute_hash(comp_content, priors_content, mcmc_config_str, method_str)
        comp_cache_path = cache / f"component_{ci}_{comp_hash}.json"

        cached_comp = _load_cache(comp_cache_path)
        if cached_comp is not None:
            for k, v in cached_comp.get("fits", {}).items():
                joint_fits[k] = v
            comp_diag = cached_comp.get("diag", {})
            joint_diag["num_divergences"] += comp_diag.get("num_divergences", 0)
            for k, v in comp_diag.get("per_param", {}).items():
                joint_diag["per_param"][k] = v
            for k, v in cached_comp.get("samples", {}).items():
                joint_samples_all[k] = v
            continue

        # Build prior specs for this component
        comp_prior_specs = {k: v for k, v in csv_priors.items() if k in comp_params}

        # Find relevant parameter groups for this component
        comp_groups = None
        if param_groups:
            from maple.core.calibration.parameter_groups import (
                ParameterGroupsConfig,
            )

            relevant = [
                g for g in param_groups.groups if any(m.name in comp_params for m in g.members)
            ]
            if relevant:
                comp_groups = ParameterGroupsConfig(groups=relevant)

        n_p = len(comp_params)
        n_t = len(comp_targets)
        logger.info(
            "  Component %d/%d: %d params, %d targets",
            ci + 1,
            len(active_components),
            n_p,
            n_t,
        )

        has_ode = any(t.calibration.forward_model.type == "custom_ode" for t in comp_targets)
        is_singleton = len(comp_targets) == 1 and not has_ode

        try:
            if has_ode:
                from maple.core.calibration.submodel_inference import (
                    run_component_npe,
                )

                logger.info("    (NPE — component has ODE targets)")
                comp_samples, comp_diag = run_component_npe(
                    comp_prior_specs,
                    comp_targets,
                    parameter_groups=comp_groups,
                    num_posterior_samples=num_samples,
                )
            elif is_singleton:
                from maple.core.calibration.yaml_to_prior import process_yaml

                logger.info("    (single-target MCMC)")
                # Find the YAML file for this target
                target = comp_targets[0]
                yf = None
                for f in yaml_files:
                    if yaml_contents.get(f.name, "").find(target.target_id) >= 0:
                        yf = f
                        break
                if yf is None:
                    logger.warning("    Could not find YAML for %s", target.target_id)
                    continue

                results = process_yaml(yf, priors_csv=priors_csv)
                comp_samples = {}
                comp_diag = {"num_divergences": 0, "per_param": {}}
                for r in results:
                    if "error" in r:
                        continue
                    pname = r["name"]
                    if "param_samples" in r:
                        comp_samples[pname] = np.array(r["param_samples"])
                    else:
                        best = r["best_dist"]
                        if best.name == "lognormal":
                            comp_samples[pname] = np.random.lognormal(
                                np.log(best.median), best.params["sigma"], num_samples
                            )
                        else:
                            comp_samples[pname] = np.full(num_samples, best.median)
                    mcmc_diag = r.get("mcmc_diagnostics", {})
                    comp_diag["num_divergences"] += mcmc_diag.get("num_divergences", 0)
                    comp_diag["per_param"][pname] = {
                        "r_hat": float(mcmc_diag.get("r_hat", 1.0)),
                        "n_eff": float(mcmc_diag.get("n_eff", 0)),
                        "contraction": float(mcmc_diag.get("contraction", 0)),
                        "z_score": float(mcmc_diag.get("z_score", 0)),
                    }
            else:
                from maple.core.calibration.submodel_inference import (
                    run_joint_inference,
                )

                logger.info("    (joint MCMC — %d targets)", n_t)
                comp_samples, comp_diag = run_joint_inference(
                    comp_prior_specs,
                    comp_targets,
                    parameter_groups=comp_groups,
                    num_warmup=500,
                    num_samples=num_samples,
                    num_chains=1,
                )
        except Exception as e:
            logger.warning("  Component %d failed: %s", ci + 1, e)
            continue

        # Fit distributions and accumulate
        comp_fits = {}
        for pname in sorted(comp_params):
            if pname not in comp_samples:
                continue
            fits = fit_distributions(comp_samples[pname])
            if not fits:
                continue
            best = fits[0]

            if best.name == "lognormal":
                post_sigma = best.params["sigma"]
            else:
                post_sigma = np.sqrt(np.log(1 + best.cv**2))

            prior_sigma = csv_priors[pname].sigma if pname in csv_priors else 1.0
            prior_mu = csv_priors[pname].mu if pname in csv_priors else 0.0
            post_mu = np.log(best.median)

            comp_fits[pname] = {
                "median": best.median,
                "cv": best.cv,
                "sigma": post_sigma,
                "dist": best.name,
                "contraction": _contraction(prior_sigma, post_sigma),
                "z_score": _z_score(prior_mu, post_mu, prior_sigma),
            }
            joint_fits[pname] = comp_fits[pname]

        # Run PPC for non-NPE components (NPE does its own PPC internally)
        if not has_ode and "ppc_coverage" not in comp_diag:
            from maple.core.calibration.submodel_inference import (
                build_numpy_forward_fns,
                build_target_likelihoods,
            )

            ppc_fns = []
            ppc_obs = []
            comp_tls = build_target_likelihoods(comp_targets, comp_prior_specs)
            for target, tl_entry in zip(comp_targets, comp_tls):
                fns = build_numpy_forward_fns(target)
                for fn, le in zip(fns, tl_entry.entries):
                    ppc_fns.append(fn)
                    ppc_obs.append(float(le.fit.median))

            n_ppc = min(200, len(comp_samples.get(next(iter(comp_samples), ""), [])))
            if n_ppc > 0 and ppc_fns:
                nuisance = {}
                for t in comp_targets:
                    for p in t.calibration.parameters:
                        if p.nuisance and p.prior:
                            nuisance[p.name] = (p.prior.mu, p.prior.sigma)
                rng = np.random.default_rng(42)
                n_covered = 0
                for obs_idx, fn in enumerate(ppc_fns):
                    preds = []
                    for i in range(n_ppc):
                        pd = {
                            pn: float(comp_samples[pn][i])
                            for pn in comp_samples
                            if i < len(comp_samples[pn])
                        }
                        for nn, (mu, sig) in nuisance.items():
                            pd[nn] = float(rng.lognormal(mu, sig))
                        try:
                            preds.append(float(fn(pd)))
                        except Exception:
                            pass
                    if len(preds) >= 10:
                        lo, hi = np.percentile(preds, [2.5, 97.5])
                        if lo <= ppc_obs[obs_idx] <= hi:
                            n_covered += 1
                comp_diag["ppc_coverage"] = float(n_covered / len(ppc_fns)) if ppc_fns else 0
                comp_diag["ppc_n_covered"] = n_covered
                comp_diag["ppc_n_total"] = len(ppc_fns)
                logger.info("    PPC: %d/%d covered", n_covered, len(ppc_fns))

        joint_diag["num_divergences"] += comp_diag.get("num_divergences", 0)
        for k, v in comp_diag.get("per_param", {}).items():
            joint_diag["per_param"][k] = v
        # Accumulate SBC results
        if "sbc" in comp_diag:
            if "sbc" not in joint_diag:
                joint_diag["sbc"] = {}
            joint_diag["sbc"].update(comp_diag["sbc"])
        # Accumulate PPC
        joint_diag["ppc_n_covered"] = joint_diag.get("ppc_n_covered", 0) + comp_diag.get(
            "ppc_n_covered", 0
        )
        joint_diag["ppc_n_total"] = joint_diag.get("ppc_n_total", 0) + comp_diag.get(
            "ppc_n_total", 0
        )

        comp_samples_list = {k: v for k, v in comp_samples.items()}
        for k, v in comp_samples_list.items():
            joint_samples_all[k] = v

        _save_cache(
            comp_cache_path,
            {
                "fits": comp_fits,
                "diag": comp_diag,
                "samples": comp_samples_list,
            },
        )

    logger.info("Phase 1: done (%d params fitted)", len(joint_fits))

    # Single-target results no longer computed separately — NPE handles
    # everything component-wise. Keep empty dicts for report compatibility.
    single_results: dict[str, list[dict]] = {}

    # ── Save structured results ──
    structured = _build_structured_results(
        csv_priors=csv_priors,
        joint_fits=joint_fits,
        single_results=single_results,
        joint_diag=joint_diag,
        all_param_names=all_param_names,
        n_targets=len(targets),
        num_samples=num_samples,
    )
    results_path = submodel_dir / "compare_inference_results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(structured, f, default_flow_style=False, sort_keys=False)
    logger.info("Structured results saved to %s", results_path)

    # ── Build comparison report ──
    lines = [
        "# Inference Comparison Report",
        "",
        f"**Targets:** {len(targets)}",
        f"**Parameters:** {len(all_param_names)}",
        f"**Method:** Component-wise NPE ({num_samples} posterior samples)",
        "",
    ]

    # Per-parameter detailed comparison
    for pname in sorted(all_param_names):
        lines.append(f"### `{pname}`")
        lines.append("")

        # CSV prior
        if pname in csv_priors:
            spec = csv_priors[pname]
            csv_median = np.exp(spec.mu)
            lines.append(
                f"**CSV prior:** median={csv_median:.4g}, "
                f"sigma={spec.sigma:.2f}, "
                f"CV={np.sqrt(np.exp(spec.sigma**2) - 1):.2f}"
            )
        else:
            lines.append("**CSV prior:** not in CSV")

        lines.append("")

        # Single-target results
        if pname in single_results:
            lines.append("| Source | Median | CV | Sigma | Contraction | z-score | Dist |")
            lines.append("|--------|--------|----|-------|-------------|---------|------|")

            for entry in single_results[pname]:
                short_tid = entry["target_id"]
                # Truncate long target IDs
                if len(short_tid) > 35:
                    short_tid = short_tid[:32] + "..."
                lines.append(
                    f"| {short_tid} | {entry['median']:.4g} | "
                    f"{entry['cv']:.2f} | {entry['sigma']:.3f} | "
                    f"{entry['contraction']:+.2f} | {entry['z_score']:.2f} | "
                    f"{entry['dist']} |"
                )

            # Joint row
            if pname in joint_fits:
                jf = joint_fits[pname]
                lines.append(
                    f"| **JOINT** | **{jf['median']:.4g}** | "
                    f"**{jf['cv']:.2f}** | **{jf['sigma']:.3f}** | "
                    f"**{jf['contraction']:+.2f}** | **{jf['z_score']:.2f}** | "
                    f"**{jf['dist']}** |"
                )
        else:
            lines.append("No single-target results (parameter only in joint model).")
            if pname in joint_fits:
                jf = joint_fits[pname]
                lines.append(
                    f"**Joint:** median={jf['median']:.4g}, CV={jf['cv']:.2f}, "
                    f"contraction={jf['contraction']:+.2f}"
                )

        lines.append("")

    # ── Summary table ──
    lines.extend(
        [
            "## Summary",
            "",
            "| Parameter | CSV Median | Joint Median | Shift | Joint Contraction |",
            "|-----------|------------|--------------|-------|-------------------|",
        ]
    )

    for pname in sorted(all_param_names):
        csv_med_str = "—"
        shift_str = "—"
        joint_str = "—"
        contr_str = "—"

        if pname in csv_priors:
            csv_median = np.exp(csv_priors[pname].mu)
            csv_med_str = f"{csv_median:.4g}"

        if pname in joint_fits:
            jf = joint_fits[pname]
            joint_str = f"{jf['median']:.4g}"
            contr_str = f"{jf['contraction']:+.2f}"

            if pname in csv_priors:
                ratio = jf["median"] / csv_median
                if ratio > 2:
                    shift_str = f"{ratio:.1f}x up"
                elif ratio < 0.5:
                    shift_str = f"{1/ratio:.1f}x down"
                else:
                    shift_str = f"{ratio:.2f}x"

        lines.append(f"| `{pname}` | {csv_med_str} | {joint_str} | {shift_str} | {contr_str} |")

    # ── Consistency check ──
    lines.extend(["", "## Consistency Check", ""])
    lines.append("Parameters where single-target estimates disagree by >3x:")
    lines.append("")

    any_disagreement = False
    for pname in sorted(single_results.keys()):
        entries = single_results[pname]
        if len(entries) < 2:
            continue
        medians = [e["median"] for e in entries]
        ratio = max(medians) / min(medians) if min(medians) > 0 else float("inf")
        if ratio > 3:
            any_disagreement = True
            vals = ", ".join(f"{m:.4g}" for m in medians)
            lines.append(f"- **`{pname}`**: {ratio:.1f}x spread ({vals})")
            for e in entries:
                lines.append(f"  - {e['target_id']}: {e['median']:.4g}")

    if not any_disagreement:
        lines.append("None — all multi-target parameters are consistent (<3x spread).")

    # ── Joint MCMC diagnostics ──
    lines.extend(["", "## Joint MCMC Diagnostics", ""])
    lines.append(f"- Divergences: {joint_diag['num_divergences']}")
    for pname in sorted(all_param_names):
        pd = joint_diag["per_param"].get(pname, {})
        if pd:
            neff = pd.get("n_eff", 0)
            rhat = pd.get("r_hat", 0)
            flag = " ⚠" if neff < 100 or rhat > 1.05 else ""
            lines.append(f"- `{pname}`: n_eff={neff:.0f}, r_hat={rhat:.3f}{flag}")

    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare single-target vs joint submodel inference"
    )
    parser.add_argument("--priors-csv", required=True, help="Path to priors CSV")
    parser.add_argument("--submodel-dir", required=True, help="Directory with SubmodelTarget YAMLs")
    parser.add_argument("--glob-pattern", default="*_PDAC_deriv*.yaml")
    parser.add_argument("--num-samples", type=int, default=4000)
    parser.add_argument("--output", help="Optional output file (markdown)")
    parser.add_argument(
        "--parameter-groups",
        help="Path to parameter_groups.yaml (auto-discovered in submodel-dir if not set)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = run_comparison(
        priors_csv=args.priors_csv,
        submodel_dir=args.submodel_dir,
        glob_pattern=args.glob_pattern,
        num_samples=args.num_samples,
        parameter_groups_path=args.parameter_groups,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
