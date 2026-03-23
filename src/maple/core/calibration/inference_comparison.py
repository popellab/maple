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

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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


def run_comparison(
    priors_csv: str | Path,
    submodel_dir: str | Path,
    glob_pattern: str = "*_PDAC_deriv*.yaml",
    num_warmup: int = 500,
    num_samples: int = 2000,
    num_chains: int = 2,
    parameter_groups_path: str | Path | None = None,
) -> str:
    """Run single-target and joint inference, return comparison report.

    Args:
        priors_csv: Path to priors CSV.
        submodel_dir: Directory containing SubmodelTarget YAMLs.
        glob_pattern: Glob for YAML files.
        num_warmup: NUTS warmup per chain.
        num_samples: Post-warmup samples per chain.
        num_chains: Number of MCMC chains.
        parameter_groups_path: Optional path to parameter_groups.yaml for hierarchical pooling.

    Returns:
        Markdown-formatted comparison report.
    """
    import yaml

    from maple.core.calibration.submodel_inference import (
        load_priors_from_csv,
        run_joint_inference,
    )
    from maple.core.calibration.submodel_target import SubmodelTarget
    from maple.core.calibration.yaml_to_prior import (
        fit_distributions,
        process_yaml,
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

    # ── Phase 1: Joint inference (run first to smoke out issues early) ──
    logger.info("Phase 1: Running joint inference")
    targets = []
    for yf in yaml_files:
        try:
            with open(yf) as f:
                data = yaml.safe_load(f)
            target = SubmodelTarget.model_validate(data)
            targets.append(target)
        except Exception as e:
            logger.warning("Failed to load %s for joint: %s", yf.name, e)

    all_param_names = set()
    for t in targets:
        for p in t.calibration.parameters:
            if not p.nuisance:
                all_param_names.add(p.name)
    # Also include grouped params — they may not appear in any target but
    # need CSV priors for resolve_base_prior and hierarchical sampling
    if param_groups:
        all_param_names |= param_groups.all_grouped_params
    joint_prior_specs = {k: v for k, v in csv_priors.items() if k in all_param_names}

    try:
        joint_samples, joint_diag = run_joint_inference(
            joint_prior_specs,
            targets,
            parameter_groups=param_groups,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
    except Exception as e:
        return f"Joint inference failed: {e}"

    # Fit distributions to joint posteriors
    joint_fits: dict[str, dict] = {}
    for pname in sorted(all_param_names):
        if pname not in joint_samples:
            continue
        fits = fit_distributions(joint_samples[pname])
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

        joint_fits[pname] = {
            "median": best.median,
            "cv": best.cv,
            "sigma": post_sigma,
            "dist": best.name,
            "contraction": _contraction(prior_sigma, post_sigma),
            "z_score": _z_score(prior_mu, post_mu, prior_sigma),
        }

    # ── Phase 2: Single-target inference ──
    logger.info("Phase 2: Running single-target inference on %d targets", len(yaml_files))
    # {param_name: [{target_id, median, cv, sigma, contraction, z_score, dist}]}
    single_results: dict[str, list[dict]] = {}
    target_param_map: dict[str, list[str]] = {}  # {target_id: [param_names]}

    for yf in yaml_files:
        try:
            results = process_yaml(yf, priors_csv=priors_csv)
            for r in results:
                if "error" in r:
                    logger.warning("Single-target %s: %s", yf.name, r["error"])
                    continue
                pname = r["name"]
                tid = r["target_id"]

                # Compute log-space sigma for contraction
                if r["best_dist"].name == "lognormal":
                    post_sigma = r["best_dist"].params["sigma"]
                else:
                    post_sigma = np.sqrt(np.log(1 + r["cv_data"] ** 2))

                prior_sigma = csv_priors[pname].sigma if pname in csv_priors else 1.0
                prior_mu = csv_priors[pname].mu if pname in csv_priors else 0.0
                post_mu = np.log(r["median_data"])

                entry = {
                    "target_id": tid,
                    "median": r["median_data"],
                    "cv": r["cv_data"],
                    "sigma": post_sigma,
                    "dist": r["best_dist"].name,
                    "contraction": _contraction(prior_sigma, post_sigma),
                    "z_score": _z_score(prior_mu, post_mu, prior_sigma),
                }

                if pname not in single_results:
                    single_results[pname] = []
                single_results[pname].append(entry)

                if tid not in target_param_map:
                    target_param_map[tid] = []
                target_param_map[tid].append(pname)
        except Exception as e:
            logger.warning("Failed to process %s: %s", yf.name, e)

    # ── Phase 3: Build comparison report ──
    lines = [
        "# Inference Comparison Report",
        "",
        f"**Targets:** {len(targets)}",
        f"**Parameters:** {len(all_param_names)}",
        f"**MCMC:** {num_warmup} warmup, {num_samples} samples, {num_chains} chains",
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
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-chains", type=int, default=2)
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
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
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
