#!/usr/bin/env python3
"""
Check de novo consistency across multiple independent extractions.

For parameters extracted multiple times (same parameter, different derivation_id),
compute consistency metrics:
- Coefficient of variation (CV) across values
- Overlap of confidence intervals
- Mean absolute deviation

Usage:
    python scripts/validate/check_denovo_consistency.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/consistency_report.json
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    parse_numeric_value,
    ValidationReport
)


class ConsistencyAnalyzer:
    """
    Analyze consistency across multiple extractions of the same parameter.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.param_groups = defaultdict(list)

    def group_by_parameter(self):
        """Group files by parameter name."""
        print(f"Loading and grouping parameters from {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']
            data = file_info['data']

            # Skip legacy files
            if '_legacy' in filename:
                continue

            # Get parameter name
            param_name = data.get('parameter_name')
            if not param_name:
                continue

            # Get value and CI
            value = None
            ci95 = None
            if 'parameter_estimates' in data:
                estimates = data['parameter_estimates']
                value = parse_numeric_value(estimates.get('mean'))
                ci95_raw = estimates.get('ci95')
                if ci95_raw and isinstance(ci95_raw, list) and len(ci95_raw) == 2:
                    ci95 = [parse_numeric_value(ci95_raw[0]), parse_numeric_value(ci95_raw[1])]

            if value is None:
                continue

            # Store in param groups
            self.param_groups[param_name].append({
                'filename': filename,
                'value': value,
                'ci95': ci95,
                'filepath': file_info['filepath']
            })

        print(f"  Found {len(self.param_groups)} unique parameters")
        multi_extraction = {k: v for k, v in self.param_groups.items() if len(v) > 1}
        print(f"  {len(multi_extraction)} parameters with multiple extractions")

        return multi_extraction

    def compute_consistency_metrics(self, extractions: list) -> dict:
        """
        Compute consistency metrics for multiple extractions.

        Args:
            extractions: List of dicts with 'value' and 'ci95'

        Returns:
            Dictionary of consistency metrics
        """
        values = [e['value'] for e in extractions]
        n = len(values)

        if n < 2:
            return None

        # Basic statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        cv = (std_val / abs(mean_val)) * 100 if mean_val != 0 else float('inf')

        # Mean absolute deviation
        mad = np.mean([abs(v - mean_val) for v in values])
        mad_percent = (mad / abs(mean_val)) * 100 if mean_val != 0 else float('inf')

        # Range
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        range_percent = (range_val / abs(mean_val)) * 100 if mean_val != 0 else float('inf')

        # CI overlap (if available)
        cis = [e['ci95'] for e in extractions if e['ci95'] is not None]
        ci_overlap_rate = None
        if len(cis) >= 2:
            ci_overlap_rate = self._compute_ci_overlap(cis)

        return {
            'n_extractions': n,
            'mean': mean_val,
            'std': std_val,
            'cv_percent': cv,
            'mad': mad,
            'mad_percent': mad_percent,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'range_percent': range_percent,
            'ci_overlap_rate': ci_overlap_rate,
            'values': values
        }

    def _compute_ci_overlap(self, cis: list) -> float:
        """
        Compute pairwise CI overlap rate.

        Returns:
            Fraction of CI pairs that overlap
        """
        n_pairs = 0
        n_overlapping = 0

        for i in range(len(cis)):
            for j in range(i + 1, len(cis)):
                ci1 = cis[i]
                ci2 = cis[j]

                # Check if CIs overlap
                if ci1[1] >= ci2[0] and ci2[1] >= ci1[0]:
                    n_overlapping += 1

                n_pairs += 1

        return n_overlapping / n_pairs if n_pairs > 0 else None

    def analyze_consistency(self, cv_threshold: float = 50.0) -> ValidationReport:
        """
        Analyze consistency across parameters with multiple extractions.

        Args:
            cv_threshold: CV threshold for "passing" (default 50%)
        """
        report = ValidationReport("De Novo Consistency")

        multi_extraction = self.group_by_parameter()

        if not multi_extraction:
            print("No parameters with multiple extractions found")
            return report

        consistency_results = {}

        for param_name, extractions in sorted(multi_extraction.items()):
            metrics = self.compute_consistency_metrics(extractions)

            if metrics is None:
                continue

            consistency_results[param_name] = metrics

            # Determine if consistent (CV < threshold)
            cv = metrics['cv_percent']
            n = metrics['n_extractions']

            if cv < cv_threshold:
                details = f"CV={cv:.1f}%, n={n}, range={metrics['range_percent']:.1f}%"
                if metrics['ci_overlap_rate'] is not None:
                    details += f", CI_overlap={metrics['ci_overlap_rate']*100:.0f}%"
                report.add_pass(param_name, details)
            else:
                reason = f"CV={cv:.1f}% > {cv_threshold}%, n={n}"
                reason += f", values={[f'{v:.2e}' for v in metrics['values'][:3]]}"
                report.add_fail(param_name, reason)

        # Add summary statistics
        all_cvs = [m['cv_percent'] for m in consistency_results.values() if m['cv_percent'] < float('inf')]
        if all_cvs:
            print(f"\nConsistency Summary Statistics:")
            print(f"  Median CV: {np.median(all_cvs):.1f}%")
            print(f"  Mean CV: {np.mean(all_cvs):.1f}%")
            print(f"  Min CV: {np.min(all_cvs):.1f}%")
            print(f"  Max CV: {np.max(all_cvs):.1f}%")

        return report

    def save_detailed_results(self, output_path: str):
        """Save detailed consistency metrics."""
        multi_extraction = self.group_by_parameter()

        results = {}
        for param_name, extractions in multi_extraction.items():
            metrics = self.compute_consistency_metrics(extractions)
            if metrics:
                # Remove values array from JSON output (too verbose)
                metrics_copy = metrics.copy()
                del metrics_copy['values']
                results[param_name] = {
                    'metrics': metrics_copy,
                    'files': [e['filename'] for e in extractions]
                }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Check consistency across multiple extractions of same parameter"
    )
    parser.add_argument("data_dir", help="Directory with YAML parameter files")
    parser.add_argument("output", help="Output JSON file for consistency report")
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=50.0,
        help="CV threshold for passing consistency check (default: 50%%)"
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = ConsistencyAnalyzer(args.data_dir)
    report = analyzer.analyze_consistency(args.cv_threshold)

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output.replace('.json', '_summary.json'))
    analyzer.save_detailed_results(args.output)
    print(f"\nDetailed consistency results saved to {args.output}")


if __name__ == "__main__":
    main()
