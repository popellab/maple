#!/usr/bin/env python3
"""
Compare LLM-extracted parameters to legacy database (both in v2 format).

Legacy files are identified by '_legacy' in the filename.
Both are stored in the same directory (../qsp-metadata-storage/parameter_estimates).

Metrics:
- Absolute % difference: |LLM_value - Legacy_value| / Legacy_value * 100
- Agreement within uncertainty: % where LLM value falls within legacy CI
- Correlation coefficient across all parameters

Usage:
    python scripts/validate/compare_to_legacy.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/comparison_report.json
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
from scipy import stats
import json

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    extract_parameter_name_from_filename,
    parse_numeric_value,
    ValidationReport
)


class LegacyComparison:
    """
    Compare LLM extractions to legacy database (v2 format).
    """

    def __init__(self, params_dir: str):
        self.params_dir = params_dir
        self.llm_data = {}
        self.legacy_data = {}
        self.comparisons = []

    def load_parameters(self):
        """Load both LLM and legacy parameters from the same directory."""
        print(f"Loading parameters from {self.params_dir}...")
        files = load_yaml_directory(self.params_dir)

        llm_count = 0
        legacy_count = 0

        for file_info in files:
            data = file_info['data']
            filename = file_info['filename']

            # Extract parameter name from YAML or filename
            param_name = data.get('parameter_name')
            if not param_name:
                # Extract param name, removing _legacy suffix if present
                param_name = extract_parameter_name_from_filename(filename)
                if param_name.endswith('_legacy'):
                    param_name = param_name[:-7]  # Remove '_legacy'

            # Extract value from parameter_estimates field (v2 schema)
            value = None
            ci95 = None
            if 'parameter_estimates' in data:
                estimates = data['parameter_estimates']
                value = parse_numeric_value(estimates.get('mean'))
                ci95_raw = estimates.get('ci95')
                if ci95_raw and isinstance(ci95_raw, list) and len(ci95_raw) == 2:
                    ci95 = [parse_numeric_value(ci95_raw[0]), parse_numeric_value(ci95_raw[1])]

            if not param_name or value is None:
                continue

            # Determine if this is a legacy file
            is_legacy = '_legacy' in filename

            # Store in appropriate dict
            target_dict = self.legacy_data if is_legacy else self.llm_data
            if param_name not in target_dict:
                target_dict[param_name] = []

            target_dict[param_name].append({
                'value': value,
                'ci95': ci95,
                'filepath': file_info['filepath'],
                'filename': filename,
                'data': data
            })

            if is_legacy:
                legacy_count += 1
            else:
                llm_count += 1

        print(f"  LLM files: {llm_count} ({len(self.llm_data)} unique parameters)")
        print(f"  Legacy files: {legacy_count} ({len(self.legacy_data)} unique parameters)")

    def compare_parameters(self) -> ValidationReport:
        """Compare LLM vs legacy parameters."""
        report = ValidationReport("Legacy Database Comparison")

        # Find overlapping parameters
        common_params = set(self.llm_data.keys()) & set(self.legacy_data.keys())
        print(f"\nFound {len(common_params)} overlapping parameters")

        if not common_params:
            print("WARNING: No overlapping parameters found!")
            print(f"  LLM parameters: {sorted(self.llm_data.keys())[:10]}")
            print(f"  Legacy parameters: {sorted(self.legacy_data.keys())[:10]}")
            return report

        for param_name in sorted(common_params):
            llm_values = [item['value'] for item in self.llm_data[param_name]]
            legacy_values = [item['value'] for item in self.legacy_data[param_name]]

            # Use mean if multiple values
            llm_mean = np.mean(llm_values)
            legacy_mean = np.mean(legacy_values)

            # Get CI if available (use first entry)
            llm_ci = self.llm_data[param_name][0].get('ci95')
            legacy_ci = self.legacy_data[param_name][0].get('ci95')

            # Calculate metrics
            abs_diff = abs(llm_mean - legacy_mean)
            pct_diff = (abs_diff / abs(legacy_mean)) * 100 if legacy_mean != 0 else float('inf')

            # Check if LLM value falls within legacy CI
            within_ci = None
            if legacy_ci and all(x is not None for x in legacy_ci):
                within_ci = legacy_ci[0] <= llm_mean <= legacy_ci[1]

            comparison = {
                'parameter': param_name,
                'llm_value': llm_mean,
                'legacy_value': legacy_mean,
                'abs_diff': abs_diff,
                'pct_diff': pct_diff,
                'llm_count': len(llm_values),
                'legacy_count': len(legacy_values),
                'llm_ci': llm_ci,
                'legacy_ci': legacy_ci,
                'within_legacy_ci': within_ci
            }

            self.comparisons.append(comparison)

            # Consider "passed" if within 50% (can adjust threshold)
            if pct_diff < 50:
                details = f"Diff: {pct_diff:.1f}%"
                if within_ci is not None:
                    details += f", Within CI: {within_ci}"
                report.add_pass(param_name, details)
            else:
                reason = f"Diff: {pct_diff:.1f}% (LLM={llm_mean:.2e}, Legacy={legacy_mean:.2e})"
                if within_ci is not None:
                    reason += f", Within CI: {within_ci}"
                report.add_fail(param_name, reason)

        return report

    def compute_correlation(self) -> dict:
        """Compute correlation statistics."""
        if not self.comparisons:
            return {}

        llm_vals = [c['llm_value'] for c in self.comparisons]
        legacy_vals = [c['legacy_value'] for c in self.comparisons]

        # Pearson correlation
        r, p = stats.pearsonr(llm_vals, legacy_vals)

        # Spearman correlation (rank-based, more robust to outliers)
        rho, p_spearman = stats.spearmanr(llm_vals, legacy_vals)

        # Calculate agreement within CI
        within_ci_count = sum(1 for c in self.comparisons if c['within_legacy_ci'] is True)
        total_with_ci = sum(1 for c in self.comparisons if c['within_legacy_ci'] is not None)

        return {
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_spearman,
            'n_comparisons': len(self.comparisons),
            'within_ci_count': within_ci_count,
            'total_with_ci': total_with_ci,
            'within_ci_rate': within_ci_count / total_with_ci if total_with_ci > 0 else None
        }

    def save_results(self, output_path: str):
        """Save detailed comparison results."""
        results = {
            'correlations': self.compute_correlation(),
            'comparisons': self.comparisons
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM extractions to legacy database (v2 format, same directory)"
    )
    parser.add_argument(
        "params_dir",
        help="Directory with both LLM and legacy parameters (legacy files have '_legacy' suffix)"
    )
    parser.add_argument("output", help="Output JSON file for detailed results")

    args = parser.parse_args()

    # Run comparison
    comparison = LegacyComparison(args.params_dir)
    comparison.load_parameters()

    report = comparison.compare_parameters()

    # Print summary
    report.print_summary()

    # Print correlation stats
    corr_stats = comparison.compute_correlation()
    if corr_stats:
        print(f"\n{'='*60}")
        print("Correlation Statistics")
        print(f"{'='*60}")
        print(f"Pearson r:  {corr_stats['pearson_r']:.3f} (p={corr_stats['pearson_p']:.3e})")
        print(f"Spearman ρ: {corr_stats['spearman_rho']:.3f} (p={corr_stats['spearman_p']:.3e})")
        print(f"N:          {corr_stats['n_comparisons']}")
        if corr_stats['within_ci_rate'] is not None:
            print(f"\nAgreement within legacy CI: {corr_stats['within_ci_count']}/{corr_stats['total_with_ci']} ({corr_stats['within_ci_rate']*100:.1f}%)")

    # Save results
    comparison.save_results(args.output)
    report.save_to_json(args.output.replace('.json', '_summary.json'))


if __name__ == "__main__":
    main()
