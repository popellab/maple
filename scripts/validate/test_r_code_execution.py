#!/usr/bin/env python3
"""
Test R code execution from parameter YAML files.

Validates:
- R code blocks execute without errors
- Required variables are created (mc_draws, mean, variance, ci95)
- Optional: reproducibility across multiple runs

Usage:
    python scripts/validate/test_r_code_execution.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/r_execution_report.json \\
        --test-reproducibility
"""
import argparse
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import json
import re

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    ValidationReport
)


class RCodeValidator:
    """
    Validate R code execution from YAML files.
    """

    def __init__(self, data_dir: str, test_reproducibility: bool = False, threshold_pct: float = 5.0):
        self.data_dir = data_dir
        self.test_reproducibility = test_reproducibility
        self.threshold_pct = threshold_pct

    def extract_r_code(self, data: dict) -> str:
        """
        Extract R code from YAML data.
        Handles both derivation_code_r and embedded code in parameter_estimates.
        """
        # Try parameter_estimates.derivation_code_r first (v2 schema)
        if 'parameter_estimates' in data:
            estimates = data['parameter_estimates']
            if isinstance(estimates, dict) and 'derivation_code_r' in estimates:
                code = estimates['derivation_code_r']
                return self._clean_code_block(code)

        # Try top-level derivation_code_r
        if 'derivation_code_r' in data:
            code = data['derivation_code_r']
            return self._clean_code_block(code)

        return None

    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code fences if present."""
        if not code:
            return None

        # Remove ```r and ``` markers
        code = re.sub(r'^```r?\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)

        return code.strip()

    def execute_r_code(self, code: str, expected_mean: float = None, expected_ci95: list = None) -> tuple:
        """
        Execute R code and check for errors.
        Compare computed values to expected values from YAML.

        Args:
            code: R code to execute
            expected_mean: Expected mean from parameter_estimates.mean
            expected_ci95: Expected CI95 from parameter_estimates.ci95

        Returns:
            (success, message, results_dict) tuple
        """
        if not code:
            return (False, "No R code found", {})

        # Create temporary R script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            temp_script = f.name

            # Write code with variable existence checks and value extraction
            f.write(code)
            f.write("\n\n# Extract computed values\n")
            f.write("cat('VALUES_START\\n')\n")
            f.write("cat('mean_param:', mean_param, '\\n')\n")
            f.write("cat('variance_param:', variance_param, '\\n')\n")
            f.write("cat('ci95_lower:', ci95_param[1], '\\n')\n")
            f.write("cat('ci95_upper:', ci95_param[2], '\\n')\n")
            f.write("cat('VALUES_END\\n')\n")

        try:
            # Execute R script
            result = subprocess.run(
                ['Rscript', '--vanilla', temp_script],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )

            # Clean up temp file
            os.unlink(temp_script)

            # Check for errors
            if result.returncode != 0:
                error_msg = result.stderr[:500]  # First 500 chars of error
                return (False, f"R execution failed: {error_msg}", {})

            # Parse computed values from output
            output = result.stdout
            computed_values = self._parse_computed_values(output)

            if not computed_values:
                return (False, "Could not extract computed values from R output", {})

            # Compare to expected values if provided (use instance threshold)
            comparison_results = {}
            if expected_mean is not None and 'mean_param' in computed_values:
                computed_mean = computed_values['mean_param']
                diff_pct = abs(computed_mean - expected_mean) / abs(expected_mean) * 100 if expected_mean != 0 else float('inf')
                comparison_results['mean_match'] = diff_pct < self.threshold_pct
                comparison_results['mean_diff_pct'] = diff_pct

            if expected_ci95 is not None and 'ci95_lower' in computed_values and 'ci95_upper' in computed_values:
                computed_ci95 = [computed_values['ci95_lower'], computed_values['ci95_upper']]
                lower_diff = abs(computed_ci95[0] - expected_ci95[0]) / abs(expected_ci95[0]) * 100 if expected_ci95[0] != 0 else float('inf')
                upper_diff = abs(computed_ci95[1] - expected_ci95[1]) / abs(expected_ci95[1]) * 100 if expected_ci95[1] != 0 else float('inf')
                comparison_results['ci95_match'] = (lower_diff < self.threshold_pct) and (upper_diff < self.threshold_pct)
                comparison_results['ci95_diff_pct'] = max(lower_diff, upper_diff)

            results = {
                'computed_values': computed_values,
                'comparison': comparison_results
            }

            # Build message and determine overall success
            # Success requires BOTH code execution AND value matching
            overall_success = True
            if comparison_results:
                mean_match = comparison_results.get('mean_match', True)
                ci95_match = comparison_results.get('ci95_match', True)
                mean_diff = comparison_results.get('mean_diff_pct', 0)
                ci95_diff = comparison_results.get('ci95_diff_pct', 0)

                # Pass only if both match (within threshold)
                overall_success = mean_match and ci95_match

                if overall_success:
                    msg = f"Values match YAML (mean: {mean_diff:.2f}%, CI95: {ci95_diff:.2f}%)"
                else:
                    issues = []
                    if not mean_match:
                        issues.append(f"mean: {mean_diff:.2f}%")
                    if not ci95_match:
                        issues.append(f"CI95: {ci95_diff:.2f}%")
                    msg = f"Values differ from YAML ({', '.join(issues)})"
            else:
                msg = "R code executed (no expected values to compare)"

            return (overall_success, msg, results)

        except subprocess.TimeoutExpired:
            os.unlink(temp_script)
            return (False, "R code execution timeout (>30s)", {})
        except Exception as e:
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return (False, f"Execution error: {str(e)}", {})

    def _parse_computed_values(self, output: str) -> dict:
        """Parse computed values from R output."""
        values = {}

        if 'VALUES_START' not in output:
            return values

        lines = output.split('\n')
        in_values_section = False

        for line in lines:
            if 'VALUES_START' in line:
                in_values_section = True
                continue
            if 'VALUES_END' in line:
                break

            if in_values_section and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        values[key] = value
                    except ValueError:
                        continue

        return values

    def test_reproducibility(self, code: str, n_runs: int = 3) -> tuple:
        """
        Test if R code produces consistent results across runs.

        Returns:
            (is_reproducible, message, cv) tuple
        """
        # Not implemented in this initial version
        # Would require extracting final values and comparing across runs
        return (None, "Reproducibility testing not yet implemented", None)

    def validate_directory(self) -> ValidationReport:
        """Validate R code in all YAML files."""
        report = ValidationReport("R Code Execution")

        print(f"Testing R code execution in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']
            data = file_info['data']

            # Skip legacy files
            if '_legacy' in filename:
                report.add_warning(filename, "Skipped legacy file")
                continue

            # Extract R code
            r_code = self.extract_r_code(data)

            if not r_code:
                report.add_warning(filename, "No R code found")
                continue

            # Extract expected values from YAML
            expected_mean = None
            expected_ci95 = None
            if 'parameter_estimates' in data:
                estimates = data['parameter_estimates']
                if 'mean' in estimates:
                    expected_mean = estimates['mean']
                if 'ci95' in estimates:
                    expected_ci95 = estimates['ci95']

            # Execute R code with comparison
            success, message, results = self.execute_r_code(r_code, expected_mean, expected_ci95)

            if success:
                report.add_pass(filename, message)
            else:
                report.add_fail(filename, message)

        # Print detailed results
        print(f"\n{'='*60}")
        print("DETAILED RESULTS")
        print(f"{'='*60}")

        if report.passed:
            print(f"\n✓ PASSED ({len(report.passed)}):")
            for item in report.passed:
                print(f"  - {item['item']}")
                if item['details']:
                    print(f"    {item['details']}")

        if report.failed:
            print(f"\n✗ FAILED ({len(report.failed)}):")
            for item in report.failed:
                print(f"  - {item['item']}")
                print(f"    {item['reason']}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Test R code execution from YAML files")
    parser.add_argument("data_dir", help="Directory with YAML parameter files")
    parser.add_argument("output", help="Output JSON file for validation report")
    parser.add_argument(
        "--test-reproducibility",
        action="store_true",
        help="Test reproducibility across multiple runs (not yet implemented)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Percentage threshold for pass/fail (default: 5.0%%)"
    )

    args = parser.parse_args()

    print(f"Using {args.threshold}% threshold for value matching")

    # Run validation
    validator = RCodeValidator(args.data_dir, args.test_reproducibility, args.threshold)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nR execution report saved to {args.output}")


if __name__ == "__main__":
    main()
