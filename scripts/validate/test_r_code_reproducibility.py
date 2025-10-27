#!/usr/bin/env python3
"""
Test R code reproducibility across multiple runs with different seeds.

Since R bootstrap code typically sets a seed (e.g., set.seed(123)), we need to
remove or randomize it to test true reproducibility of the statistical method.

Validates:
- Code produces consistent distributions across different random seeds
- Coefficient of variation (CV) of mean estimates is low
- Bootstrap sampling is working correctly

Usage:
    python scripts/validate/test_r_code_reproducibility.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/reproducibility_report.json \\
        --n-runs 5 \\
        --cv-threshold 5.0
"""
import argparse
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import json
import re
import numpy as np

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    ValidationReport
)


class ReproducibilityTester:
    """
    Test reproducibility of R code across multiple runs with different seeds.
    """

    def __init__(self, data_dir: str, n_runs: int = 5, cv_threshold: float = 5.0):
        self.data_dir = data_dir
        self.n_runs = n_runs
        self.cv_threshold = cv_threshold

    def extract_r_code(self, data: dict) -> str:
        """Extract R code from YAML data."""
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

    def remove_set_seed(self, code: str) -> str:
        """
        Remove or comment out set.seed() calls to allow different random seeds.
        """
        # Comment out set.seed lines
        code = re.sub(r'^(\s*)set\.seed\([^)]*\)', r'\1# set.seed() removed for reproducibility testing', code, flags=re.MULTILINE)
        return code

    def execute_r_code_with_seed(self, code: str, seed: int) -> dict:
        """
        Execute R code with a specific random seed and extract results.

        Args:
            code: R code to execute (with set.seed removed)
            seed: Random seed to use

        Returns:
            Dictionary with extracted statistics, or None on failure
        """
        # Create temporary R script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
            temp_script = f.name

            # Set seed at the start
            f.write(f"set.seed({seed})\n\n")

            # Write the modified code
            f.write(code)

            # Extract final statistics
            f.write("\n\n# Extract final statistics for reproducibility testing\n")
            f.write("cat('RESULTS_START\\n')\n")

            # Try different variable names for mean
            f.write("if (exists('mean_param')) {\n")
            f.write("  cat('mean:', mean_param, '\\n')\n")
            f.write("} else if (exists('mu')) {\n")
            f.write("  cat('mean:', mu, '\\n')\n")
            f.write("} else if (exists('mean_stat')) {\n")
            f.write("  cat('mean:', mean_stat, '\\n')\n")
            f.write("}\n")

            # Try different variable names for variance
            f.write("if (exists('variance_param')) {\n")
            f.write("  cat('variance:', variance_param, '\\n')\n")
            f.write("} else if (exists('s2')) {\n")
            f.write("  cat('variance:', s2, '\\n')\n")
            f.write("} else if (exists('variance_stat')) {\n")
            f.write("  cat('variance:', variance_stat, '\\n')\n")
            f.write("}\n")

            # Extract CI if available
            f.write("if (exists('ci95_param')) {\n")
            f.write("  cat('ci95_lower:', ci95_param[1], '\\n')\n")
            f.write("  cat('ci95_upper:', ci95_param[2], '\\n')\n")
            f.write("} else if (exists('natural_scale_ci95')) {\n")
            f.write("  cat('ci95_lower:', natural_scale_ci95[1], '\\n')\n")
            f.write("  cat('ci95_upper:', natural_scale_ci95[2], '\\n')\n")
            f.write("} else if (exists('ci95_stat')) {\n")
            f.write("  cat('ci95_lower:', ci95_stat[1], '\\n')\n")
            f.write("  cat('ci95_upper:', ci95_stat[2], '\\n')\n")
            f.write("}\n")

            # Extract sample size
            f.write("if (exists('mc_draws')) {\n")
            f.write("  cat('n_samples:', length(mc_draws), '\\n')\n")
            f.write("}\n")

            f.write("cat('RESULTS_END\\n')\n")

        try:
            # Execute R script
            result = subprocess.run(
                ['Rscript', '--vanilla', temp_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up temp file
            os.unlink(temp_script)

            if result.returncode != 0:
                return None

            # Parse results
            return self._parse_results(result.stdout)

        except subprocess.TimeoutExpired:
            os.unlink(temp_script)
            return None
        except Exception as e:
            if os.path.exists(temp_script):
                os.unlink(temp_script)
            return None

    def _parse_results(self, output: str) -> dict:
        """Parse extracted statistics from R output."""
        if 'RESULTS_START' not in output:
            return None

        lines = output.split('\n')
        in_results = False
        parsed = {}

        for line in lines:
            if 'RESULTS_START' in line:
                in_results = True
                continue
            if 'RESULTS_END' in line:
                break

            if in_results and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        value = float(parts[1].strip())
                        parsed[key] = value
                    except ValueError:
                        continue

        return parsed if parsed else None

    def test_reproducibility(self, code: str) -> tuple:
        """
        Test reproducibility by running code multiple times with different seeds.

        Returns:
            (is_reproducible, message, metrics) tuple
        """
        if not code:
            return (False, "No R code found", None)

        # Remove set.seed calls
        code_no_seed = self.remove_set_seed(code)

        # Run with different seeds
        results = []
        seeds = [100 + i * 11 for i in range(self.n_runs)]  # 100, 111, 122, 133, 144

        for seed in seeds:
            result = self.execute_r_code_with_seed(code_no_seed, seed)
            if result and 'mean' in result:
                results.append(result)

        if len(results) == 0:
            return (False, "No successful runs", None)

        if len(results) < self.n_runs:
            return (False, f"Only {len(results)}/{self.n_runs} runs completed", None)

        # Analyze reproducibility
        return self._analyze_reproducibility(results)

    def _analyze_reproducibility(self, results: list) -> tuple:
        """
        Analyze reproducibility across multiple runs.

        Returns:
            (is_reproducible, message, metrics) tuple
        """
        # Extract values across runs
        means = [r['mean'] for r in results]
        variances = [r.get('variance') for r in results if 'variance' in r]
        ci_widths = [(r.get('ci95_upper', 0) - r.get('ci95_lower', 0)) for r in results
                     if 'ci95_upper' in r and 'ci95_lower' in r]
        n_samples = [r.get('n_samples') for r in results if 'n_samples' in r]

        # Calculate CV of means across runs
        mean_of_means = np.mean(means)
        std_of_means = np.std(means, ddof=1) if len(means) > 1 else 0.0
        cv_means = (std_of_means / abs(mean_of_means)) * 100 if mean_of_means != 0 else float('inf')

        # Calculate CV of variances if available
        cv_variances = None
        if len(variances) > 1:
            mean_of_vars = np.mean(variances)
            std_of_vars = np.std(variances, ddof=1)
            cv_variances = (std_of_vars / abs(mean_of_vars)) * 100 if mean_of_vars != 0 else float('inf')

        # Calculate CV of CI widths if available
        cv_ci_widths = None
        if len(ci_widths) > 1:
            mean_ci_width = np.mean(ci_widths)
            std_ci_width = np.std(ci_widths, ddof=1)
            cv_ci_widths = (std_ci_width / abs(mean_ci_width)) * 100 if mean_ci_width != 0 else float('inf')

        # Check sample size consistency
        n_samples_consistent = len(set(n_samples)) == 1 if n_samples else None

        metrics = {
            'n_runs': len(results),
            'mean_values': means,
            'mean_of_means': mean_of_means,
            'std_of_means': std_of_means,
            'cv_means_percent': cv_means,
            'variance_values': variances if variances else None,
            'cv_variances_percent': cv_variances,
            'ci_widths': ci_widths if ci_widths else None,
            'cv_ci_widths_percent': cv_ci_widths,
            'n_samples': n_samples[0] if n_samples_consistent else None,
            'n_samples_consistent': n_samples_consistent
        }

        # Determine if reproducible
        is_reproducible = cv_means < self.cv_threshold

        if is_reproducible:
            message = f"Reproducible: CV={cv_means:.2f}% < {self.cv_threshold}% across {len(results)} runs"
        else:
            message = f"Variable: CV={cv_means:.2f}% >= {self.cv_threshold}% across {len(results)} runs"

        return (is_reproducible, message, metrics)

    def validate_directory(self) -> ValidationReport:
        """Test reproducibility for all YAML files with R code."""
        report = ValidationReport("R Code Reproducibility")

        print(f"Testing R code reproducibility in {self.data_dir}...")
        print(f"  Runs per file: {self.n_runs}")
        print(f"  CV threshold: {self.cv_threshold}%")

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

            # Test reproducibility
            is_reproducible, message, metrics = self.test_reproducibility(r_code)

            if is_reproducible:
                report.add_pass(filename, message)
            elif metrics is not None:
                report.add_fail(filename, message)
            else:
                report.add_fail(filename, message)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Test R code reproducibility across runs with different seeds"
    )
    parser.add_argument("data_dir", help="Directory with YAML parameter files")
    parser.add_argument("output", help="Output JSON file for reproducibility report")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs with different seeds (default: 5)"
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=5.0,
        help="CV threshold for passing reproducibility (default: 5.0%%)"
    )

    args = parser.parse_args()

    # Run validation
    tester = ReproducibilityTester(args.data_dir, args.n_runs, args.cv_threshold)
    report = tester.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nReproducibility report saved to {args.output}")


if __name__ == "__main__":
    main()
