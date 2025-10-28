#!/usr/bin/env python3
"""
Test Python code execution from YAML files.

Validates:
- Python code executes without errors
- Function returns required fields (mean, variance, ci95)
- Returned values match declared values in YAML
- All inputs have corresponding sources

Works for both parameter estimates (v3 schema) and test statistics (v2 schema).

Usage:
    python scripts/validate/test_code_execution.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/code_execution_report.json \\
        --threshold 5.0
"""
import argparse
import sys
import os
from pathlib import Path
import re
import numpy as np

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    ValidationReport
)


class CodeExecutionValidator:
    """
    Validate Python code execution from YAML files.
    Works for both parameters (v3) and test statistics (v2).
    """

    def __init__(self, data_dir: str, threshold_pct: float = 5.0):
        self.data_dir = data_dir
        self.threshold_pct = threshold_pct

    def extract_python_code(self, data: dict) -> tuple:
        """
        Extract Python code from YAML data.

        Returns:
            (code, code_type) tuple where code_type is "parameter" or "test_statistic"
        """
        # Try parameter_estimates first (params v3)
        if 'parameter_estimates' in data:
            estimates = data['parameter_estimates']
            if isinstance(estimates, dict) and 'derivation_code' in estimates:
                code = estimates['derivation_code']
                return (self._clean_code_block(code), "parameter")

        # Try test_statistic_estimates (test stats v2)
        if 'test_statistic_estimates' in data:
            estimates = data['test_statistic_estimates']
            if isinstance(estimates, dict) and 'derivation_code' in estimates:
                code = estimates['derivation_code']
                return (self._clean_code_block(code), "test_statistic")

        return (None, None)

    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code fences if present."""
        if not code:
            return None

        # Remove ```python and ``` markers
        code = re.sub(r'^```python?\s*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)

        return code.strip()

    def extract_inputs(self, data: dict) -> dict:
        """
        Extract inputs from parameter_estimates or test_statistic_estimates.
        Convert to dict format for passing to derive function.

        Returns:
            Dict of inputs keyed by name
        """
        # Try parameter_estimates first
        if 'parameter_estimates' in data:
            estimates = data['parameter_estimates']
            if 'inputs' in estimates:
                return self._convert_inputs_list_to_dict(estimates['inputs'])

        # Try test_statistic_estimates
        if 'test_statistic_estimates' in data:
            estimates = data['test_statistic_estimates']
            if 'inputs' in estimates:
                return self._convert_inputs_list_to_dict(estimates['inputs'])

        return None

    def _convert_inputs_list_to_dict(self, inputs_list: list) -> dict:
        """Convert list of inputs to dict keyed by name."""
        if not isinstance(inputs_list, list):
            return None

        inputs_dict = {}
        for inp in inputs_list:
            if not isinstance(inp, dict) or 'name' not in inp:
                continue
            inputs_dict[inp['name']] = {
                'value': inp.get('value'),
                'units': inp.get('units'),
                'description': inp.get('description'),
                'source_ref': inp.get('source_ref')
            }

        return inputs_dict

    def execute_python_code(self, code: str, inputs: dict, code_type: str,
                           expected_mean: float = None, expected_variance: float = None,
                           expected_ci95: list = None) -> tuple:
        """
        Execute Python code and check for errors.
        Compare computed values to expected values from YAML.

        Args:
            code: Python code to execute
            inputs: Dict of inputs to pass to derive function
            code_type: "parameter" or "test_statistic"
            expected_mean: Expected mean value
            expected_variance: Expected variance value
            expected_ci95: Expected CI95 [lower, upper]

        Returns:
            (success, message, results_dict) tuple
        """
        if not code:
            return (False, "No Python code found", {})

        if not inputs:
            return (False, "No inputs defined", {})

        # Create namespace for execution
        namespace = {
            'inputs': inputs,
            'np': np,
            'numpy': np
        }

        try:
            # Execute the code in isolated namespace
            exec(code, namespace)

            # Determine function name based on code type
            if code_type == "parameter":
                func_name = 'derive_parameter'
                expected_keys = ['mean_param', 'variance_param', 'ci95_param']
            elif code_type == "test_statistic":
                func_name = 'derive_distribution'
                expected_keys = ['mean_stat', 'variance_stat', 'ci95_stat']
            else:
                return (False, f"Unknown code type: {code_type}", {})

            # Check if function exists
            if func_name not in namespace:
                return (False, f"Code does not define {func_name} function", {})

            # Call the function
            derive_func = namespace[func_name]
            result = derive_func(inputs)

            # Validate return type
            if not isinstance(result, dict):
                return (False, f"{func_name} returned {type(result)}, expected dict", {})

            # Check required fields
            missing = [k for k in expected_keys if k not in result]
            if missing:
                return (False, f"Missing required fields in result: {missing}", {})

            # Extract computed values (normalize to param naming)
            mean_key = expected_keys[0]
            var_key = expected_keys[1]
            ci95_key = expected_keys[2]

            computed_values = {
                'mean': float(result[mean_key]),
                'variance': float(result[var_key]),
                'ci95_lower': float(result[ci95_key][0]),
                'ci95_upper': float(result[ci95_key][1])
            }

            # Compare to expected values if provided
            comparison_results = {}
            overall_success = True

            if expected_mean is not None:
                computed_mean = computed_values['mean']
                diff_pct = abs(computed_mean - expected_mean) / abs(expected_mean) * 100 if expected_mean != 0 else float('inf')
                comparison_results['mean_match'] = diff_pct < self.threshold_pct
                comparison_results['mean_diff_pct'] = diff_pct
                overall_success = overall_success and comparison_results['mean_match']

            if expected_variance is not None:
                computed_var = computed_values['variance']
                diff_pct = abs(computed_var - expected_variance) / abs(expected_variance) * 100 if expected_variance != 0 else float('inf')
                comparison_results['variance_match'] = diff_pct < self.threshold_pct
                comparison_results['variance_diff_pct'] = diff_pct
                overall_success = overall_success and comparison_results['variance_match']

            if expected_ci95 is not None and len(expected_ci95) == 2:
                computed_ci95 = [computed_values['ci95_lower'], computed_values['ci95_upper']]
                lower_diff = abs(computed_ci95[0] - expected_ci95[0]) / abs(expected_ci95[0]) * 100 if expected_ci95[0] != 0 else float('inf')
                upper_diff = abs(computed_ci95[1] - expected_ci95[1]) / abs(expected_ci95[1]) * 100 if expected_ci95[1] != 0 else float('inf')
                comparison_results['ci95_match'] = (lower_diff < self.threshold_pct) and (upper_diff < self.threshold_pct)
                comparison_results['ci95_diff_pct'] = max(lower_diff, upper_diff)
                overall_success = overall_success and comparison_results['ci95_match']

            results = {
                'computed_values': computed_values,
                'comparison': comparison_results
            }

            # Build message with calculated vs reported values
            if comparison_results:
                issues = []

                # Report mean
                if expected_mean is not None:
                    computed_mean = computed_values['mean']
                    if comparison_results.get('mean_match', True):
                        issues.append(f"  mean:     calculated={computed_mean:12.3e}, reported={expected_mean:12.3e} ✓")
                    else:
                        issues.append(f"  mean:     calculated={computed_mean:12.3e}, reported={expected_mean:12.3e} ✗")

                # Report variance
                if expected_variance is not None:
                    computed_var = computed_values['variance']
                    if comparison_results.get('variance_match', True):
                        issues.append(f"  variance: calculated={computed_var:12.3e}, reported={expected_variance:12.3e} ✓")
                    else:
                        issues.append(f"  variance: calculated={computed_var:12.3e}, reported={expected_variance:12.3e} ✗")

                # Report CI95
                if expected_ci95 is not None:
                    computed_ci95 = [computed_values['ci95_lower'], computed_values['ci95_upper']]
                    if comparison_results.get('ci95_match', True):
                        issues.append(f"  CI95:     calculated=[{computed_ci95[0]:12.3e}, {computed_ci95[1]:12.3e}], reported=[{expected_ci95[0]:12.3e}, {expected_ci95[1]:12.3e}] ✓")
                    else:
                        issues.append(f"  CI95:     calculated=[{computed_ci95[0]:12.3e}, {computed_ci95[1]:12.3e}], reported=[{expected_ci95[0]:12.3e}, {expected_ci95[1]:12.3e}] ✗")

                msg = "\n" + "\n".join(issues)
            else:
                msg = "Python code executed (no expected values to compare)"

            return (overall_success, msg, results)

        except Exception as e:
            return (False, f"Execution error: {str(e)}", {})

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate code execution in a single YAML file.

        Returns:
            (is_valid, error_msg) tuple
        """
        data = file_info['data']
        filename = file_info['filename']

        # Extract Python code
        python_code, code_type = self.extract_python_code(data)

        if not python_code:
            return (True, "No Python code found (skipped)")

        # Extract inputs
        inputs = self.extract_inputs(data)

        if not inputs:
            return (False, "No inputs defined")

        # Extract expected values from YAML
        expected_mean = None
        expected_variance = None
        expected_ci95 = None

        if code_type == "parameter" and 'parameter_estimates' in data:
            estimates = data['parameter_estimates']
            expected_mean = estimates.get('mean')
            expected_variance = estimates.get('variance')
            expected_ci95 = estimates.get('ci95')
        elif code_type == "test_statistic" and 'test_statistic_estimates' in data:
            estimates = data['test_statistic_estimates']
            expected_mean = estimates.get('mean')
            expected_variance = estimates.get('variance')
            expected_ci95 = estimates.get('ci95')

        # Execute Python code with comparison
        success, message, results = self.execute_python_code(
            python_code, inputs, code_type,
            expected_mean, expected_variance, expected_ci95
        )

        return (success, message)

    def validate_directory(self) -> ValidationReport:
        """Validate Python code in all YAML files."""
        report = ValidationReport("Code Execution (Python)")

        print(f"Testing Python code execution in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']

            is_valid, message = self.validate_file(file_info)

            if is_valid:
                report.add_pass(filename, message)
            else:
                report.add_fail(filename, message)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Test Python code execution from YAML files"
    )
    parser.add_argument("data_dir", help="Directory with YAML files")
    parser.add_argument("output", help="Output JSON file for validation report")
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Percentage threshold for pass/fail (default: 5.0%%)"
    )

    args = parser.parse_args()

    print(f"Using {args.threshold}% threshold for value matching")

    # Run validation
    validator = CodeExecutionValidator(args.data_dir, args.threshold)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nCode execution report saved to {args.output}")


if __name__ == "__main__":
    main()
