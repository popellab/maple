#!/usr/bin/env python3
"""
Test Python code execution from parameter YAML files (v3 schema).

Validates:
- Python code executes without errors
- Function returns required fields (mean_param, variance_param, ci95_param)
- Returned values match declared values in YAML
- All inputs have corresponding sources

Usage:
    python scripts/validate/test_python_code_execution.py \
        ../qsp-metadata-storage/parameter_estimates \
        output/python_execution_report.json \
        --threshold 5.0
"""
import argparse
import sys
import os
from pathlib import Path
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


class PythonCodeValidator:
    """
    Validate Python code execution from YAML files (v3 schema).
    """

    def __init__(self, data_dir: str, threshold_pct: float = 5.0):
        self.data_dir = data_dir
        self.threshold_pct = threshold_pct

    def extract_python_code(self, data: dict) -> str:
        """
        Extract Python code from YAML data (v3 schema).
        """
        if 'parameter_estimates' not in data:
            return None

        estimates = data['parameter_estimates']
        if 'derivation_code' not in estimates:
            return None

        code = estimates['derivation_code']
        return self._clean_code_block(code)

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
        Extract inputs from parameter_estimates.inputs list.
        Convert to dict format for passing to derive_parameter function.
        """
        if 'parameter_estimates' not in data:
            return None

        estimates = data['parameter_estimates']
        if 'inputs' not in estimates:
            return None

        inputs_list = estimates['inputs']
        if not isinstance(inputs_list, list):
            return None

        # Convert list of inputs to dict keyed by name
        inputs_dict = {}
        for inp in inputs_list:
            if not isinstance(inp, dict) or 'name' not in inp:
                continue
            inputs_dict[inp['name']] = {
                'value': inp.get('value'),
                'units': inp.get('units'),
                'description': inp.get('description'),
                'source_ref': inp.get('source_ref'),
                'assumption': inp.get('assumption')
            }

        return inputs_dict

    def validate_inputs_sources(self, data: dict, inputs: dict) -> tuple:
        """
        Validate that all inputs have corresponding sources.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        if not inputs:
            return (True, errors)

        # Collect all source keys
        all_sources = set()
        for source_type in ['primary_data_sources', 'secondary_data_sources', 'methodological_sources']:
            if source_type in data and isinstance(data[source_type], dict):
                all_sources.update(data[source_type].keys())

        # Check each input
        for name, inp in inputs.items():
            source_ref = inp.get('source_ref')
            assumption = inp.get('assumption')

            # Input must have either a source_ref or an assumption
            if not source_ref and not assumption:
                errors.append(f"Input '{name}' has neither source_ref nor assumption")
                continue

            # If source_ref is provided, it must exist in sources
            if source_ref and source_ref not in all_sources:
                errors.append(f"Input '{name}' references source '{source_ref}' which doesn't exist")

        return (len(errors) == 0, errors)

    def execute_python_code(self, code: str, inputs: dict, expected_mean: float = None,
                           expected_variance: float = None, expected_ci95: list = None) -> tuple:
        """
        Execute Python code and check for errors.
        Compare computed values to expected values from YAML.

        Args:
            code: Python code to execute
            inputs: Dict of inputs to pass to derive_parameter function
            expected_mean: Expected mean from parameter_estimates.mean
            expected_variance: Expected variance from parameter_estimates.variance
            expected_ci95: Expected CI95 from parameter_estimates.ci95

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

            # Check if derive_parameter function exists
            if 'derive_parameter' not in namespace:
                return (False, "Code does not define derive_parameter function", {})

            # Call the function
            derive_parameter = namespace['derive_parameter']
            result = derive_parameter(inputs)

            # Validate return type
            if not isinstance(result, dict):
                return (False, f"derive_parameter returned {type(result)}, expected dict", {})

            # Check required fields
            required_fields = ['mean_param', 'variance_param', 'ci95_param']
            missing = [f for f in required_fields if f not in result]
            if missing:
                return (False, f"Missing required fields in result: {missing}", {})

            # Extract computed values
            computed_values = {
                'mean_param': float(result['mean_param']),
                'variance_param': float(result['variance_param']),
                'ci95_lower': float(result['ci95_param'][0]),
                'ci95_upper': float(result['ci95_param'][1])
            }

            # Compare to expected values if provided
            comparison_results = {}
            overall_success = True

            if expected_mean is not None:
                computed_mean = computed_values['mean_param']
                diff_pct = abs(computed_mean - expected_mean) / abs(expected_mean) * 100 if expected_mean != 0 else float('inf')
                comparison_results['mean_match'] = diff_pct < self.threshold_pct
                comparison_results['mean_diff_pct'] = diff_pct
                overall_success = overall_success and comparison_results['mean_match']

            if expected_variance is not None:
                computed_var = computed_values['variance_param']
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

            # Build message
            if comparison_results:
                mean_diff = comparison_results.get('mean_diff_pct', 0)
                var_diff = comparison_results.get('variance_diff_pct', 0)
                ci95_diff = comparison_results.get('ci95_diff_pct', 0)

                if overall_success:
                    msg = f"Values match YAML (mean: {mean_diff:.2f}%, var: {var_diff:.2f}%, CI95: {ci95_diff:.2f}%)"
                else:
                    issues = []
                    if not comparison_results.get('mean_match', True):
                        issues.append(f"mean: {mean_diff:.2f}%")
                    if not comparison_results.get('variance_match', True):
                        issues.append(f"variance: {var_diff:.2f}%")
                    if not comparison_results.get('ci95_match', True):
                        issues.append(f"CI95: {ci95_diff:.2f}%")
                    msg = f"Values differ from YAML ({', '.join(issues)})"
            else:
                msg = "Python code executed (no expected values to compare)"

            return (overall_success, msg, results)

        except Exception as e:
            return (False, f"Execution error: {str(e)}", {})

    def validate_directory(self) -> ValidationReport:
        """Validate Python code in all YAML files."""
        report = ValidationReport("Python Code Execution (v3)")

        print(f"Testing Python code execution in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']
            data = file_info['data']

            # Skip if not v3 schema
            schema_version = data.get('schema_version', '')
            if schema_version != 'v3':
                report.add_warning(filename, f"Not v3 schema (version: {schema_version})")
                continue

            # Extract Python code
            python_code = self.extract_python_code(data)

            if not python_code:
                report.add_warning(filename, "No Python code found")
                continue

            # Extract inputs
            inputs = self.extract_inputs(data)

            if not inputs:
                report.add_fail(filename, "No inputs defined")
                continue

            # Validate inputs have sources
            sources_valid, source_errors = self.validate_inputs_sources(data, inputs)
            if not sources_valid:
                report.add_fail(filename, f"Input-source validation failed: {'; '.join(source_errors)}")
                continue

            # Extract expected values from YAML
            expected_mean = None
            expected_variance = None
            expected_ci95 = None
            if 'parameter_estimates' in data:
                estimates = data['parameter_estimates']
                expected_mean = estimates.get('mean')
                expected_variance = estimates.get('variance')
                expected_ci95 = estimates.get('ci95')

            # Execute Python code with comparison
            success, message, results = self.execute_python_code(
                python_code, inputs, expected_mean, expected_variance, expected_ci95
            )

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

        if report.warnings:
            print(f"\n⚠ WARNINGS ({len(report.warnings)}):")
            for item in report.warnings:
                print(f"  - {item['item']}")
                print(f"    {item['reason']}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Test Python code execution from YAML files (v3 schema)")
    parser.add_argument("data_dir", help="Directory with YAML parameter files")
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
    validator = PythonCodeValidator(args.data_dir, args.threshold)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nPython execution report saved to {args.output}")


if __name__ == "__main__":
    main()
