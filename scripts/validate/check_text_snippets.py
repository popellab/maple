#!/usr/bin/env python3
"""
Validate that text snippets contain their declared values.

Checks:
- value_snippet contains the declared value
- units_snippet contains the declared value

Handles multiple numeric formats:
- Scientific notation: 1.5e-6 vs "1.5×10⁻⁶" or "0.0000015"
- Percentages: 0.28 vs "28%"
- Dimensional values: 100 with units "mg/kg"

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_text_snippets.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/text_snippet_validation.json
"""
import argparse
import sys
import os
from pathlib import Path
import re

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    parse_numeric_value,
    ValidationReport
)


class TextSnippetValidator:
    """
    Validate that text snippets contain their declared values.
    Works for both parameters and test statistics.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def value_to_search_patterns(self, value, units: str = None) -> list:
        """
        Convert a numeric value to multiple search patterns.

        Args:
            value: Numeric value to search for
            units: Optional units string

        Returns:
            List of string patterns to search for
        """
        if value is None:
            return []

        patterns = []
        numeric_val = parse_numeric_value(value)

        if numeric_val is None:
            # If not numeric, just search for the value as-is
            patterns.append(str(value))
            return patterns

        # Standard decimal notation with various precisions
        patterns.append(f"{numeric_val:.0f}")  # Integer
        patterns.append(f"{numeric_val:.1f}")
        patterns.append(f"{numeric_val:.2f}")
        patterns.append(f"{numeric_val:.3f}")
        patterns.append(f"{numeric_val:.4f}")

        # Scientific notation
        if abs(numeric_val) < 0.01 or abs(numeric_val) > 10000:
            patterns.append(f"{numeric_val:.1e}")
            patterns.append(f"{numeric_val:.2e}")
            # Also add version with × instead of e
            sci_notation = f"{numeric_val:.2e}"
            parts = sci_notation.split('e')
            if len(parts) == 2:
                mantissa = parts[0]
                exponent = int(parts[1])
                patterns.append(f"{mantissa}×10^{exponent}")
                patterns.append(f"{mantissa}×10^{{{exponent}}}")  # LaTeX style
                patterns.append(f"{mantissa} × 10^{exponent}")
                patterns.append(f"{mantissa}×10⁻{abs(exponent)}" if exponent < 0 else f"{mantissa}×10{exponent}")

        # Percentage format (if value is between 0 and 1)
        if 0 <= numeric_val <= 1:
            pct = numeric_val * 100
            patterns.append(f"{pct:.0f}%")
            patterns.append(f"{pct:.1f}%")
            patterns.append(f"{pct:.2f}%")

        # With units if provided
        if units:
            patterns_with_units = []
            for pattern in patterns:
                patterns_with_units.append(f"{pattern} {units}")
                patterns_with_units.append(f"{pattern}{units}")  # No space
            patterns.extend(patterns_with_units)

        return patterns

    def check_snippet_contains_value(self, snippet: str, value, units: str = None) -> tuple:
        """
        Check if snippet contains the declared value in any format.

        Args:
            snippet: Text snippet to search in
            value: Declared value
            units: Optional units string

        Returns:
            (found, matched_pattern) tuple
        """
        if not snippet:
            return (False, None)

        patterns = self.value_to_search_patterns(value, units)

        # Search for each pattern (case-insensitive, handle whitespace variations)
        snippet_normalized = re.sub(r'\s+', ' ', snippet.lower())

        for pattern in patterns:
            pattern_normalized = pattern.lower().strip()
            if pattern_normalized in snippet_normalized:
                return (True, pattern)

        return (False, None)

    def extract_inputs_from_yaml(self, data: dict) -> list:
        """
        Extract inputs from either parameter_estimates or test_statistic_estimates.

        Returns:
            List of input dicts
        """
        # Try parameter_estimates first (params)
        if 'parameter_estimates' in data and isinstance(data['parameter_estimates'], dict):
            if 'inputs' in data['parameter_estimates']:
                return data['parameter_estimates']['inputs']

        # Try test_statistic_estimates (test stats)
        if 'test_statistic_estimates' in data and isinstance(data['test_statistic_estimates'], dict):
            if 'inputs' in data['test_statistic_estimates']:
                return data['test_statistic_estimates']['inputs']

        return []

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate text snippets in a single YAML file.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        data = file_info['data']
        filename = file_info['filename']

        # Extract inputs
        inputs = self.extract_inputs_from_yaml(data)

        if not inputs:
            return (True, [])  # No inputs to validate

        # Check each input
        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            name = inp.get('name', 'unnamed')
            value = inp.get('value')
            units = inp.get('units')

            # Check value_snippet
            value_snippet = inp.get('value_snippet')
            if value_snippet:
                found, pattern = self.check_snippet_contains_value(value_snippet, value, units)
                if not found:
                    errors.append(
                        f"Input '{name}': value_snippet does not contain declared value {value} "
                        f"(tried formats: decimal, scientific, percentage)"
                    )

            # Check units_snippet
            units_snippet = inp.get('units_snippet')
            if units_snippet:
                found, pattern = self.check_snippet_contains_value(units_snippet, value, units)
                if not found:
                    errors.append(
                        f"Input '{name}': units_snippet does not contain declared value {value}"
                    )

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def validate_directory(self) -> ValidationReport:
        """Validate text snippets in all YAML files."""
        report = ValidationReport("Text Snippet Validation")

        print(f"Validating text snippets in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']

            is_valid, errors = self.validate_file(file_info)

            if is_valid:
                report.add_pass(filename)
            else:
                error_msg = "; ".join(errors)
                report.add_fail(filename, error_msg)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate that text snippets contain declared values"
    )
    parser.add_argument("data_dir", help="Directory with YAML files to validate")
    parser.add_argument("output", help="Output JSON file for validation report")

    args = parser.parse_args()

    # Run validation
    validator = TextSnippetValidator(args.data_dir)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nText snippet validation report saved to {args.output}")


if __name__ == "__main__":
    main()
