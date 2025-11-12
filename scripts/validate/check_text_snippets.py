#!/usr/bin/env python3
"""
Validate that text snippets contain their declared values.

Checks:
- value_snippet contains the declared value
- units_snippet is present (but units matching is left to manual review)

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

    def text_to_number(self, text: str) -> int:
        """
        Convert text-encoded numbers to integers.

        Args:
            text: Text like "fifty-two", "twenty-three", or "one hundred"

        Returns:
            Integer value, or None if not recognized
        """
        # Dictionary of text numbers
        ones = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19
        }
        tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }

        text = text.lower().strip()

        # Handle "hundred" patterns first
        if 'hundred' in text:
            # Remove "hundred" and split into parts
            parts = text.replace('hundred', '').strip().split()

            result = 0
            i = 0

            # Get the multiplier before "hundred" (e.g., "one" in "one hundred")
            if i < len(parts):
                multiplier = ones.get(parts[i], 1)  # Default to 1 if not specified
                result = multiplier * 100
                i += 1
            else:
                # Just "hundred" by itself
                result = 100

            # Check if there are additional parts after "hundred"
            # e.g., "one hundred fifty-two" or "one hundred fifty two"
            remaining = ' '.join(parts[i:])
            if remaining:
                # Handle hyphenated like "fifty-two"
                if '-' in remaining:
                    sub_parts = remaining.split('-')
                    if len(sub_parts) == 2:
                        ten_val = tens.get(sub_parts[0], 0)
                        one_val = ones.get(sub_parts[1], 0)
                        result += ten_val + one_val
                # Handle space-separated or single number
                elif remaining in ones:
                    result += ones[remaining]
                elif remaining in tens:
                    result += tens[remaining]
                else:
                    # Try multi-word like "fifty two"
                    remaining_parts = remaining.split()
                    if len(remaining_parts) == 2:
                        result += tens.get(remaining_parts[0], 0) + ones.get(remaining_parts[1], 0)
                    elif len(remaining_parts) == 1:
                        result += ones.get(remaining_parts[0], 0) + tens.get(remaining_parts[0], 0)

            return result if result > 0 else None

        # Handle single words
        if text in ones:
            return ones[text]
        if text in tens:
            return tens[text]

        # Handle hyphenated numbers like "fifty-two"
        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                ten_val = tens.get(parts[0], 0)
                one_val = ones.get(parts[1], 0)
                if ten_val > 0 or one_val > 0:
                    return ten_val + one_val

        # Handle space-separated like "fifty two"
        if ' ' in text:
            parts = text.split()
            if len(parts) == 2:
                ten_val = tens.get(parts[0], 0)
                one_val = ones.get(parts[1], 0)
                if ten_val > 0 or one_val > 0:
                    return ten_val + one_val

        return None

    def number_to_text(self, num: float) -> str:
        """
        Convert a number to text representation (limited to common values).

        Args:
            num: Numeric value

        Returns:
            Text representation, or None if not in supported range
        """
        # Only handle integers in reasonable range
        if not isinstance(num, (int, float)) or num != int(num):
            return None

        num = int(num)

        # Dictionary of number words
        ones = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
            5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
            10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen',
            14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen',
            18: 'eighteen', 19: 'nineteen'
        }
        tens = {
            20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
            60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'
        }

        # Handle 0-19
        if 0 <= num <= 19:
            return ones[num]

        # Handle 20-99
        if 20 <= num <= 99:
            tens_val = (num // 10) * 10
            ones_val = num % 10
            if ones_val == 0:
                return tens[tens_val]
            else:
                return f"{tens[tens_val]}-{ones[ones_val]}"

        # Handle 100-999
        if 100 <= num <= 999:
            hundreds_val = num // 100
            remainder = num % 100

            result = f"{ones[hundreds_val]} hundred"

            if remainder > 0:
                # Add the remaining part
                if 0 <= remainder <= 19:
                    result += f" {ones[remainder]}"
                elif 20 <= remainder <= 99:
                    tens_val = (remainder // 10) * 10
                    ones_val = remainder % 10
                    if ones_val == 0:
                        result += f" {tens[tens_val]}"
                    else:
                        result += f" {tens[tens_val]}-{ones[ones_val]}"

            return result

        # Numbers outside supported range
        return None

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

                # Create mantissa variations (with and without trailing zeros)
                mantissa_variations = [mantissa]
                # Strip trailing zeros: "5.00" -> "5.0" -> "5"
                mantissa_stripped = mantissa.rstrip('0').rstrip('.')
                if mantissa_stripped != mantissa:
                    mantissa_variations.append(mantissa_stripped)
                # Also try intermediate: "5.00" -> "5.0"
                if mantissa.endswith('0') and not mantissa.endswith('.0'):
                    mantissa_variations.append(mantissa.rstrip('0'))

                # Generate patterns for each mantissa variation
                for m in mantissa_variations:
                    patterns.append(f"{m}×10^{exponent}")
                    patterns.append(f"{m}×10^{{{exponent}}}")  # LaTeX style with braces
                    patterns.append(f"{m} × 10^{exponent}")
                    patterns.append(f"{m} × 10^{{{exponent}}}")  # LaTeX style with braces and space
                    patterns.append(f"{m}×10⁻{abs(exponent)}" if exponent < 0 else f"{m}×10{exponent}")

        # Percentage format (if value is between 0 and 1)
        if 0 <= numeric_val <= 1:
            pct = numeric_val * 100
            # Add both with and without % sign
            # (snippets often have "30.62 ± 16.80%" where % is separated)
            patterns.append(f"{pct:.0f}%")
            patterns.append(f"{pct:.1f}%")
            patterns.append(f"{pct:.2f}%")
            patterns.append(f"{pct:.0f}")  # Without %
            patterns.append(f"{pct:.1f}")  # Without %
            patterns.append(f"{pct:.2f}")  # Without %

        # Text-encoded numbers (e.g., "one hundred" for 100)
        text_pattern = self.number_to_text(numeric_val)
        if text_pattern:
            patterns.append(text_pattern)
            # Also add capitalized version
            patterns.append(text_pattern.capitalize())
            # Title case (e.g., "One Hundred")
            patterns.append(text_pattern.title())

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
        # Remove commas for numeric matching (e.g., "4,623" becomes "4623")
        snippet_normalized = re.sub(r'\s+', ' ', snippet.lower())
        snippet_normalized = snippet_normalized.replace(',', '')

        # Sort patterns by length (longest first) to match most specific patterns first
        # This ensures "16.07" is tried before "16"
        patterns_sorted = sorted(patterns, key=len, reverse=True)

        for pattern in patterns_sorted:
            pattern_normalized = pattern.lower().strip()
            if pattern_normalized in snippet_normalized:
                return (True, pattern)

        # Also check for text-encoded numbers in the snippet
        numeric_val = parse_numeric_value(value)
        if numeric_val is not None and numeric_val == int(numeric_val):
            # Extract potential text numbers from snippet
            words = re.findall(r'\b[a-z]+(?:-[a-z]+)?\b', snippet_normalized)
            for i, word in enumerate(words):
                text_num = self.text_to_number(word)
                if text_num == int(numeric_val):
                    return (True, f"text:{word}")
                # Try two-word combinations
                if i < len(words) - 1:
                    two_word = f"{word} {words[i+1]}"
                    text_num = self.text_to_number(two_word)
                    if text_num == int(numeric_val):
                        return (True, f"text:{two_word}")

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
            (is_valid, errors, input_results) tuple where input_results is a list of dicts
        """
        errors = []
        input_results = []
        data = file_info['data']

        # Extract inputs
        inputs = self.extract_inputs_from_yaml(data)

        if not inputs:
            return (True, [], [])  # No inputs to validate

        # Check each input
        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            name = inp.get('name', 'unnamed')
            value = inp.get('value')
            units = inp.get('units')

            # Check value_snippet contains the value
            value_snippet = inp.get('value_snippet')
            if value_snippet:
                # Handle list values by checking each element
                if isinstance(value, list):
                    all_found = True
                    matched_patterns = []
                    missing_values = []

                    for val in value:
                        found, pattern = self.check_snippet_contains_value(value_snippet, val, units)
                        if found:
                            matched_patterns.append(f"{val}→{pattern}")
                        else:
                            all_found = False
                            missing_values.append(val)

                    input_result = {
                        'input_name': name,
                        'value': value,
                        'found': all_found,
                        'matched_pattern': '; '.join(matched_patterns) if matched_patterns else None
                    }
                    input_results.append(input_result)

                    if not all_found:
                        errors.append(
                            f"Input '{name}': value_snippet does not contain all declared values. "
                            f"Missing: {missing_values} "
                            f"(tried formats: decimal, scientific, percentage, text-encoded)"
                        )
                else:
                    # Single value (not a list)
                    found, pattern = self.check_snippet_contains_value(value_snippet, value, units)

                    input_result = {
                        'input_name': name,
                        'value': value,
                        'found': found,
                        'matched_pattern': pattern
                    }
                    input_results.append(input_result)

                    if not found:
                        errors.append(
                            f"Input '{name}': value_snippet does not contain declared value {value} "
                            f"(tried formats: decimal, scientific, percentage, text-encoded)"
                        )

            # Note: units_snippet validation is intentionally skipped
            # Units can be expressed in many ways (mg/kg vs milligrams per kilogram)
            # This is left to manual review checklist (see MANUAL_REVIEW_CHECKLIST.md)

        is_valid = len(errors) == 0
        return (is_valid, errors, input_results)

    def validate_directory(self) -> ValidationReport:
        """Validate text snippets in all YAML files."""
        report = ValidationReport("Text Snippet Validation")

        print(f"Validating text snippets in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']

            is_valid, errors, input_results = self.validate_file(file_info)

            # Report on each input validated
            for inp_result in input_results:
                input_name = inp_result['input_name']
                value = inp_result['value']
                found = inp_result['found']
                pattern = inp_result['matched_pattern']

                item_desc = f"{filename} / input '{input_name}' (value={value})"

                if found:
                    report.add_pass(item_desc, f"Found as: {pattern}")
                else:
                    report.add_fail(
                        item_desc,
                        f"value_snippet does not contain value {value} (tried: decimal, scientific, percentage, text-encoded)"
                    )

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

    # Exit with error code if any validations failed
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
