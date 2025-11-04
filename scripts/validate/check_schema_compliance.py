#!/usr/bin/env python3
"""
Validate YAML files against expected schema.

Checks:
- Valid YAML parsing
- Required fields present
- Field types match expectations
- Numeric values in valid ranges

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_schema_compliance.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        templates/parameter_metadata_template.yaml \\
        output/schema_validation.json
"""
import argparse
import sys
import os
from pathlib import Path
import json

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    load_yaml_file,
    parse_numeric_value,
    ValidationReport
)


class SchemaValidator:
    """
    Validate YAML files against template schema.
    Uses template file to determine required fields dynamically.
    """

    def __init__(self, data_dir: str, template_path: str):
        self.data_dir = data_dir
        self.template_path = template_path
        self.template = load_yaml_file(template_path)

        if not self.template:
            raise ValueError(f"Could not load template from {template_path}")

        # Extract required fields from template
        self.required_top_level = self._get_required_fields(self.template)

    def _get_required_fields(self, obj, path=""):
        """
        Extract required fields from template by finding non-placeholder values.
        Placeholder patterns: UPPERCASE, NUMERIC_VALUE, etc.

        Special handling for sources: don't treat example keys (PRIMARY_STUDY, etc.)
        as required - only check that the dict structure exists.
        """
        required = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Skip example placeholder keys in sources sections
                if key in ['PRIMARY_STUDY', 'SECONDARY_STUDY', 'METHOD_REFERENCE']:
                    continue

                # Field is required if it exists in template
                # (We consider all fields in template as required unless they have specific markers)
                if isinstance(value, str):
                    # Skip if it's an optional field marker
                    if "OR_NULL" in value or (isinstance(key, str) and "optional" in key.lower()):
                        continue
                    required.append(current_path)
                elif isinstance(value, (int, float, bool)):
                    required.append(current_path)
                elif isinstance(value, dict):
                    required.append(current_path)
                    # Don't recursively check inside sources - those are examples
                    if not (path.endswith('data_sources') or path.endswith('methodological_sources')):
                        nested = self._get_required_fields(value, current_path)
                        required.extend(nested)
                elif isinstance(value, list) and len(value) > 0:
                    required.append(current_path)

        return required

    def _check_field_exists(self, data, field_path):
        """
        Check if a nested field exists in data.
        field_path is dot-separated like 'parameter_estimates.mean'
        """
        parts = field_path.split('.')
        current = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
            if current is None:
                return False

        return True

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate a single YAML file against template.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        data = file_info['data']
        filename = file_info['filename']

        # Check all required fields from template
        for field_path in self.required_top_level:
            if not self._check_field_exists(data, field_path):
                # Check if this field has OR_NULL in template (making it optional)
                template_value = self._get_field_value(self.template, field_path)
                if template_value and isinstance(template_value, str) and "OR_NULL" in template_value:
                    continue  # Optional field
                errors.append(f"Missing required field: {field_path}")

        # Additional specific validations (schema-agnostic)
        # Validate ci95 is a 2-element list (works for both params and test stats)
        ci95_paths = ['parameter_estimates.ci95', 'test_statistic_estimates.ci95']
        for ci95_path in ci95_paths:
            if self._check_field_exists(data, ci95_path):
                ci95 = self._get_field_value(data, ci95_path)
                if not isinstance(ci95, list) or len(ci95) != 2:
                    errors.append(f"{ci95_path} must be a 2-element list [lower, upper]")
                elif None in ci95:
                    errors.append(f"{ci95_path} contains null values")

        # Validate numeric fields are actually numeric (works for both params and test stats)
        numeric_field_paths = [
            'parameter_estimates.mean',
            'parameter_estimates.variance',
            # Test statistics can have old (mean/variance) or new (median/iqr) fields
            'test_statistic_estimates.mean',
            'test_statistic_estimates.variance',
            'test_statistic_estimates.median',
            'test_statistic_estimates.iqr'
        ]
        for field_path in numeric_field_paths:
            if self._check_field_exists(data, field_path):
                val = self._get_field_value(data, field_path)
                if parse_numeric_value(val) is None:
                    errors.append(f"{field_path} is not a valid number")

        # Validate pooling weights are in [0, 1] (parameter-specific, optional)
        if 'pooling_weights' in data:
            weights = data['pooling_weights']
            if isinstance(weights, dict):
                for weight_name, weight_obj in weights.items():
                    if isinstance(weight_obj, dict) and 'value' in weight_obj:
                        val = parse_numeric_value(weight_obj['value'])
                        if val is not None and (val < 0 or val > 1):
                            errors.append(f"pooling_weights.{weight_name}.value must be in [0, 1], got {val}")

        # Validate validation weights are in [0, 1] (both params and test stats)
        if 'validation_weights' in data:
            weights = data['validation_weights']
            if isinstance(weights, dict):
                for weight_name, weight_obj in weights.items():
                    if isinstance(weight_obj, dict) and 'value' in weight_obj:
                        val = parse_numeric_value(weight_obj['value'])
                        if val is not None and (val < 0 or val > 1):
                            errors.append(f"validation_weights.{weight_name}.value must be in [0, 1], got {val}")

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def _get_field_value(self, data, field_path):
        """
        Get value of a nested field.
        field_path is dot-separated like 'parameter_estimates.mean'
        """
        parts = field_path.split('.')
        current = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]

        return current

    def validate_directory(self) -> ValidationReport:
        """Validate all YAML files in directory."""
        report = ValidationReport("Schema Compliance")

        print(f"Validating files in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info['filename']

            # Note: Legacy files should be in separate legacy directories and
            # won't be in the data_dir being validated, so no need to skip them here

            is_valid, errors = self.validate_file(file_info)

            if is_valid:
                report.add_pass(filename)
            else:
                error_msg = "; ".join(errors)
                report.add_fail(filename, error_msg)

        return report


def main():
    parser = argparse.ArgumentParser(description="Validate YAML files against template schema")
    parser.add_argument("data_dir", help="Directory with YAML files to validate")
    parser.add_argument("template", help="Path to template YAML file (e.g., templates/parameter_metadata_template_v2.yaml)")
    parser.add_argument("output", help="Output JSON file for validation report")

    args = parser.parse_args()

    # Run validation
    validator = SchemaValidator(args.data_dir, args.template)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nValidation report saved to {args.output}")

    # Exit with error code if any validations failed
    if report.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
