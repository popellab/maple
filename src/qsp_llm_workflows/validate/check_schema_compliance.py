#!/usr/bin/env python3
"""
Validate YAML files against expected schema using Pydantic models.

Checks:
- Valid YAML parsing
- Required fields present (via Pydantic)
- Field types match expectations (via Pydantic)
- Numeric values in valid ranges

Works for both parameter estimates and test statistics.
"""

from typing import Type
from pydantic import BaseModel, ValidationError as PydanticValidationError

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import (
    load_yaml_directory,
    parse_numeric_value,
    ValidationReport,
)


class SchemaValidator(Validator):
    """
    Validate YAML files against Pydantic model schema.
    Uses Pydantic models as single source of truth for schema validation.
    """

    def __init__(self, data_dir: str, model_class: Type[BaseModel] = None, validation_context: dict = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.model_class = model_class or kwargs.get("model_class")
        self.validation_context = validation_context

    @property
    def name(self) -> str:
        return "Template Compliance Validation"

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate a single YAML file against Pydantic model.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        data = file_info["data"]

        # Try to validate with Pydantic model
        try:
            # Extract only the LLM-generated content (exclude header fields)
            # Header fields are added during unpacking, not validated here
            model_data = {
                k: v
                for k, v in data.items()
                if k
                not in [
                    "parameter_name",
                    "parameter_units",
                    "parameter_definition",
                    "cancer_type",
                    "tags",
                    "derivation_id",
                    "derivation_timestamp",
                    "model_context",
                    "test_statistic_id",
                    "scenario_context",
                    "required_species",
                    "derived_species_description",
                    "validation",
                ]
            }

            self.model_class.model_validate(model_data, context=self.validation_context)
        except PydanticValidationError as e:
            # Convert Pydantic errors to human-readable messages
            for error in e.errors():
                field_path = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{field_path}: {msg}")

        # Additional specific validations (beyond Pydantic)
        # Validate ci95 is a 2-element list (Pydantic checks type, we check structure)
        ci95_paths = ["parameter_estimates.ci95", "test_statistic_estimates.ci95"]
        for ci95_path in ci95_paths:
            if self._check_field_exists(data, ci95_path):
                ci95 = self._get_field_value(data, ci95_path)
                if not isinstance(ci95, list) or len(ci95) != 2:
                    errors.append(f"{ci95_path} must be a 2-element list [lower, upper]")
                elif None in ci95:
                    errors.append(f"{ci95_path} contains null values")

        # Validate numeric fields are actually numeric
        numeric_field_paths = [
            "parameter_estimates.median",
            "parameter_estimates.iqr",
            "test_statistic_estimates.median",
            "test_statistic_estimates.iqr",
        ]
        for field_path in numeric_field_paths:
            if self._check_field_exists(data, field_path):
                val = self._get_field_value(data, field_path)
                if parse_numeric_value(val) is None:
                    errors.append(f"{field_path} is not a valid number")

        # Validate pooling weights are in [0, 1] (parameter-specific, optional)
        if "pooling_weights" in data:
            weights = data["pooling_weights"]
            if isinstance(weights, dict):
                for weight_name, weight_obj in weights.items():
                    if isinstance(weight_obj, dict) and "value" in weight_obj:
                        val = parse_numeric_value(weight_obj["value"])
                        if val is not None and (val < 0 or val > 1):
                            errors.append(
                                f"pooling_weights.{weight_name}.value must be in [0, 1], got {val}"
                            )

        # Validate validation weights are in [0, 1] (both params and test stats)
        if "validation_weights" in data:
            weights = data["validation_weights"]
            if isinstance(weights, dict):
                for weight_name, weight_obj in weights.items():
                    if isinstance(weight_obj, dict) and "value" in weight_obj:
                        val = parse_numeric_value(weight_obj["value"])
                        if val is not None and (val < 0 or val > 1):
                            errors.append(
                                f"validation_weights.{weight_name}.value must be in [0, 1], got {val}"
                            )

        # Validate biological_relevance weights are in [0, 1] (parameter-specific)
        if "biological_relevance" in data:
            weights = data["biological_relevance"]
            if isinstance(weights, dict):
                for weight_name, weight_obj in weights.items():
                    if isinstance(weight_obj, dict) and "value" in weight_obj:
                        val = parse_numeric_value(weight_obj["value"])
                        if val is not None and (val < 0 or val > 1):
                            errors.append(
                                f"biological_relevance.{weight_name}.value must be in [0, 1], got {val}"
                            )

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def _check_field_exists(self, data, field_path):
        """
        Check if a nested field exists in data.
        field_path is dot-separated like 'parameter_estimates.median'
        """
        parts = field_path.split(".")
        current = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
            if current is None:
                return False

        return True

    def _get_field_value(self, data, field_path):
        """
        Get value of a nested field.
        field_path is dot-separated like 'parameter_estimates.median'
        """
        parts = field_path.split(".")
        current = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]

        return current

    def validate(self) -> ValidationReport:
        """Validate all YAML files in directory."""
        report = ValidationReport(self.name)

        print(f"Validating files in {self.data_dir}...")
        files = load_yaml_directory(self.data_dir)

        for file_info in files:
            filename = file_info["filename"]

            is_valid, errors = self.validate_file(file_info)

            if is_valid:
                report.add_pass(filename)
            else:
                error_msg = "; ".join(errors)
                report.add_fail(filename, error_msg)

        return report
