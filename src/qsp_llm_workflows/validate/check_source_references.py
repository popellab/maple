#!/usr/bin/env python3
"""
Validate source reference integrity.

Checks:
- Every non-null source_ref in inputs has a matching source definition
- All sources are properly formatted (title, first_author, year, doi present)
- No orphaned source references

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_source_references.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/source_reference_validation.json
"""

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


class SourceReferenceValidator(Validator):
    """
    Validate source reference integrity.
    Works for both parameters and test statistics.
    """

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)

    @property
    def name(self) -> str:
        return "Source Reference Validation"

    def collect_sources(self, data: dict) -> dict:
        """
        Collect all defined sources from YAML.

        Returns:
            Dict mapping source_tag to source definition
        """
        sources = {}

        # Collect from primary_data_sources
        if "primary_data_sources" in data:
            pds = data["primary_data_sources"]
            if isinstance(pds, list):
                for source in pds:
                    if isinstance(source, dict) and "source_tag" in source:
                        sources[source["source_tag"]] = {"type": "primary", "definition": source}
            elif isinstance(pds, dict):
                for tag, source in pds.items():
                    sources[tag] = {
                        "type": "primary",
                        "definition": source if isinstance(source, dict) else {"source_tag": tag},
                    }

        # Collect from secondary_data_sources
        if "secondary_data_sources" in data:
            sds = data["secondary_data_sources"]
            if isinstance(sds, list):
                for source in sds:
                    if isinstance(source, dict) and "source_tag" in source:
                        sources[source["source_tag"]] = {"type": "secondary", "definition": source}
            elif isinstance(sds, dict):
                for tag, source in sds.items():
                    sources[tag] = {
                        "type": "secondary",
                        "definition": source if isinstance(source, dict) else {"source_tag": tag},
                    }

        # Collect from methodological_sources
        if "methodological_sources" in data:
            ms = data["methodological_sources"]
            if isinstance(ms, list):
                for source in ms:
                    if isinstance(source, dict) and "source_tag" in source:
                        sources[source["source_tag"]] = {
                            "type": "methodological",
                            "definition": source,
                        }
            elif isinstance(ms, dict):
                for tag, source in ms.items():
                    sources[tag] = {
                        "type": "methodological",
                        "definition": source if isinstance(source, dict) else {"source_tag": tag},
                    }

        return sources

    def extract_inputs_from_yaml(self, data: dict) -> list:
        """
        Extract inputs from either parameter_estimates or test_statistic_estimates.

        Returns:
            List of input dicts
        """
        # Try parameter_estimates first (params)
        if "parameter_estimates" in data and isinstance(data["parameter_estimates"], dict):
            if "inputs" in data["parameter_estimates"]:
                return data["parameter_estimates"]["inputs"]

        # Try test_statistic_estimates (test stats)
        if "test_statistic_estimates" in data and isinstance(
            data["test_statistic_estimates"], dict
        ):
            if "inputs" in data["test_statistic_estimates"]:
                return data["test_statistic_estimates"]["inputs"]

        return []

    def collect_source_refs(self, data: dict) -> list:
        """
        Collect all source_ref values from inputs.

        Returns:
            List of (input_name, source_ref) tuples
        """
        refs = []
        inputs = self.extract_inputs_from_yaml(data)

        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            name = inp.get("name", "unnamed")
            source_ref = inp.get("source_ref")

            # Include both null and non-null refs for tracking
            refs.append((name, source_ref))

        return refs

    def validate_source_definition(self, source_tag: str, source_info: dict) -> list:
        """
        Validate that a source definition is complete.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        definition = source_info["definition"]
        source_type = source_info["type"]

        # Required fields for all sources
        common_fields = ["title", "first_author", "year"]

        for field in common_fields:
            if field not in definition or not definition[field]:
                errors.append(f"Source '{source_tag}': missing required field '{field}'")

        # Check DOI field based on source type
        # Primary sources use 'doi', secondary/methodological use 'doi_or_url'
        if source_type == "primary":
            if "doi" not in definition or not definition["doi"]:
                errors.append(f"Source '{source_tag}': missing required field 'doi'")
        else:  # secondary or methodological
            if "doi_or_url" not in definition or not definition["doi_or_url"]:
                errors.append(f"Source '{source_tag}': missing required field 'doi_or_url'")

        return errors

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate source references in a single YAML file.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        data = file_info["data"]
        file_info["filename"]

        # Collect all defined sources
        sources = self.collect_sources(data)

        # Collect all source_refs from inputs
        source_refs = self.collect_source_refs(data)

        # Check each source reference
        for input_name, source_ref in source_refs:
            # Null source_refs are allowed (e.g., for assumptions, random seeds)
            if source_ref is None:
                continue

            # Check if source_ref exists in sources
            if source_ref not in sources:
                errors.append(
                    f"Input '{input_name}': references source '{source_ref}' which is not defined"
                )

        # Validate all source definitions are complete
        for source_tag, source_info in sources.items():
            source_errors = self.validate_source_definition(source_tag, source_info)
            errors.extend(source_errors)

        is_valid = len(errors) == 0
        return (is_valid, errors)

    def validate(self) -> ValidationReport:
        """Validate source references in all YAML files."""
        report = ValidationReport(self.name)

        print(f"Validating source references in {self.data_dir}...")
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
