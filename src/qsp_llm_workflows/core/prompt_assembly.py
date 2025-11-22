#!/usr/bin/env python3
"""
Prompt assembly system for QSP LLM workflows.
Handles loading, assembling, and generating prompts from modular components.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

from qsp_llm_workflows.core.header_utils import HeaderManager
from qsp_llm_workflows.core.resource_utils import (
    read_prompt,
    read_template,
    read_config,
    read_shared_prompt,
    get_package_root,
)


class PromptAssembler:
    """Assembles prompts from modular components based on configuration."""

    def __init__(self, base_dir: Path):
        """Initialize the prompt assembler with base directory."""
        self.base_dir = Path(base_dir)  # Only used for HeaderManager
        self.config = None
        self.header_manager = HeaderManager(base_dir)

    def load_config(self, config_name: str = "prompt_assembly.yaml") -> Dict[str, Any]:
        """Load prompt assembly configuration."""
        config_text = read_config(config_name)
        self.config = yaml.safe_load(config_text)
        return self.config

    def load_template(self, template_path: str) -> str:
        """
        Load a template file, excluding header fields.

        Uses HeaderManager to strip header fields from ALL template types.
        Headers are added back during result unpacking.

        Args:
            template_path: Relative path like "templates/parameter_metadata_template.yaml"
        """
        package_root = get_package_root()
        full_path = package_root / template_path
        return self.header_manager.strip_headers_from_template(full_path)

    def load_example(self, example_path: str) -> str:
        """
        Load an example file.

        Args:
            example_path: Relative path like "templates/examples/k_ECM_fib_sec_example.yaml"
        """
        package_root = get_package_root()
        full_path = package_root / example_path
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    def format_content(self, content: str, source_config: Dict[str, str]) -> str:
        """Format content based on source configuration."""
        format_type = source_config.get("format", "raw")
        prefix = source_config.get("prefix", "")

        if format_type == "yaml_code_block":
            return f"{prefix}```yaml\n{content}\n```"
        elif format_type == "raw":
            return f"{prefix}{content}"
        else:
            return content

    def _get_example_title(self, example_path: Path) -> str:
        """Extract a meaningful title from example file path."""
        path = Path(example_path)
        filename = path.stem

        # Convert common naming patterns to readable titles
        if "test_statistic_example" in filename:
            return "Tumor Volume Envelope Deviation Test Statistic"
        elif "test_statistic_derived_example" in filename:
            return "CD8+/Treg Ratio Peak Test Statistic"
        elif filename.endswith("_example"):
            # Generic pattern: remove _example and capitalize
            base_name = filename.replace("_example", "").replace("_", " ").title()
            return f"{base_name} Example"
        else:
            # Default: capitalize and replace underscores
            return filename.replace("_", " ").title()

    def assemble_prompt(self, prompt_type: str, runtime_data: Dict[str, str]) -> str:
        """Assemble a complete prompt from components."""
        if self.config is None:
            self.load_config()

        if prompt_type not in self.config["prompt_types"]:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt_config = self.config["prompt_types"][prompt_type]

        # Load base prompt from package resources
        # prompt_config["base_prompt"] is like "prompts/qsp_parameter_extraction_prompt.md"
        prompt_path = prompt_config["base_prompt"]
        if prompt_path.startswith("prompts/"):
            prompt_name = prompt_path[len("prompts/"):]
            prompt_text = read_prompt(prompt_name)
        else:
            raise ValueError(f"Unexpected prompt path format: {prompt_path}")

        # Process placeholders
        for placeholder_config in prompt_config["placeholders"]:
            placeholder_name = placeholder_config["name"]
            placeholder_tag = f"{{{{{placeholder_name}}}}}"
            source = placeholder_config["source"]

            if placeholder_tag not in prompt_text:
                continue  # Skip if placeholder not found

            replacement_content = ""

            if source == "template_file":
                template_path = prompt_config["template"]
                template_content = self.load_template(template_path)
                source_config = self.config["placeholder_sources"]["template_file"]
                replacement_content = self.format_content(template_content, source_config)

            elif source == "example_files":
                examples = prompt_config.get("examples", [])
                example_contents = []
                package_root = get_package_root()
                for example_path in examples:
                    full_example_path = package_root / example_path
                    if full_example_path.exists():
                        example_content = self.load_example(example_path)
                        example_contents.append(example_content)

                if example_contents:
                    # If multiple examples, add headers to separate them
                    if len(example_contents) > 1:
                        formatted_examples = []
                        for i, content in enumerate(example_contents, 1):
                            header = f"#### Example {i}: {self._get_example_title(examples[i-1])}"
                            formatted_examples.append(f"{header}\n\n```yaml\n{content}\n```")
                        combined_examples = "\n\n".join(formatted_examples)
                        # Don't apply additional formatting since we've already added yaml blocks
                        replacement_content = f"A complete example of well-constructed metadata:\n\n{combined_examples}"
                    else:
                        # Single example - use normal formatting
                        combined_examples = example_contents[0]
                        source_config = self.config["placeholder_sources"]["example_files"]
                        replacement_content = self.format_content(combined_examples, source_config)
                else:
                    # No example files found - remove the placeholder entirely
                    replacement_content = ""

                    # Also remove preceding "## Example" header if it exists
                    example_header_pattern = r"## Example\s*\n\s*" + re.escape(placeholder_tag)
                    if re.search(example_header_pattern, prompt_text):
                        prompt_text = re.sub(example_header_pattern, "", prompt_text)
                        continue  # Skip the normal replacement since we already handled it

            elif source == "runtime":
                if placeholder_name in runtime_data:
                    source_config = self.config["placeholder_sources"]["runtime"]
                    replacement_content = self.format_content(
                        runtime_data[placeholder_name], source_config
                    )
                else:
                    replacement_content = f"[{placeholder_name} - TO BE PROVIDED]"

            elif source == "shared_file":
                # Load content from shared file using package resources
                shared_path = placeholder_config.get("path", "")
                if shared_path:
                    # shared_path is like "prompts/shared/source_and_validation_rubrics.md"
                    if shared_path.startswith("prompts/shared/"):
                        shared_name = shared_path[len("prompts/shared/"):]
                        try:
                            shared_content = read_shared_prompt(shared_name)
                            source_config = self.config["placeholder_sources"]["shared_file"]
                            replacement_content = self.format_content(shared_content, source_config)
                        except Exception as e:
                            print(f"Warning: Could not read shared file {shared_path}: {e}")
                            replacement_content = f"[{placeholder_name} - FILE NOT FOUND]"
                    else:
                        print(f"Warning: Unexpected shared file path format: {shared_path}")
                        replacement_content = f"[{placeholder_name} - INVALID PATH]"
                else:
                    replacement_content = f"[{placeholder_name} - NO PATH SPECIFIED]"

            elif source == "used_primary_studies":
                # Special handling for USED_PRIMARY_STUDIES placeholder
                # This queries existing files to prevent duplicate source usage
                replacement_content = self._get_used_primary_studies(prompt_type, runtime_data)

            # Replace placeholder in prompt
            prompt_text = prompt_text.replace(placeholder_tag, replacement_content)

        return prompt_text

    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types."""
        if self.config is None:
            self.load_config()
        return list(self.config["prompt_types"].keys())

    def validate_runtime_data(self, prompt_type: str, runtime_data: Dict[str, str]) -> bool:
        """Validate that required runtime data is provided."""
        if self.config is None:
            self.load_config()

        prompt_config = self.config["prompt_types"][prompt_type]
        required_runtime_placeholders = [
            p["name"] for p in prompt_config["placeholders"] if p["source"] == "runtime"
        ]

        missing = [key for key in required_runtime_placeholders if key not in runtime_data]
        if missing:
            raise ValueError(f"Missing required runtime data for {prompt_type}: {missing}")

        return True

    def _get_used_primary_studies(self, prompt_type: str, runtime_data: Dict[str, str]) -> str:
        """
        Get used primary studies based on prompt type.

        Args:
            prompt_type: Type of prompt (parameter or test_statistic)
            runtime_data: Runtime data containing identifiers

        Returns:
            Formatted string listing used primary studies
        """
        # Determine storage directory based on prompt type
        if prompt_type == "parameter_extraction":
            storage_dir = self.base_dir.parent / "qsp-metadata-storage" / "parameter_estimates"
            runtime_data.get("PARAMETER_INFO", "")
            # Extract parameter name from PARAMETER_INFO if needed
            # For now, assume it's passed directly in runtime_data
            param_name = runtime_data.get("parameter_name", "")
            context_hash = runtime_data.get("context_hash", "")

            if not param_name or not context_hash:
                # If we don't have the required info, return default message
                return "None - this is the first derivation for this parameter"

            return self._get_used_primary_studies_for_parameter(
                param_name, context_hash, storage_dir
            )

        elif prompt_type == "test_statistic":
            storage_dir = self.base_dir.parent / "qsp-metadata-storage" / "test_statistics"
            test_stat_id = runtime_data.get("test_statistic_id", "")
            context_hash = runtime_data.get("context_hash", "")

            if not test_stat_id or not context_hash:
                # If we don't have the required info, return default message
                return "None - this is the first derivation for this test statistic"

            return self._get_used_primary_studies_for_test_statistic(
                test_stat_id, context_hash, storage_dir
            )

        else:
            # Unknown prompt type, return default message
            return "None - this is the first derivation"

    def _get_used_primary_studies_for_parameter(
        self, parameter_name: str, context_hash: str, storage_dir: Path
    ) -> str:
        """
        Query existing parameter files with matching name and context_hash
        to extract primary studies that have already been used.

        Args:
            parameter_name: Name of the parameter
            context_hash: Context hash identifying the parameter context
            storage_dir: Path to parameter storage directory

        Returns:
            Formatted string listing used primary studies, or empty string if none found
        """
        # Find all YAML files matching the parameter name pattern
        pattern = storage_dir / f"{parameter_name}_*.yaml"
        matching_files = glob.glob(str(pattern))

        used_studies = []

        for filepath in matching_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # Check if context_hash matches
                file_context_hash = data.get("context_hash", "")
                if file_context_hash != context_hash:
                    continue

                # Extract primary_data_sources
                primary_sources = data.get("primary_data_sources", [])
                for source in primary_sources:
                    title = source.get("title", "")
                    first_author = source.get("first_author", "")
                    year = source.get("year", "")
                    doi = source.get("doi", "")

                    if title and first_author and year:
                        # Format: "Title (First Author et al. Year, DOI: xxx)"
                        study_str = f"{title} ({first_author} et al. {year}"
                        if doi and doi.lower() != "null":
                            study_str += f", DOI: {doi}"
                        study_str += ")"

                        # Avoid duplicates
                        if study_str not in used_studies:
                            used_studies.append(study_str)

            except Exception as e:
                # Skip files that can't be parsed
                print(f"Warning: Could not parse {filepath}: {e}")
                continue

        if not used_studies:
            return "None - this is the first derivation for this parameter"

        # Format as bulleted list
        formatted_list = "\n".join([f"- {study}" for study in used_studies])
        return formatted_list

    def _get_used_primary_studies_for_test_statistic(
        self, test_statistic_id: str, context_hash: str, storage_dir: Path
    ) -> str:
        """
        Query existing test statistic files with matching ID and context_hash
        to extract primary studies that have already been used.

        Args:
            test_statistic_id: ID of the test statistic
            context_hash: Context hash identifying the test scenario
            storage_dir: Path to test statistics storage directory

        Returns:
            Formatted string listing used primary studies, or empty string if none found
        """
        # Find all YAML files matching the test statistic ID pattern
        pattern = storage_dir / f"{test_statistic_id}_*.yaml"
        matching_files = glob.glob(str(pattern))

        used_studies = []

        for filepath in matching_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # Check if context_hash matches
                file_context_hash = data.get("context_hash", "")
                if file_context_hash != context_hash:
                    continue

                # Extract primary_data_sources
                primary_sources = data.get("primary_data_sources", [])
                for source in primary_sources:
                    title = source.get("title", "")
                    first_author = source.get("first_author", "")
                    year = source.get("year", "")
                    doi = source.get("doi", "")

                    if title and first_author and year:
                        # Format: "Title (First Author et al. Year, DOI: xxx)"
                        study_str = f"{title} ({first_author} et al. {year}"
                        if doi and doi.lower() != "null":
                            study_str += f", DOI: {doi}"
                        study_str += ")"

                        # Avoid duplicates
                        if study_str not in used_studies:
                            used_studies.append(study_str)

            except Exception as e:
                # Skip files that can't be parsed
                print(f"Warning: Could not parse {filepath}: {e}")
                continue

        if not used_studies:
            return "None - this is the first derivation for this test statistic"

        # Format as bulleted list
        formatted_list = "\n".join([f"- {study}" for study in used_studies])
        return formatted_list
