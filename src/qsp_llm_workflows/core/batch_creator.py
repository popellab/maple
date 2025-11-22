#!/usr/bin/env python3
"""
Base class for creating batch requests for different types of LLM processing tasks.

This module provides a common framework for generating OpenAI batch API requests
with consistent patterns for request creation, file output, and error handling.
"""

import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from pydantic import BaseModel

from openai.lib._pydantic import to_strict_json_schema

from qsp_llm_workflows.core.prompt_assembly import PromptAssembler
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic


class BatchCreator(ABC):
    """
    Abstract base class for creating OpenAI batch API requests.

    Provides common functionality for request creation, file I/O, and batch processing
    while allowing subclasses to implement specific logic for different prompt types.
    """

    def __init__(self, base_dir: Path, prompt_assembler: Optional[PromptAssembler] = None):
        """
        Initialize the batch creator.

        Args:
            base_dir: Base directory of the project (used for relative path resolution)
            prompt_assembler: Optional PromptAssembler instance (created if not provided)
        """
        self.base_dir = Path(base_dir)
        self.prompt_assembler = prompt_assembler or PromptAssembler(self.base_dir)

    def create_request(
        self, custom_id: str, prompt: str, pydantic_model: Type[BaseModel], **kwargs
    ) -> Dict[str, Any]:
        """
        Create a standardized batch API request with structured outputs.

        Args:
            custom_id: Unique identifier for this request
            prompt: The prompt text to send to the model
            pydantic_model: Pydantic model class for structured output
            **kwargs: Additional request parameters (model, reasoning effort, etc.)

        Returns:
            Dictionary representing a batch API request
        """
        # Set defaults
        model = kwargs.get("model", "gpt-5")
        reasoning_effort = kwargs.get("reasoning_effort", "high")

        # Convert Pydantic model to strict JSON schema for batch API
        schema_name = pydantic_model.__name__.lower()
        schema = to_strict_json_schema(pydantic_model)

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "input": prompt,
                "reasoning": {"effort": reasoning_effort},
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            },
        }

        return request

    def write_batch_file(self, requests: List[Dict], output_path: Path) -> None:
        """
        Write batch requests to JSONL file.

        Args:
            requests: List of batch request dictionaries
            output_path: Path where to write the JSONL file
        """
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        print(f"Created {len(requests)} requests in {output_path}")

    def get_default_output_path(self) -> Path:
        """
        Get the default output path for this batch creator type.

        Returns:
            Default path for batch request files
        """
        return self.base_dir / "batch_jobs" / f"{self.get_batch_type()}_requests.jsonl"

    @abstractmethod
    def get_batch_type(self) -> str:
        """
        Get the type identifier for this batch creator.

        Used for default file naming and logging.

        Returns:
            String identifier for this batch type (e.g., "parameter", "pooling_metadata")
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Process input data and generate batch requests.

        This method should be implemented by subclasses to handle their specific
        input processing and prompt generation logic.

        Returns:
            List of batch request dictionaries ready for API submission
        """
        pass

    def run(self, output_path: Optional[Path] = None, *args, **kwargs) -> Path:
        """
        Execute the complete batch creation process.

        Args:
            output_path: Optional output path (uses default if not provided)
            *args, **kwargs: Arguments passed to the process() method

        Returns:
            Path to the created batch file
        """
        # Generate requests
        requests = self.process(*args, **kwargs)

        # Determine output path
        if output_path is None:
            output_path = self.get_default_output_path()
        else:
            output_path = Path(output_path)

        # Write batch file
        self.write_batch_file(requests, output_path)

        # Log batch type information
        print(f"Batch type: {self.get_batch_type()}")

        return output_path


class ParameterBatchCreator(BatchCreator):
    """
    Creates batch requests for parameter extraction from scientific literature.

    Processes CSV input with embedded parameter definitions and model context, generating
    prompts for comprehensive literature extraction. Requires columns: cancer_type,
    parameter_name, parameter_units, parameter_description, model_context, definition_hash.
    """

    def get_batch_type(self) -> str:
        return "parameter"

    def format_model_context(self, model_context_json: str) -> str:
        """
        Parse and format the model context JSON into readable text for the LLM.

        Args:
            model_context_json: JSON string containing reactions_and_rules with model context

        Returns:
            Formatted markdown text describing the model context
        """
        import json

        try:
            context_data = json.loads(model_context_json)
        except json.JSONDecodeError as e:
            return f"Error parsing model context: {e}"

        output = []

        # Add derived from context if available
        if "derived_from_context" in context_data:
            derived = context_data["derived_from_context"]
            if derived:
                output.append("## Parameter Context")
                for item in derived:
                    output.append(
                        f"- **{item.get('name', 'Unknown')}**: {item.get('description', 'No description')}"
                    )
                output.append("")

        # Add reactions and rules
        if "reactions_and_rules" in context_data:
            reactions = context_data["reactions_and_rules"]
            if reactions:
                output.append("## Model Usage")
                output.append(
                    f"This parameter appears in {len(reactions)} reaction(s) and/or rule(s):"
                )
                output.append("")

                for i, rxn in enumerate(reactions, 1):
                    # Reaction or rule
                    if rxn.get("reaction"):
                        output.append(f"### {i}. Reaction: `{rxn['reaction']}`")
                        if rxn.get("reaction_rate"):
                            output.append(f"**Rate:** `{rxn['reaction_rate']}`")
                    elif rxn.get("rule"):
                        output.append(f"### {i}. Rule ({rxn.get('rule_type', 'unknown type')})")
                        output.append(f"**Expression:** `{rxn['rule']}`")

                    output.append("")

                    # Other parameters
                    other_params = rxn.get("other_parameters", [])
                    if other_params:
                        output.append("**Related Parameters:**")
                        for param in other_params:
                            name = param.get("name", "Unknown")
                            desc = param.get("description", "")
                            if desc:
                                output.append(f"- `{name}`: {desc}")
                            else:
                                output.append(f"- `{name}`")
                        output.append("")

                    # Other species
                    other_species = rxn.get("other_species", [])
                    if other_species:
                        output.append("**Related Species:**")
                        for species in other_species:
                            name = species.get("name", "Unknown")
                            desc = species.get("description", "")
                            if desc:
                                output.append(f"- `{name}`: {desc}")
                            else:
                                output.append(f"- `{name}`")
                        output.append("")

        return "\n".join(output) if output else "No model context available."

    def process(self, input_csv: Path, parameter_storage_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Process parameter extraction inputs and generate batch requests.

        Args:
            input_csv: CSV file with columns: cancer_type, parameter_name, parameter_units,
                      parameter_description, model_context (JSON), definition_hash
            parameter_storage_dir: Optional path to parameter storage directory for existing studies

        Returns:
            List of batch request dictionaries
        """
        import csv
        from .parameter_utils import render_parameter_to_search, collect_existing_studies

        # Process CSV and create requests
        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                cancer_type = row["cancer_type"]
                parameter_name = row["parameter_name"]
                units = row.get("parameter_units", "")
                definition = row.get("parameter_description", "")
                model_context_json = row.get("model_context", "{}")
                definition_hash = row.get("definition_hash", "")

                # Format the model context from JSON
                model_context_block = self.format_model_context(model_context_json)

                # Build parameter info block with cancer type
                parameter_block = render_parameter_to_search(
                    parameter_name, units, definition, cancer_type
                )

                # Collect existing studies to avoid re-extracting from same sources
                # Use definition_hash from CSV (same as context_hash in model_context)
                if parameter_storage_dir is None:
                    parameter_storage_dir = (
                        self.base_dir.parent / "qsp-metadata-storage" / "parameter_estimates"
                    )
                existing_studies = collect_existing_studies(
                    cancer_type, parameter_name, parameter_storage_dir, definition_hash
                )

                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "EXISTING_STUDIES": existing_studies,
                    "PARAMETER_INFO": parameter_block,
                    "MODEL_CONTEXT": model_context_block,
                    "parameter_name": parameter_name,
                    "context_hash": definition_hash,
                }

                # Assemble the prompt
                prompt = self.prompt_assembler.assemble_prompt("parameter_extraction", runtime_data)

                # Create batch request with structured outputs
                custom_id = f"{cancer_type}_{parameter_name}_{i}"
                request = self.create_request(custom_id, prompt, ParameterMetadata)
                requests.append(request)

        return requests


class TestStatisticBatchCreator(BatchCreator):
    """
    Creates batch requests for test statistic generation from biological expectations.

    Processes CSV input with test statistic descriptions and biological expectations, generating
    prompts that create comprehensive test statistic definitions for QSP model validation.
    """

    def get_batch_type(self) -> str:
        return "test_stat"

    def _get_default_species_units(self) -> Dict[str, str]:
        """
        Get default species units mapping for common species types.

        Returns:
            Dictionary mapping species names to their default units
        """
        return {
            "CD8": "cells",
            "CD4": "cells",
            "Treg": "cells",
            "Th": "cells",
            "APC": "cells",
            "mAPC": "cells",
            "DC": "cells",
            "Mac_M1": "cells",
            "Mac_M2": "cells",
            "MDSC": "cells",
            "C": "cells",
            "C_x": "cells",
            "C1": "cells",
            "GVAX_cells": "cells",
            "TumorVolume": "cm³",
            "K": "dimensionless",
            "IL2": "pg/mL",
            "IL10": "pg/mL",
            "IL12": "pg/mL",
            "IFNg": "pg/mL",
            "TNFa": "pg/mL",
            "TGFb": "pg/mL",
            "GMCSF": "pg/mL",
            "CCL2": "pg/mL",
            "NO": "μM",
            "ArgI": "units/mg protein",
            "ECM": "dimensionless",
            "CAF": "cells",
            "Fib": "cells",
            "c_vas": "pg/mL",
            "aPD1": "nanomolarity",
            "aPDL1": "nanomolarity",
            "aCTLA4": "nanomolarity",
            "T_eff": "cells",
            "CD8_exh": "cells",
            "Th_exh": "cells",
        }

    def _load_species_units_mapping(self) -> Dict[str, str]:
        """
        Get species units mapping for common species types.

        Since model_context.txt now contains all species with units,
        this method returns default unit mappings as a helper for formatting.

        Returns:
            Dictionary mapping species names/patterns to units
        """
        return self._get_default_species_units()

    def _parse_species_with_units(
        self, required_species: str, species_units_mapping: Dict[str, str]
    ) -> str:
        """
        Parse required_species string and format with units information.

        Args:
            required_species: Comma-separated string of species (e.g., "V_T.CD8,V_T.Treg")
            species_units_mapping: Dictionary mapping species names to units

        Returns:
            Formatted string with species and their units
        """
        if not required_species.strip():
            return ""

        species_list = [s.strip() for s in required_species.split(",") if s.strip()]
        formatted_species = []

        for species in species_list:
            # Remove any square brackets that might be in the input
            species = species.strip("[]").strip()
            # Extract the species name from compartment notation (e.g., V_T.CD8 -> CD8)
            if "." in species:
                compartment, species_name = species.split(".", 1)
            else:
                compartment, species_name = "", species

            # Look up units
            units = "dimensionless"  # default

            # Try exact match first
            if species_name in species_units_mapping:
                units = species_units_mapping[species_name]
            else:
                # Try partial matches for compound names
                for pattern, mapped_units in species_units_mapping.items():
                    if pattern.lower() in species_name.lower():
                        units = mapped_units
                        break

            # Format the output
            if compartment:
                formatted_species.append(
                    f"- `{species}`: {species_name} in {compartment} compartment (units: {units})"
                )
            else:
                formatted_species.append(f"- `{species}`: {species_name} (units: {units})")

        return "\n".join(formatted_species)

    def process(self, input_csv: Path, model_context_csv: Path = None) -> List[Dict[str, Any]]:
        """
        Process test statistic inputs and generate batch requests.

        Args:
            input_csv: CSV file with test_statistic_id and biological expectation columns
            model_context_csv: Optional CSV file with model structure information

        Returns:
            List of batch request dictionaries
        """
        import csv
        import pandas as pd

        # Load species units from simbio_parameters.csv
        species_units_mapping = self._load_species_units_mapping()

        # Load model context if provided
        model_context_info = {}
        if model_context_csv and Path(model_context_csv).exists():
            model_df = pd.read_csv(model_context_csv)
            # Create a lookup for model variables and their descriptions
            for _, row in model_df.iterrows():
                var_name = str(row.get("Variable", "")).strip()
                if var_name:
                    model_context_info[var_name] = {
                        "description": str(row.get("Description", "")).strip(),
                        "units": str(row.get("Units", "")).strip(),
                        "compartment": str(row.get("Compartment", "")).strip(),
                    }

        # Process CSV and create requests
        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                test_statistic_id = row.get("test_statistic_id", f"test_stat_{i}")
                model_context = row.get("model_context", "")
                scenario_context = row.get("scenario_context", "")
                required_species = row.get("required_species", "")
                derived_species_description = row.get("derived_species_description", "")

                # Extract or generate context hash
                context_hash = row.get("context_hash", "")
                if not context_hash:
                    # Auto-generate hash from model_context + scenario_context
                    # (excludes required_species so all test statistics for same model+scenario share hash)
                    import hashlib

                    context_str = f"{model_context}_{scenario_context}"
                    context_hash = hashlib.md5(context_str.encode()).hexdigest()[:8]

                if not model_context.strip():
                    print(f"Warning: Empty model context for {test_statistic_id}, skipping")
                    continue

                if not scenario_context.strip():
                    print(f"Warning: Empty scenario context for {test_statistic_id}, skipping")
                    continue

                if not required_species.strip():
                    print(f"Warning: Empty required species for {test_statistic_id}, skipping")
                    continue

                if not derived_species_description.strip():
                    print(
                        f"Warning: Empty derived species description for {test_statistic_id}, skipping"
                    )
                    continue

                # Use provided context directly, with optional model context CSV enhancement
                model_context_block = model_context
                if model_context_info:
                    model_context_block += "\n\n**Additional Model Variables:**\n\n"
                    for var_name, info in model_context_info.items():
                        model_context_block += f"- `{var_name}`: {info['description']}"
                        if info["units"]:
                            model_context_block += f" (units: {info['units']})"
                        if info["compartment"]:
                            model_context_block += f" [compartment: {info['compartment']}]"
                        model_context_block += "\n"

                # Use provided scenario context directly
                scenario_context_block = scenario_context

                # Parse required species with units information
                required_species_with_units = self._parse_species_with_units(
                    required_species, species_units_mapping
                )

                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "MODEL_CONTEXT": model_context_block,
                    "SCENARIO_CONTEXT": scenario_context_block,
                    "REQUIRED_SPECIES_WITH_UNITS": required_species_with_units,
                    "DERIVED_SPECIES_DESCRIPTION": derived_species_description,
                    "test_statistic_id": test_statistic_id,
                    "context_hash": context_hash,
                }

                # Assemble the prompt
                prompt = self.prompt_assembler.assemble_prompt("test_statistic", runtime_data)

                # Create batch request with structured outputs
                custom_id = f"test_stat_{test_statistic_id}_{i}"
                request = self.create_request(custom_id, prompt, TestStatistic, reasoning_effort="high")
                requests.append(request)

        return requests


class ValidationFixBatchCreator(BatchCreator):
    """
    Creates batch requests to fix validation errors in YAML files.

    Loads YAMLs with validation failures and creates prompts asking
    LLM to fix the specific errors while preserving all other content.
    """

    def __init__(self, base_dir: Path):
        """Initialize with base directory."""
        super().__init__(base_dir)
        from header_utils import HeaderManager

        self.header_manager = HeaderManager(base_dir)

    def get_batch_type(self) -> str:
        return "validation_fix"

    def _extract_filename_from_item(self, item: str) -> str:
        """
        Extract filename from validation item string.

        Handles multiple formats:
        - Simple: "filename.yaml"
        - Reference-level: "filename.yaml → REFERENCE"
        - Input-level: "filename.yaml / input 'name' (value=X)"

        Args:
            item: Item string from validation report

        Returns:
            Extracted filename
        """
        # Check for reference-level format (arrow)
        if " → " in item:
            return item.split(" → ")[0].strip()

        # Check for input-level format (slash)
        if " / input " in item:
            return item.split(" / input ")[0].strip()

        # Simple format - just return as-is
        return item.strip()

    def load_validation_results(self, validation_dir: Path) -> Dict[str, List[str]]:
        """
        Load all validation JSON reports and aggregate errors by filename.

        Args:
            validation_dir: Directory containing validation JSON reports

        Returns:
            Dictionary mapping filename to list of error messages
        """
        errors_by_file = {}

        # Load each validation report
        validation_files = list(validation_dir.glob("*.json"))
        if not validation_files:
            print(f"Warning: No validation JSON files found in {validation_dir}")
            return errors_by_file

        for validation_file in validation_files:
            # Skip master summary
            if validation_file.name == "master_validation_summary.json":
                continue

            try:
                with open(validation_file, "r", encoding="utf-8") as f:
                    report = json.load(f)

                # Extract validation type from filename
                validation_type = validation_file.stem.replace("_", " ").title()

                # Process failed items
                failed_items = report.get("failed", [])
                for item in failed_items:
                    item_str = item["item"]
                    reason = item["reason"]

                    # Extract filename from item (handles reference/input-level formats)
                    filename = self._extract_filename_from_item(item_str)

                    # Format error with validation type
                    error_msg = f"{validation_type}:\n    {reason}"

                    if filename not in errors_by_file:
                        errors_by_file[filename] = []
                    errors_by_file[filename].append(error_msg)

            except Exception as e:
                print(f"Warning: Could not load {validation_file}: {e}")
                continue

        return errors_by_file

    def load_yaml_content(self, yaml_path: Path) -> Optional[str]:
        """
        Load YAML file content.

        Args:
            yaml_path: Path to YAML file

        Returns:
            YAML content as string, or None if error
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not load {yaml_path}: {e}")
            return None

    def load_template_content(self, template_path: Path) -> Optional[str]:
        """
        Load template YAML file content.

        Args:
            template_path: Path to template file

        Returns:
            Template content as string, or None if error
        """
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not load template {template_path}: {e}")
            return None

    def _generate_example_json(self, template_type: str) -> str:
        """
        Generate example JSON output format based on template type.

        Args:
            template_type: Type of template (parameter_metadata, test_statistic)

        Returns:
            Example JSON string formatted with code fence
        """
        if template_type == "test_statistic":
            return """Example output format for test statistics:
```json
{
  "model_output": {
    "code": "import numpy as np\\n\\ndef compute_test_statistic(...):\\n    ..."
  },
  "test_statistic_definition": "...",
  "study_overview": "...",
  "study_design": "...",
  "test_statistic_estimates": {
    "inputs": [...],
    "derivation_code": "...",
    "median": 1.23,
    "iqr": 0.45,
    "ci95": [0.5, 2.0],
    "units": "...",
    "key_assumptions": {
      "1": "...",
      "2": "...",
      "3": "..."
    }
  },
  "derivation_explanation": "...",
  "key_study_limitations": "...",
  "primary_data_sources": [...],
  "secondary_data_sources": [...],
  "methodological_sources": [...],
  "validation_weights": {
    "species_match": {"value": 1.0, "justification": "..."},
    ...
  }
}
```"""
        elif template_type == "parameter_metadata":
            return """Example output format for parameter estimates:
```json
{
  "mathematical_role": "...",
  "parameter_range": "positive_reals",
  "study_overview": "...",
  "study_design": "...",
  "parameter_estimates": {
    "inputs": [
      {
        "name": "...",
        "value": 1.23,
        "units": "...",
        "description": "...",
        "source_ref": "SOURCE_TAG",
        "value_table_or_section": "Table 2",
        "value_snippet": "...",
        "units_table_or_section": "Table 2",
        "units_snippet": "..."
      }
    ],
    "derivation_code": "```python\\n...\\n```",
    "median": 1.23,
    "iqr": 0.45,
    "ci95": [0.5, 2.0],
    "units": "1/day",
    "key_assumptions": {
      "1": "...",
      "2": "...",
      "3": "..."
    }
  },
  "derivation_explanation": "...",
  "key_study_limitations": "...",
  "primary_data_sources": [...],
  "secondary_data_sources": [...],
  "methodological_sources": [...],
  "biological_relevance": {
    "species_match": {"value": 1.0, "justification": "..."},
    ...
  }
}
```"""

    def create_fix_prompt(
        self, yaml_content: str, errors: List[str], template_content: str, template_type: str
    ) -> str:
        """
        Create fix prompt by assembling content (without headers), errors, and template.

        Args:
            yaml_content: Original YAML content
            errors: List of error messages for this file
            template_content: Template YAML content for reference
            template_type: Type of template (parameter_metadata, test_statistic)

        Returns:
            Complete fix prompt
        """
        # Parse YAML to detect schema version and strip headers
        data = yaml.safe_load(yaml_content)
        schema_version = data.get("schema_version", "v1")

        # Strip headers from original YAML (LLM should only see content)
        headers, content_dict = self.header_manager.strip_headers_from_yaml_string(
            yaml_content, template_type, schema_version
        )

        # Convert content dict back to YAML string
        content_yaml = yaml.dump(
            content_dict, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

        # Use the validation fix prompt template
        prompt_path = get_prompt_path("validation_fix_prompt.md")

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()

            # Format errors
            formatted_errors = "\n".join(f"- {error}" for error in errors)

            # Generate appropriate example JSON based on template type
            example_json = self._generate_example_json(template_type)

            # Fill placeholders (use content YAML without headers)
            prompt = template.replace("{{YAML_CONTENT}}", content_yaml)
            prompt = prompt.replace("{{VALIDATION_ERRORS}}", formatted_errors)
            prompt = prompt.replace("{{TEMPLATE_CONTENT}}", template_content)
            prompt = prompt.replace("{{EXAMPLE_JSON}}", example_json)

            return prompt
        else:
            # Fallback if template doesn't exist yet
            formatted_errors = "\n".join(f"- {error}" for error in errors)

            return f"""You are helping to fix validation errors in a QSP metadata YAML file.

**Original YAML:**
```yaml
{yaml_content}
```

**Template for Reference:**
```yaml
{template_content}
```

**Task:**
Fix the validation errors listed below. While doing so, also review ALL units_snippet fields (even if not flagged) to ensure they mention the units.

**Guidelines:**
- Maintain original structure and formatting
- Don't change fields that aren't mentioned in errors
- If adding missing fields, use appropriate null/placeholder values
- For schema compliance errors, consult the template
- For code execution errors, verify the derivation code makes sense for the parameter; ensure reported values exactly match calculated values
- For text snippet errors, look up the source and extract snippets verbatim; ensure they're from the correct context
- For source reference errors, ensure all inputs have valid source_ref fields
- For DOI errors, verify the source exists, citation fields match the DOI, and the reference is appropriate for the data

---

**Validation Errors to Fix:**
{formatted_errors}

---

**Output:**
Return the corrected metadata as JSON inside a ```json code fence. The unpacker will convert to YAML with proper headers. Do not include header fields like cancer_type, tags, schema_version - those will be added during unpacking."""

    def process(
        self, validation_dir: Path, yaml_dir: Path, template_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Process validation results and generate fix batch requests.

        Args:
            validation_dir: Directory with validation JSON files
            yaml_dir: Directory with YAML files to fix
            template_path: Path to template for reference

        Returns:
            List of batch request dictionaries
        """
        # Load validation results
        errors_by_file = self.load_validation_results(validation_dir)

        if not errors_by_file:
            print("No validation errors found. Nothing to fix!")
            return []

        print(f"Found {len(errors_by_file)} files with validation errors")

        # Load template content (with headers stripped)
        template_content = self.header_manager.strip_headers_from_template(template_path)
        if not template_content:
            print(f"Error: Could not load template from {template_path}")
            return []

        # Detect template type from template path
        template_type = self._detect_template_type_from_path(template_path)

        # Create fix requests
        requests = []
        for filename, errors in errors_by_file.items():
            # Find the YAML file
            yaml_path = Path(yaml_dir) / filename
            if not yaml_path.exists():
                print(f"Warning: YAML file not found: {yaml_path}, skipping")
                continue

            # Load YAML content
            yaml_content = self.load_yaml_content(yaml_path)
            if not yaml_content:
                continue

            # Create fix prompt (strips headers from YAML content)
            prompt = self.create_fix_prompt(yaml_content, errors, template_content, template_type)

            # Select Pydantic model based on template type
            pydantic_model = (
                ParameterMetadata if template_type == "parameter_metadata" else TestStatistic
            )

            # Create custom ID from filename
            file_stem = yaml_path.stem
            custom_id = f"fix_{file_stem}"

            request = self.create_request(custom_id, prompt, pydantic_model, reasoning_effort="high")

            requests.append(request)
            print(f"  Created fix request for {filename} ({len(errors)} error(s))")

        return requests

    def _detect_template_type_from_path(self, template_path: Path) -> str:
        """
        Detect template type from file path.

        Args:
            template_path: Path to template file

        Returns:
            Template type string (parameter_metadata, test_statistic)
        """
        name = template_path.name.lower()

        if "parameter" in name:
            return "parameter_metadata"
        elif "test_stat" in name:
            return "test_statistic"
        else:
            # Default to parameter_metadata
            return "parameter_metadata"
