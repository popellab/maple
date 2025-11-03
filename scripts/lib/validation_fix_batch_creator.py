#!/usr/bin/env python3
"""
Batch creator for fixing validation errors in YAML metadata files.

Creates batch requests that send failed YAMLs with their validation errors
back to the LLM for automatic correction.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import sys

# Ensure lib directory is in path for imports
lib_dir = Path(__file__).parent
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from batch_creator import BatchCreator
from header_utils import HeaderManager


class ValidationFixBatchCreator(BatchCreator):
    """
    Creates batch requests to fix validation errors in YAML files.

    Loads YAMLs with validation failures and creates prompts asking
    LLM to fix the specific errors while preserving all other content.
    """

    def __init__(self, base_dir: Path):
        """Initialize with base directory."""
        super().__init__(base_dir)
        self.header_manager = HeaderManager(base_dir)

    def get_batch_type(self) -> str:
        return "validation_fix"

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
                with open(validation_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                # Extract validation type from filename
                validation_type = validation_file.stem.replace('_', ' ').title()

                # Process failed items
                failed_items = report.get('failed', [])
                for item in failed_items:
                    filename = item['item']
                    reason = item['reason']

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
            with open(yaml_path, 'r', encoding='utf-8') as f:
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
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not load template {template_path}: {e}")
            return None

    def create_fix_prompt(self, yaml_content: str, errors: List[str],
                         template_content: str, template_type: str) -> str:
        """
        Create fix prompt by assembling content (without headers), errors, and template.

        Args:
            yaml_content: Original YAML content
            errors: List of error messages for this file
            template_content: Template YAML content for reference
            template_type: Type of template (parameter_metadata, test_statistic, quick_estimate)

        Returns:
            Complete fix prompt
        """
        # Parse YAML to detect schema version and strip headers
        data = yaml.safe_load(yaml_content)
        schema_version = data.get('schema_version', 'v1')

        # Strip headers from original YAML (LLM should only see content)
        headers, content_dict = self.header_manager.strip_headers_from_yaml_string(
            yaml_content, template_type, schema_version
        )

        # Convert content dict back to YAML string
        content_yaml = yaml.dump(content_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Use the validation fix prompt template
        prompt_path = self.base_dir / "prompts" / "validation_fix_prompt.md"

        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()

            # Format errors
            formatted_errors = "\n".join(f"- {error}" for error in errors)

            # Fill placeholders (use content YAML without headers)
            prompt = template.replace("{{YAML_CONTENT}}", content_yaml)
            prompt = prompt.replace("{{VALIDATION_ERRORS}}", formatted_errors)
            prompt = prompt.replace("{{TEMPLATE_CONTENT}}", template_content)

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

    def process(self, validation_dir: Path, yaml_dir: Path,
                template_path: Path) -> List[Dict[str, Any]]:
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

            # Create custom ID from filename
            file_stem = yaml_path.stem
            custom_id = f"fix_{file_stem}"

            request = self.create_request(
                custom_id,
                prompt,
                reasoning_effort="high"
            )

            requests.append(request)
            print(f"  Created fix request for {filename} ({len(errors)} error(s))")

        return requests

    def _detect_template_type_from_path(self, template_path: Path) -> str:
        """
        Detect template type from file path.

        Args:
            template_path: Path to template file

        Returns:
            Template type string (parameter_metadata, test_statistic, quick_estimate)
        """
        name = template_path.name.lower()

        if 'parameter' in name:
            return 'parameter_metadata'
        elif 'test_stat' in name:
            return 'test_statistic'
        elif 'quick' in name:
            return 'quick_estimate'
        else:
            # Default to parameter_metadata
            return 'parameter_metadata'
