#!/usr/bin/env python3
"""
Simplified unpacking of batch results to YAML files.

Handles: parameter extraction, test statistics.
Generates derivation IDs and filenames deterministically.
"""

import sys
import json
import re
import csv
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic
from qsp_llm_workflows.core.header_utils import HeaderManager


def extract_json_from_content(content: str) -> Optional[str]:
    """Extract JSON from structured output response."""
    # With structured outputs, content is already valid JSON
    try:
        json.loads(content)
        return content
    except Exception:
        return None


def load_metadata(input_csv: Path, batch_type: str) -> Dict:
    """Load metadata from input CSV."""
    metadata = {}

    if not input_csv or not input_csv.exists():
        return metadata

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if batch_type == "test_statistic":
                key = row["test_statistic_id"]
                metadata[key] = {
                    "test_statistic_id": key,
                    "cancer_type": row["cancer_type"],
                    "context_hash": row.get("context_hash", ""),
                    "model_context": row.get("model_context", ""),
                    "scenario_context": row.get("scenario_context", ""),
                    "required_species": row.get("required_species", ""),
                    "derived_species_description": row.get("derived_species_description", ""),
                }
            else:  # parameter
                key = (row["cancer_type"], row["parameter_name"])
                metadata[key] = {
                    "parameter_name": row["parameter_name"],
                    "parameter_units": row.get("parameter_units", ""),
                    "parameter_definition": row.get("parameter_description", ""),
                    "cancer_type": row["cancer_type"],
                    "model_context": row.get("model_context", ""),
                    "context_hash": row.get("definition_hash", ""),
                }

    return metadata


def find_next_derivation_number(directory: Path, base_pattern: str) -> int:
    """Find next available derivation number for base pattern."""
    existing = list(directory.glob(f"{base_pattern}_deriv*.yaml"))

    if not existing:
        return 1

    max_num = 0
    for file in existing:
        match = re.search(r"_deriv(\d+)\.yaml$", file.name)
        if match:
            max_num = max(max_num, int(match.group(1)))

    return max_num + 1


def generate_derivation_id(
    param_name: str, cancer_type: str, context_hash: str, deriv_num: int
) -> str:
    """Generate derivation_id: {param}_{cancer}_{context_hash}_deriv{num}"""
    return f"{param_name}_{cancer_type}_{context_hash}_deriv{deriv_num:03d}"


def add_header_fields(json_data: dict, metadata: dict, batch_type: str) -> dict:
    """Add header fields to JSON data based on batch type."""
    if batch_type == "test_statistic":
        # Test statistics have different header structure
        json_data["test_statistic_id"] = metadata["test_statistic_id"]
        json_data["cancer_type"] = metadata["cancer_type"]
        json_data["context_hash"] = metadata["context_hash"]
        json_data["model_context"] = metadata.get("model_context", "")
        json_data["scenario_context"] = metadata.get("scenario_context", "")
        json_data["schema_version"] = "v2"

        # Parse required_species from comma-separated string to list
        required_species_str = metadata.get("required_species", "")
        if not required_species_str.strip():
            raise ValueError(
                f"required_species is required for test statistic "
                f"'{metadata['test_statistic_id']}'"
            )
        json_data["required_species"] = [
            s.strip() for s in required_species_str.split(",") if s.strip()
        ]

        derived_species_desc = metadata.get("derived_species_description", "")
        if not derived_species_desc.strip():
            raise ValueError(
                f"derived_species_description is required for test statistic "
                f"'{metadata['test_statistic_id']}'"
            )
        json_data["derived_species_description"] = derived_species_desc

    else:  # parameter
        param_name = metadata["parameter_name"]
        cancer_type = metadata["cancer_type"]
        context_hash = metadata["context_hash"]

        json_data["parameter_name"] = param_name
        json_data["parameter_units"] = metadata["parameter_units"]
        json_data["parameter_definition"] = metadata["parameter_definition"]
        json_data["cancer_type"] = cancer_type

        # Add tags
        tags = ["ai-generated"]
        json_data["tags"] = tags

        json_data["context_hash"] = context_hash
        json_data["schema_version"] = "v3"

        # Parse model_context if it's a JSON string
        model_context = metadata.get("model_context", "")
        if model_context:
            try:
                json_data["model_context"] = json.loads(model_context)
            except Exception:
                json_data["model_context"] = model_context

    return json_data


def move_field_to_top(data: dict, field_name: str) -> dict:
    """Move a field to the top of the dictionary."""
    if field_name in data:
        value = data.pop(field_name)
        return {field_name: value, **data}
    return data


def convert_to_yaml(json_data: dict) -> str:
    """Convert JSON to YAML with proper formatting."""

    # Custom representer for multi-line strings
    def str_representer(dumper, data):
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)

    return yaml.dump(
        json_data,
        Dumper=yaml.SafeDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=1000,
    )


def parse_custom_id(custom_id: str) -> Tuple[str, str, str]:
    """
    Parse custom_id to determine batch type and extract identifiers.

    Returns: (batch_type, cancer_type, identifier)
    """
    parts = custom_id.split("_")

    if parts[0] == "fix":
        # fix_FILENAME (validation fix batch)
        # Return the original filename as identifier
        return ("validation_fix", "", "_".join(parts[1:]))

    elif parts[0] == "test" and parts[1] == "stat":
        # test_stat_TEST_STATISTIC_ID_INDEX
        return ("test_statistic", "", "_".join(parts[2:-1]))

    else:
        # CANCER_PARAMETER_INDEX (regular parameter extraction)
        return ("parameter", parts[0], "_".join(parts[1:-1]))


def process_results(results_file: Path, output_dir: Path, input_csv: Path = None):
    """Process batch results and write YAML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect batch type from first line
    with open(results_file, "r") as f:
        first_line = f.readline()
        first_result = json.loads(first_line)
        custom_id = first_result["custom_id"]
        batch_type, _, _ = parse_custom_id(custom_id)

    # Load metadata (skip for validation fixes)
    if batch_type == "validation_fix":
        metadata = {}
    else:
        metadata = load_metadata(input_csv, batch_type)
        if not metadata:
            print(f"Warning: No metadata loaded from {input_csv}")

    # Process each result
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]

            # Parse custom_id
            batch_type, cancer_type, identifier = parse_custom_id(custom_id)

            # Get response content - handle both batch and immediate formats
            body = result["response"]["body"]

            # Check if this is batch format (has "output" key) or immediate format (direct JSON)
            if "output" in body:
                # Batch format: extract from output[type=message].content[0].text
                try:
                    output_items = body["output"]
                    message_item = next(
                        item for item in output_items if item.get("type") == "message"
                    )
                    content = message_item["content"][0]["text"]

                    # Extract JSON from markdown code fence
                    json_content = extract_json_from_content(content)
                    if not json_content:
                        print(f"Error: No JSON found for {custom_id}")
                        continue

                    json_data = json.loads(json_content)
                except (KeyError, StopIteration, json.JSONDecodeError) as e:
                    print(
                        f"Error: Could not extract content from batch format for {custom_id}: {e}"
                    )
                    continue
            else:
                # Immediate format: extract from output_parsed field
                try:
                    if "output_parsed" in body:
                        json_data = body["output_parsed"]
                    else:
                        # Fallback for older format without output_parsed wrapper
                        json_data = body
                except Exception as e:
                    print(f"Error: Invalid immediate format for {custom_id}: {e}")
                    continue

            # Handle validation fixes specially
            if batch_type == "validation_fix":
                # For validation fixes, load original YAML to get headers
                original_filename = f"{identifier}.yaml"
                original_path = output_dir / original_filename

                if not original_path.exists():
                    print(f"Error: Original file not found for fixing: {original_path}")
                    continue

                # Load original YAML to determine model type
                with open(original_path, "r", encoding="utf-8") as f:
                    original_yaml = yaml.safe_load(f.read())

                # Determine model class from file structure
                if "parameter_name" in original_yaml:
                    model_class = ParameterMetadata
                elif "test_statistic_id" in original_yaml:
                    model_class = TestStatistic
                else:
                    print(f"Error: Could not determine model type for {original_filename}")
                    continue

                # Use HeaderManager to extract headers from original file
                header_manager = HeaderManager()
                headers = header_manager.extract_headers_from_yaml(original_path, model_class)

                # Merge headers with fixed content (headers take precedence)
                json_data = {**json_data, **headers.model_dump()}

                # Move header fields to top for readability (reverse order)
                header_field_names = list(type(headers).model_fields.keys())
                for field in reversed(header_field_names):
                    if field in json_data:
                        json_data = move_field_to_top(json_data, field)

                filename = original_filename

            else:
                # Look up metadata for extraction workflows
                if batch_type == "test_statistic":
                    meta = metadata.get(identifier)
                    if meta:
                        cancer_type = meta["cancer_type"]
                else:
                    meta = metadata.get((cancer_type, identifier))

                if not meta:
                    print(f"Warning: No metadata for {custom_id}, skipping")
                    continue

                # Add header fields
                json_data = add_header_fields(json_data, meta, batch_type)

            # Generate filename with derivation numbering (for extraction workflows only)
            if batch_type != "validation_fix":
                if batch_type == "test_statistic":
                    # test_stat_id_cancer_hash_deriv001.yaml
                    base = f"{identifier}_{cancer_type}_{meta['context_hash']}"
                    deriv_num = find_next_derivation_number(output_dir, base)
                    filename = f"{base}_deriv{deriv_num:03d}.yaml"

                    # Move header fields to top (reverse order since we prepend)
                    json_data = move_field_to_top(json_data, "derived_species_description")
                    json_data = move_field_to_top(json_data, "required_species")
                    json_data = move_field_to_top(json_data, "scenario_context")
                    json_data = move_field_to_top(json_data, "model_context")
                    json_data = move_field_to_top(json_data, "context_hash")
                    json_data = move_field_to_top(json_data, "cancer_type")
                    json_data = move_field_to_top(json_data, "test_statistic_id")
                    json_data = move_field_to_top(json_data, "schema_version")

                else:  # parameter
                    # param_cancer_hash_deriv001.yaml (v3 schema)
                    base = f"{identifier}_{cancer_type}_{meta['context_hash']}"
                    deriv_num = find_next_derivation_number(output_dir, base)

                    # Generate derivation_id and add to JSON
                    derivation_id = generate_derivation_id(
                        identifier, cancer_type, meta["context_hash"], deriv_num
                    )
                    json_data["derivation_id"] = derivation_id
                    json_data["derivation_timestamp"] = datetime.now().isoformat()

                    # Move all header fields to top in correct order (reverse order since we prepend)
                    json_data = move_field_to_top(json_data, "context_hash")
                    json_data = move_field_to_top(json_data, "model_context")
                    json_data = move_field_to_top(json_data, "derivation_timestamp")
                    json_data = move_field_to_top(json_data, "derivation_id")
                    json_data = move_field_to_top(json_data, "tags")
                    json_data = move_field_to_top(json_data, "cancer_type")
                    json_data = move_field_to_top(json_data, "parameter_definition")
                    json_data = move_field_to_top(json_data, "parameter_units")
                    json_data = move_field_to_top(json_data, "parameter_name")
                    json_data = move_field_to_top(json_data, "schema_version")

                    filename = f"{derivation_id}.yaml"

            # Convert to YAML
            yaml_content = convert_to_yaml(json_data)

            # Write file
            output_path = output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            print(f"Saved: {filename}")


def main():
    if len(sys.argv) < 3:
        print("Usage: unpack_results.py <results.jsonl> <output_dir> [input.csv]")
        print()
        print("  results.jsonl - Batch results file")
        print("  output_dir    - Output directory for YAML files")
        print(
            "  input.csv     - Input CSV with metadata (optional, not needed for validation fixes)"
        )
        sys.exit(1)

    results_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    input_csv = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

    process_results(results_file, output_dir, input_csv)


if __name__ == "__main__":
    main()
