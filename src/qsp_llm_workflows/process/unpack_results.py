#!/usr/bin/env python3
"""
Simplified unpacking of extraction results to YAML files.

Handles: parameter extraction, test statistics.
Generates derivation IDs and filenames deterministically.
"""

import sys
import json
import re
import csv
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
from ruamel.yaml.comments import CommentedMap


# Threshold for converting strings to block scalars
BLOCK_SCALAR_THRESHOLD = 80


def extract_json_from_content(content: str) -> Optional[str]:
    """Extract JSON from structured output response."""
    # With structured outputs, content is already valid JSON
    try:
        json.loads(content)
        return content
    except Exception:
        return None


def load_metadata(input_csv: Path, workflow_type: str) -> Dict:
    """Load metadata from input CSV."""
    metadata = {}

    if not input_csv or not input_csv.exists():
        return metadata

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if workflow_type == "test_statistic":
                key = row["test_statistic_id"]
                metadata[key] = {
                    "test_statistic_id": key,
                    "cancer_type": row["cancer_type"],
                    "model_context": row.get("model_context", ""),
                    "scenario_context": row.get("scenario_context", ""),
                    "required_species": row.get("required_species", ""),
                    "derived_species_description": row.get("derived_species_description", ""),
                }
            elif workflow_type == "calibration_target":
                key = row["calibration_target_id"]
                metadata[key] = {
                    "calibration_target_id": key,
                    "cancer_type": row["cancer_type"],
                    "model_context": row.get("model_context", ""),
                    "observable_description": row.get("observable_description", ""),
                }
            elif workflow_type == "isolated_system_target":
                key = row["target_id"]
                metadata[key] = {
                    "target_id": key,
                    "cancer_type": row["cancer_type"],
                    "observable_description": row.get("observable_description", ""),
                    "model_species": row.get("model_species", ""),
                    "model_indication": row.get("model_indication", ""),
                    "model_compartment": row.get("model_compartment", ""),
                    "model_system": row.get("model_system", ""),
                }
            else:  # parameter
                key = (row["cancer_type"], row["parameter_name"])
                metadata[key] = {
                    "parameter_name": row["parameter_name"],
                    "parameter_units": row.get("parameter_units", ""),
                    "parameter_definition": row.get("parameter_description", ""),
                    "cancer_type": row["cancer_type"],
                    "model_context": row.get("model_context", ""),
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


def generate_derivation_id(param_name: str, cancer_type: str, deriv_num: int) -> str:
    """Generate derivation_id: {param}_{cancer}_deriv{num}"""
    return f"{param_name}_{cancer_type}_deriv{deriv_num:03d}"


def add_footer_fields(json_data: dict, metadata: dict, workflow_type: str) -> dict:
    """Add footer fields to JSON data based on workflow type."""
    if workflow_type == "calibration_target":
        # Calibration targets have their own footer structure
        json_data["calibration_target_id"] = metadata["calibration_target_id"]
        json_data["cancer_type"] = metadata["cancer_type"]

        # Add tags
        tags = ["ai-generated"]
        json_data["tags"] = tags

        # Parse model_context if it's a JSON string
        model_context = metadata.get("model_context", "")
        if model_context:
            try:
                json_data["model_context"] = json.loads(model_context)
            except Exception:
                json_data["model_context"] = model_context

    elif workflow_type == "isolated_system_target":
        # Isolated system targets - the LLM output already contains all data
        # Just add tags
        tags = ["ai-generated"]
        json_data["tags"] = tags

    elif workflow_type == "test_statistic":
        # Test statistics have different footer structure
        json_data["test_statistic_id"] = metadata["test_statistic_id"]
        json_data["cancer_type"] = metadata["cancer_type"]
        json_data["model_context"] = metadata.get("model_context", "")
        json_data["scenario_context"] = metadata.get("scenario_context", "")

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

        json_data["parameter_name"] = param_name
        json_data["parameter_units"] = metadata["parameter_units"]
        json_data["parameter_definition"] = metadata["parameter_definition"]
        json_data["cancer_type"] = cancer_type

        # Add tags
        tags = ["ai-generated"]
        json_data["tags"] = tags

        # Parse model_context if it's a JSON string
        model_context = metadata.get("model_context", "")
        if model_context:
            try:
                json_data["model_context"] = json.loads(model_context)
            except Exception:
                json_data["model_context"] = model_context

    return json_data


def _convert_long_strings_to_block(obj, threshold=BLOCK_SCALAR_THRESHOLD):
    """
    Recursively convert long strings to LiteralScalarString for block scalar formatting.

    Args:
        obj: The object to process (dict, list, or scalar)
        threshold: Strings longer than this become block scalars

    Returns:
        The processed object with long strings converted
    """
    if isinstance(obj, dict):
        return {k: _convert_long_strings_to_block(v, threshold) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_long_strings_to_block(item, threshold) for item in obj]
    elif isinstance(obj, str):
        # Convert to block scalar if string is long or contains newlines
        if len(obj) > threshold or "\n" in obj:
            # Ensure string ends with newline for clean block scalar
            text = obj if obj.endswith("\n") else obj + "\n"
            return LiteralScalarString(text)
        return obj
    else:
        return obj


def _sanitize_null_bytes(obj):
    """
    Recursively replace null bytes (\x00) with ** in all strings.

    LLM sometimes outputs null bytes instead of ^ when copying from PDFs with superscripts.
    Replace with ** which is Pint's exponentiation syntax (mm**2 instead of mm^2).
    """
    if isinstance(obj, str):
        return obj.replace("\x00", "**")
    elif isinstance(obj, dict):
        return {k: _sanitize_null_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_null_bytes(item) for item in obj]
    else:
        return obj


def _add_blank_lines_between_sections(data: dict) -> CommentedMap:
    """
    Convert dict to CommentedMap with blank lines between root-level sections.

    Adds a blank line before each root-level key (except the first) for readability.
    """
    cm = CommentedMap(data)

    # Add blank line before each key except the first
    keys = list(cm.keys())
    for i, key in enumerate(keys[1:], start=1):
        cm.yaml_set_comment_before_after_key(key, before="\n")

    return cm


def convert_to_yaml(json_data: dict) -> str:
    """Convert JSON to YAML with proper formatting using ruamel.yaml."""
    # Sanitize null bytes from LLM output
    json_data = _sanitize_null_bytes(json_data)

    # Convert long strings to block scalars
    processed_data = _convert_long_strings_to_block(json_data)

    # Add blank lines between root-level sections
    processed_data = _add_blank_lines_between_sections(processed_data)

    # Configure ruamel.yaml
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.allow_unicode = True
    yaml.width = 120  # Reasonable line width for readability
    # Best practice indent for readable YAML:
    # - mapping keys at 2-space indent
    # - sequence "-" at 2-space indent
    # - content after "-" on same line (offset=0 means no extra indent after dash)
    yaml.indent(mapping=2, sequence=2, offset=0)

    # Dump to string
    stream = io.StringIO()
    yaml.dump(processed_data, stream)
    return stream.getvalue()


def parse_custom_id(custom_id: str) -> Tuple[str, str, str]:
    """
    Parse custom_id to determine workflow type and extract identifiers.

    Returns: (workflow_type, cancer_type, identifier)
    """
    parts = custom_id.split("_")

    if parts[0] == "test" and parts[1] == "stat":
        # test_stat_TEST_STATISTIC_ID_INDEX
        return ("test_statistic", "", "_".join(parts[2:-1]))

    elif parts[0] == "cal" and parts[1] == "target":
        # cal_target_CALIBRATION_TARGET_ID_INDEX
        return ("calibration_target", "", "_".join(parts[2:-1]))

    elif parts[0] == "isolated" and parts[1] == "target":
        # isolated_target_TARGET_ID_INDEX
        return ("isolated_system_target", "", "_".join(parts[2:-1]))

    else:
        # CANCER_PARAMETER_INDEX (regular parameter extraction)
        return ("parameter", parts[0], "_".join(parts[1:-1]))


def unpack_single_result(
    result: dict,
    output_dir: Path,
    workflow_type: str,
    metadata: Dict,
    progress_callback: Optional[callable] = None,
) -> Optional[Path]:
    """
    Unpack a single result to YAML file immediately.

    Args:
        result: Single result dictionary from processor
        output_dir: Output directory for YAML files
        workflow_type: "parameter", "test_statistic", or "calibration_target"
        metadata: Pre-loaded metadata dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Path to created YAML file, or None if unpacking failed
    """
    custom_id = result["custom_id"]

    # Parse custom_id
    _, cancer_type, identifier = parse_custom_id(custom_id)

    # Get response content
    body = result["response"]["body"]

    # Check if this is response format (has "output" key) or direct format (direct JSON)
    if "output" in body:
        # Response format: extract from output[type=message].content[0].text
        try:
            output_items = body["output"]
            message_item = next(item for item in output_items if item.get("type") == "message")
            content = message_item["content"][0]["text"]

            # Extract JSON from markdown code fence
            json_content = extract_json_from_content(content)
            if not json_content:
                if progress_callback:
                    progress_callback(f"Error: No JSON found for {custom_id}")
                return None

            json_data = json.loads(json_content)
        except (KeyError, StopIteration, json.JSONDecodeError) as e:
            if progress_callback:
                progress_callback(f"Error: Could not extract content for {custom_id}: {e}")
            return None
    else:
        # Immediate format: extract from output_parsed field
        try:
            if "output_parsed" in body:
                json_data = body["output_parsed"]
            else:
                # Fallback for older format without output_parsed wrapper
                json_data = body
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: Invalid immediate format for {custom_id}: {e}")
            return None

    # Look up metadata for extraction workflows
    if workflow_type == "test_statistic":
        meta = metadata.get(identifier)
        if meta:
            cancer_type = meta["cancer_type"]
    elif workflow_type == "calibration_target":
        meta = metadata.get(identifier)
        if meta:
            cancer_type = meta["cancer_type"]
    elif workflow_type == "isolated_system_target":
        meta = metadata.get(identifier)
        if meta:
            cancer_type = meta["cancer_type"]
    else:
        meta = metadata.get((cancer_type, identifier))

    if not meta:
        if progress_callback:
            progress_callback(f"Warning: No metadata for {custom_id}, skipping")
        return None

    # Add footer fields
    json_data = add_footer_fields(json_data, meta, workflow_type)

    # Generate filename with derivation numbering
    if workflow_type == "test_statistic":
        # test_stat_id_cancer_deriv001.yaml
        base = f"{identifier}_{cancer_type}"
        deriv_num = find_next_derivation_number(output_dir, base)
        filename = f"{base}_deriv{deriv_num:03d}.yaml"

    elif workflow_type == "calibration_target":
        # cal_target_id_cancer_deriv001.yaml
        base = f"{identifier}_{cancer_type}"
        deriv_num = find_next_derivation_number(output_dir, base)
        filename = f"{base}_deriv{deriv_num:03d}.yaml"

    elif workflow_type == "isolated_system_target":
        # target_id_cancer_deriv001.yaml
        base = f"{identifier}_{cancer_type}"
        deriv_num = find_next_derivation_number(output_dir, base)
        filename = f"{base}_deriv{deriv_num:03d}.yaml"

    else:  # parameter
        # param_cancer_deriv001.yaml
        base = f"{identifier}_{cancer_type}"
        deriv_num = find_next_derivation_number(output_dir, base)

        # Generate derivation_id and add to JSON
        derivation_id = generate_derivation_id(identifier, cancer_type, deriv_num)
        json_data["derivation_id"] = derivation_id
        json_data["derivation_timestamp"] = datetime.now().isoformat()

        filename = f"{derivation_id}.yaml"

    # Convert to YAML
    yaml_content = convert_to_yaml(json_data)

    # Write file
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    if progress_callback:
        progress_callback(f"Saved: {filename}")

    return output_path


def process_results(results_file: Path, output_dir: Path, input_csv: Path = None):
    """Process extraction results and write YAML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect workflow type from first line
    with open(results_file, "r") as f:
        first_line = f.readline()
        first_result = json.loads(first_line)
        custom_id = first_result["custom_id"]
        workflow_type, _, _ = parse_custom_id(custom_id)

    # Load metadata
    metadata = load_metadata(input_csv, workflow_type)
    if not metadata:
        print(f"Warning: No metadata loaded from {input_csv}")

    # Process each result
    with open(results_file, "r") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]

            # Parse custom_id
            workflow_type, cancer_type, identifier = parse_custom_id(custom_id)

            # Get response content - handle result format
            body = result["response"]["body"]

            # Check if this is response format (has "output" key) or direct format (direct JSON)
            if "output" in body:
                # Response format: extract from output[type=message].content[0].text
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
                        f"Error: Could not extract content from response format for {custom_id}: {e}"
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

            # Look up metadata for extraction workflows
            if workflow_type == "test_statistic":
                meta = metadata.get(identifier)
                if meta:
                    cancer_type = meta["cancer_type"]
            elif workflow_type == "calibration_target":
                meta = metadata.get(identifier)
                if meta:
                    cancer_type = meta["cancer_type"]
            elif workflow_type == "isolated_system_target":
                meta = metadata.get(identifier)
                if meta:
                    cancer_type = meta["cancer_type"]
            else:
                meta = metadata.get((cancer_type, identifier))

            if not meta:
                print(f"Warning: No metadata for {custom_id}, skipping")
                continue

            # Add footer fields
            json_data = add_footer_fields(json_data, meta, workflow_type)

            # Generate filename with derivation numbering
            if workflow_type == "test_statistic":
                # test_stat_id_cancer_deriv001.yaml
                base = f"{identifier}_{cancer_type}"
                deriv_num = find_next_derivation_number(output_dir, base)
                filename = f"{base}_deriv{deriv_num:03d}.yaml"

            elif workflow_type == "calibration_target":
                # cal_target_id_cancer_deriv001.yaml
                base = f"{identifier}_{cancer_type}"
                deriv_num = find_next_derivation_number(output_dir, base)
                filename = f"{base}_deriv{deriv_num:03d}.yaml"

            elif workflow_type == "isolated_system_target":
                # target_id_cancer_deriv001.yaml
                base = f"{identifier}_{cancer_type}"
                deriv_num = find_next_derivation_number(output_dir, base)
                filename = f"{base}_deriv{deriv_num:03d}.yaml"

            else:  # parameter
                # param_cancer_deriv001.yaml
                base = f"{identifier}_{cancer_type}"
                deriv_num = find_next_derivation_number(output_dir, base)

                # Generate derivation_id and add to JSON
                derivation_id = generate_derivation_id(identifier, cancer_type, deriv_num)
                json_data["derivation_id"] = derivation_id
                json_data["derivation_timestamp"] = datetime.now().isoformat()

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
        print("  results.jsonl - Results file from extraction")
        print("  output_dir    - Output directory for YAML files")
        print("  input.csv     - Input CSV with metadata (optional)")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    input_csv = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

    process_results(results_file, output_dir, input_csv)


if __name__ == "__main__":
    main()
