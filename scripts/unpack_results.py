#!/usr/bin/env python3
"""
Unpack batch results to parameter folders.

Adds header fields from input CSV to unpacked YAML files.
"""

import json
import re
import sys
import yaml
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

def extract_yaml_from_content(content):
    """Extract YAML section from content."""
    # First try to find YAML code block
    yaml_match = re.search(r'```yaml\n(.*?)\n```', content, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1)
    
    # If no code block found, try parsing the entire content as YAML
    # This handles cases where LLM forgot the ```yaml tags but generated valid YAML
    try:
        # Test if the entire content can be parsed as YAML
        yaml.safe_load(content.strip())
        return content.strip()
    except yaml.YAMLError:
        pass
    
    # Legacy fallback: check if content starts with parameter_name
    # (kept for backwards compatibility, though above should catch this too)
    if re.match(r'^\s*parameter_name\s*:', content, re.MULTILINE):
        return content.strip()
    
    return None

def extract_first_source_tag(yaml_content):
    """Extract the first source tag from YAML content."""
    try:
        data = yaml.safe_load(yaml_content)
        if 'sources' in data and isinstance(data['sources'], dict):
            # Get first source key
            return list(data['sources'].keys())[0]
    except Exception:
        pass
    return None

def get_unique_filename(base_path):
    """Get a unique filename by adding v2, v3, etc. if file exists."""
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 2
    while True:
        new_name = f"{stem}_v{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def load_parameter_metadata(input_csv: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load parameter metadata from input CSV.

    Returns dict keyed by (cancer_type, parameter_name) with metadata values.
    """
    metadata = {}

    if not input_csv or not input_csv.exists():
        return metadata

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cancer_type = row['cancer_type']
            parameter_name = row['parameter_name']

            key = (cancer_type, parameter_name)
            metadata[key] = {
                'parameter_name': parameter_name,
                'parameter_units': row.get('parameter_units', ''),
                'parameter_definition': row.get('parameter_description', ''),
                'cancer_type': cancer_type,
                'tags': [],  # Could be populated if available in CSV
                'model_context': row.get('model_context', ''),
                'context_hash': row.get('definition_hash', '')
            }

    return metadata

def prepend_header_fields(yaml_content: str, header_data: Dict) -> str:
    """
    Prepend header fields to LLM-generated YAML content.

    Args:
        yaml_content: YAML from LLM response
        header_data: Header fields to prepend

    Returns:
        Complete YAML with header fields
    """
    # Parse the LLM YAML
    try:
        llm_data = yaml.safe_load(yaml_content)
    except Exception as e:
        print(f"Warning: Could not parse LLM YAML: {e}")
        return yaml_content

    # Build complete document with header fields first
    complete_data = {}

    # Add header section comment
    header_comment = "# PARAMETER DEFINITION (from model context)\n"
    header_comment += "# " + "=" * 76 + "\n"

    # Add header fields in order
    complete_data['parameter_name'] = header_data['parameter_name']
    complete_data['parameter_units'] = header_data['parameter_units']
    complete_data['parameter_definition'] = header_data['parameter_definition']
    complete_data['cancer_type'] = header_data['cancer_type']
    complete_data['tags'] = header_data['tags']

    # Parse and add model_context (it's JSON in the CSV)
    if header_data['model_context']:
        try:
            model_context = json.loads(header_data['model_context'])
            complete_data['model_context'] = model_context
        except json.JSONDecodeError:
            complete_data['model_context'] = header_data['model_context']

    complete_data['context_hash'] = header_data['context_hash']

    # Add all LLM-generated fields
    complete_data.update(llm_data)

    # Convert to YAML string with header comment
    yaml_str = yaml.dump(complete_data,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False)

    return header_comment + yaml_str

def main():
    if len(sys.argv) < 3:
        print("Usage: unpack_results.py results.jsonl target_project_dir [input_csv]")
        print("       results.jsonl: Batch results file")
        print("       target_project_dir: path to the QSP project directory")
        print("       input_csv: (optional) Input CSV with parameter metadata for header fields")
        sys.exit(1)

    results_file = sys.argv[1]
    target_project_dir = sys.argv[2]
    input_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    target_project_path = Path(target_project_dir).resolve()

    # Load parameter metadata if input CSV provided
    param_metadata = load_parameter_metadata(input_csv) if input_csv else {}
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            
            custom_id = result["custom_id"]
            response = result["response"]
            
            if response["status_code"] != 200:
                continue
            
            # GPT-5 response structure: body.output[1].content[0].text
            content = response["body"]["output"][1]["content"][0]["text"]
            
            # Parse custom_id and determine directory structure
            parts = custom_id.split('_')

            # Handle different batch types based on custom_id prefix
            if parts[0] == "defn":
                # Parameter definition: defn_CANCER_TYPE_PARAMETER_NAME_INDEX
                cancer_type = parts[1]
                parameter_name = '_'.join(parts[2:-1])  # Everything between cancer_type and index
                # Put in parameter-definitions directory
                param_dir = target_project_path / "parameter-definitions" / cancer_type / parameter_name
                param_dir.mkdir(parents=True, exist_ok=True)
                filename_default = "definition.yaml"
                use_flat_structure = False
            elif parts[0] == "quick":
                # Quick estimate: quick_CANCER_TYPE_PARAMETER_NAME_INDEX
                cancer_type = parts[1]
                parameter_name = '_'.join(parts[2:-1])  # Everything between cancer_type and index
                # Put in to-review/quick-estimates directory
                param_dir = target_project_path / "to-review" / "quick-estimates" / cancer_type / parameter_name
                param_dir.mkdir(parents=True, exist_ok=True)
                filename_default = "quick_estimate.yaml"
                use_flat_structure = False
            else:
                # Regular parameter extraction: CANCER_TYPE_PARAMETER_NAME_INDEX
                # Save directly to parameter_estimates with flat structure
                cancer_type = parts[0]
                parameter_name = '_'.join(parts[1:-1])  # Everything except first and last
                param_dir = target_project_path  # Flat structure at parameter_estimates level
                filename_default = None  # Will be computed from metadata
                use_flat_structure = True

            # Extract YAML content
            yaml_content = extract_yaml_from_content(content)
            if yaml_content:
                # Prepend header fields if metadata available
                key = (cancer_type, parameter_name)
                if key in param_metadata:
                    yaml_content = prepend_header_fields(yaml_content, param_metadata[key])
                    definition_hash = param_metadata[key].get('context_hash', '')
                else:
                    definition_hash = ''

                # Determine filename based on structure type
                if use_flat_structure:
                    # New format: {param_name}_{author_year}_{definition_hash}.yaml
                    author_year = extract_first_source_tag(yaml_content)
                    if author_year and definition_hash:
                        yaml_filename = f"{parameter_name}_{author_year}_{definition_hash}.yaml"
                    else:
                        # Fallback if missing components
                        yaml_filename = f"{parameter_name}_unknown.yaml"
                else:
                    # Legacy format: use first source tag or default
                    first_source = extract_first_source_tag(yaml_content)
                    if first_source:
                        yaml_filename = f"{first_source}.yaml"
                    else:
                        yaml_filename = filename_default

                # Get unique filename
                yaml_path = get_unique_filename(param_dir / yaml_filename)

                # Save YAML file
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)

                if use_flat_structure:
                    print(f"Saved: {yaml_path.name}")
                else:
                    print(f"Saved: {cancer_type}/{parameter_name}/{yaml_path.name}")
            else:
                print(f"No YAML found: {cancer_type}/{parameter_name}")

if __name__ == "__main__":
    main()