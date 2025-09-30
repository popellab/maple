#!/usr/bin/env python3
"""
Unpack a single JSON file (e.g., from chat interface) to YAML.

This is a convenience script for manually extracted parameters.
"""

import json
import sys
import yaml
from pathlib import Path
from typing import Dict


def convert_json_to_yaml(json_file: Path, output_file: Path, header_data: Dict = None):
    """
    Convert a JSON file to YAML with optional header fields.

    Args:
        json_file: Path to input JSON file
        output_file: Path to output YAML file
        header_data: Optional dict with header fields (parameter_name, units, etc.)
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If header data provided, prepend it
    if header_data:
        complete_data = {}

        # Add header comment
        header_comment = "# PARAMETER DEFINITION (from model context)\n"
        header_comment += "# " + "=" * 76 + "\n"

        # Add header fields if provided
        if 'parameter_name' in header_data:
            complete_data['parameter_name'] = header_data['parameter_name']
        if 'parameter_units' in header_data:
            complete_data['parameter_units'] = header_data['parameter_units']
        if 'parameter_definition' in header_data:
            complete_data['parameter_definition'] = header_data['parameter_definition']
        if 'cancer_type' in header_data:
            complete_data['cancer_type'] = header_data['cancer_type']

        # Add tags with "ai-generated" marker
        tags = header_data.get('tags', [])
        if not isinstance(tags, list):
            tags = []
        if 'ai-generated' not in tags:
            tags.append('ai-generated')
        complete_data['tags'] = tags

        if 'model_context' in header_data:
            complete_data['model_context'] = header_data['model_context']
        if 'context_hash' in header_data:
            complete_data['context_hash'] = header_data['context_hash']

        # Add LLM-generated fields
        complete_data.update(data)

        # Convert to YAML
        yaml_str = yaml.dump(complete_data,
                            default_flow_style=False,
                            allow_unicode=True,
                            sort_keys=False)

        final_output = header_comment + yaml_str
    else:
        # No header data, just convert JSON to YAML
        final_output = yaml.dump(data,
                                default_flow_style=False,
                                allow_unicode=True,
                                sort_keys=False)

    # Write YAML file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)

    print(f"Converted {json_file} -> {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: unpack_single_json.py input.json output.yaml [options]")
        print("")
        print("Options:")
        print("  --param-name NAME        Parameter name")
        print("  --param-units UNITS      Parameter units")
        print("  --param-def DEFINITION   Parameter definition")
        print("  --cancer-type TYPE       Cancer type")
        print("  --context-hash HASH      Context hash")
        print("")
        print("Example:")
        print("  python scripts/unpack_single_json.py chat_response.json output.yaml \\")
        print("    --param-name k_fib_death \\")
        print("    --param-units '1/day' \\")
        print("    --cancer-type PDAC")
        sys.exit(1)

    json_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not json_file.exists():
        print(f"Error: {json_file} not found")
        sys.exit(1)

    # Parse optional header arguments
    header_data = {}
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--param-name' and i + 1 < len(sys.argv):
            header_data['parameter_name'] = sys.argv[i + 1]
            i += 2
        elif arg == '--param-units' and i + 1 < len(sys.argv):
            header_data['parameter_units'] = sys.argv[i + 1]
            i += 2
        elif arg == '--param-def' and i + 1 < len(sys.argv):
            header_data['parameter_definition'] = sys.argv[i + 1]
            i += 2
        elif arg == '--cancer-type' and i + 1 < len(sys.argv):
            header_data['cancer_type'] = sys.argv[i + 1]
            i += 2
        elif arg == '--context-hash' and i + 1 < len(sys.argv):
            header_data['context_hash'] = sys.argv[i + 1]
            i += 2
        else:
            print(f"Warning: Unknown argument {arg}")
            i += 1

    # Convert
    convert_json_to_yaml(json_file, output_file, header_data if header_data else None)


if __name__ == "__main__":
    main()
