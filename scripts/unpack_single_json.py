#!/usr/bin/env python3
"""
Unpack a single JSON file (e.g., from chat interface) to YAML.

Automatically extracts metadata from JSON and saves to qsp-metadata-storage
with proper filename format.
"""

import json
import sys
import yaml
from pathlib import Path


def extract_first_source_tag(data: dict) -> str:
    """Extract the first source tag from sources field."""
    try:
        if 'sources' in data and isinstance(data['sources'], dict):
            return list(data['sources'].keys())[0]
    except Exception:
        pass
    return None


def get_unique_filename(base_path: Path) -> Path:
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


def convert_json_to_yaml(json_file: Path, output_dir: Path = None):
    """
    Convert a JSON file to YAML and save to qsp-metadata-storage.

    Extracts metadata from JSON to construct filename automatically.

    Args:
        json_file: Path to input JSON file
        output_dir: Optional output directory (defaults to ../qsp-metadata-storage/parameter_estimates)
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract required fields from JSON
    parameter_name = data.get('parameter_name')
    context_hash = data.get('context_hash')

    if not parameter_name:
        print("Error: JSON must contain 'parameter_name' field")
        sys.exit(1)

    if not context_hash:
        print("Warning: No 'context_hash' field found, using 'unknown'")
        context_hash = 'unknown'

    # Extract first source tag for filename
    author_year = extract_first_source_tag(data)
    if not author_year:
        print("Warning: No sources found, using 'unknown' for author_year")
        author_year = 'unknown'

    # Add "ai-generated" tag if not present
    tags = data.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    if 'ai-generated' not in tags:
        tags.append('ai-generated')
    data['tags'] = tags

    # Add header comment
    header_comment = "# PARAMETER DEFINITION (from model context)\n"
    header_comment += "# " + "=" * 76 + "\n"

    # Convert to YAML
    yaml_str = yaml.dump(data,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False)

    final_output = header_comment + yaml_str

    # Determine output directory
    if output_dir is None:
        # Default to qsp-metadata-storage/parameter_estimates
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent.parent / "qsp-metadata-storage" / "parameter_estimates"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct filename: {param_name}_{author_year}_{context_hash}.yaml
    filename = f"{parameter_name}_{author_year}_{context_hash}.yaml"
    output_file = get_unique_filename(output_dir / filename)

    # Write YAML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)

    print(f"Saved: {output_file}")
    print(f"  Parameter: {parameter_name}")
    print(f"  Source: {author_year}")
    print(f"  Hash: {context_hash}")


def main():
    if len(sys.argv) < 2:
        print("Usage: unpack_single_json.py input.json [output_dir]")
        print("")
        print("Extracts metadata from JSON and saves to qsp-metadata-storage")
        print("with filename format: {param_name}_{author_year}_{context_hash}.yaml")
        print("")
        print("The JSON must contain:")
        print("  - parameter_name (required)")
        print("  - context_hash (optional, defaults to 'unknown')")
        print("  - sources (optional, first key used for author_year)")
        print("")
        print("Examples:")
        print("  # Save to default location (../qsp-metadata-storage/parameter_estimates)")
        print("  python scripts/unpack_single_json.py chat_response.json")
        print("")
        print("  # Save to custom directory")
        print("  python scripts/unpack_single_json.py chat_response.json /path/to/output")
        sys.exit(1)

    json_file = Path(sys.argv[1])

    if not json_file.exists():
        print(f"Error: {json_file} not found")
        sys.exit(1)

    # Optional output directory
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Convert
    convert_json_to_yaml(json_file, output_dir)


if __name__ == "__main__":
    main()
