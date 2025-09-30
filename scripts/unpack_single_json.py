#!/usr/bin/env python3
"""
Unpack a single JSON file (e.g., from chat interface) to YAML.

Automatically extracts metadata from JSON and saves to qsp-metadata-storage
with filename format: {param_name}_{author_year}_{cancer_type}_{context_hash}.yaml
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


def load_header_from_csv(csv_file: Path, parameter_name: str) -> dict:
    """
    Load header fields from extraction CSV for a specific parameter.

    Args:
        csv_file: Path to extraction CSV
        parameter_name: Parameter name to search for

    Returns:
        Dict with header fields or None if not found
    """
    import csv

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['parameter_name'] == parameter_name:
                return {
                    'parameter_name': row['parameter_name'],
                    'parameter_units': row.get('parameter_units', ''),
                    'parameter_definition': row.get('parameter_description', ''),
                    'cancer_type': row['cancer_type'],
                    'model_context': row.get('model_context', ''),
                    'context_hash': row.get('definition_hash', '')
                }

    return None


def convert_json_to_yaml(json_file: Path, output_dir: Path = None, csv_file: Path = None, param_name: str = None):
    """
    Convert a JSON file to YAML and save to qsp-metadata-storage.

    Extracts metadata from JSON to construct filename automatically.
    Optionally loads header fields from extraction CSV.

    Args:
        json_file: Path to input JSON file
        output_dir: Optional output directory (defaults to ../qsp-metadata-storage/parameter_estimates)
        csv_file: Optional extraction CSV for header fields
        param_name: Parameter name to look up in CSV
    """
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    # Load header fields from CSV if provided
    if csv_file and param_name:
        header_data = load_header_from_csv(csv_file, param_name)
        if not header_data:
            print(f"Error: Parameter '{param_name}' not found in {csv_file}")
            sys.exit(1)

        # Build complete data with header fields first
        data = {}
        data['parameter_name'] = header_data['parameter_name']
        data['parameter_units'] = header_data['parameter_units']
        data['parameter_definition'] = header_data['parameter_definition']
        data['cancer_type'] = header_data['cancer_type']

        # Parse model_context JSON
        if header_data['model_context']:
            try:
                data['model_context'] = json.loads(header_data['model_context'])
            except json.JSONDecodeError:
                data['model_context'] = header_data['model_context']

        data['context_hash'] = header_data['context_hash']

        # Add tags with "ai-generated"
        tags = []
        if 'ai-generated' not in tags:
            tags.append('ai-generated')
        data['tags'] = tags

        # Add LLM-generated fields
        data.update(llm_data)

        parameter_name = header_data['parameter_name']
        context_hash = header_data['context_hash']
    else:
        # Use data from JSON itself
        data = llm_data

        parameter_name = data.get('parameter_name')
        context_hash = data.get('context_hash')

        if not parameter_name:
            print("Error: JSON must contain 'parameter_name' field (or use --csv and --param-name)")
            sys.exit(1)

        if not context_hash:
            print("Warning: No 'context_hash' field found, using 'unknown'")
            context_hash = 'unknown'

        # Add "ai-generated" tag if not present
        tags = data.get('tags', [])
        if not isinstance(tags, list):
            tags = []
        if 'ai-generated' not in tags:
            tags.append('ai-generated')
        data['tags'] = tags

    # Extract cancer_type for filename
    cancer_type = data.get('cancer_type', 'unknown')

    # Extract first source tag for filename
    author_year = extract_first_source_tag(data)
    if not author_year:
        print("Warning: No sources found, using 'unknown' for author_year")
        author_year = 'unknown'

    # Add header comment
    header_comment = "# PARAMETER DEFINITION (from model context)\n"
    header_comment += "# " + "=" * 76 + "\n"

    # Custom representer for multi-line strings
    def str_representer(dumper, data):
        if '\n' in data:
            # Use literal block style for multi-line strings
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)

    # Convert to YAML
    yaml_str = yaml.dump(data,
                        Dumper=yaml.SafeDumper,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                        width=1000)

    final_output = header_comment + yaml_str

    # Determine output directory
    if output_dir is None:
        # Default to qsp-metadata-storage/parameter_estimates
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent.parent / "qsp-metadata-storage" / "parameter_estimates"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct filename: {param_name}_{author_year}_{cancer_type}_{context_hash}.yaml
    filename = f"{parameter_name}_{author_year}_{cancer_type}_{context_hash}.yaml"
    output_file = get_unique_filename(output_dir / filename)

    # Write YAML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)

    print(f"Saved: {output_file}")
    print(f"  Cancer type: {cancer_type}")
    print(f"  Parameter: {parameter_name}")
    print(f"  Source: {author_year}")
    print(f"  Hash: {context_hash}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unpack a single JSON file to YAML in qsp-metadata-storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extracts metadata from JSON and saves to qsp-metadata-storage
with filename format: {param_name}_{author_year}_{cancer_type}_{context_hash}.yaml

Examples:
  # Save to default location (../qsp-metadata-storage/parameter_estimates)
  python scripts/unpack_single_json.py chat_response.json

  # Save to custom directory
  python scripts/unpack_single_json.py chat_response.json --output-dir /path/to/output

  # Load header fields from extraction CSV
  python scripts/unpack_single_json.py chat_response.json --csv batch_jobs/input.csv --param-name k_fib_death
        """
    )

    parser.add_argument('json_file', type=Path, help='Input JSON file')
    parser.add_argument('--output-dir', type=Path, help='Output directory (default: ../qsp-metadata-storage/parameter_estimates)')
    parser.add_argument('--csv', type=Path, help='Extraction CSV file for header fields')
    parser.add_argument('--param-name', type=str, help='Parameter name to look up in CSV')

    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: {args.json_file} not found")
        sys.exit(1)

    # Validate CSV arguments
    if args.csv and not args.param_name:
        print("Error: --param-name required when --csv is provided")
        sys.exit(1)

    if args.param_name and not args.csv:
        print("Error: --csv required when --param-name is provided")
        sys.exit(1)

    # Convert
    convert_json_to_yaml(args.json_file, args.output_dir, args.csv, args.param_name)


if __name__ == "__main__":
    main()
