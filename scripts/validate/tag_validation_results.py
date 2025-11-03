#!/usr/bin/env python3
"""
Tag YAML files with validation results.

Adds a 'validation_tags' field to each YAML file with the list of
validation checks that passed.

Usage:
    python scripts/validate/tag_validation_results.py \
        ../qsp-metadata-storage/parameter_estimates \
        schema_compliance code_execution text_snippets
"""
import argparse
import sys
import os
from pathlib import Path
import yaml
from datetime import datetime

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import load_yaml_directory


def tag_file(file_path: Path, validation_tags: list) -> bool:
    """
    Add validation tags to a YAML file by appending to the end.
    Preserves original formatting and comments.

    Args:
        file_path: Path to YAML file
        validation_tags: List of validation tag strings

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read original file content
        with open(file_path, 'r') as f:
            content = f.read()

        # Check if validation section already exists
        if '\nvalidation:' in content or content.startswith('validation:'):
            # Remove existing validation section
            lines = content.split('\n')
            new_lines = []
            in_validation_section = False

            for line in lines:
                # Check if we're entering validation section
                if line.strip().startswith('validation:'):
                    in_validation_section = True
                    continue

                # Check if we're exiting validation section (new top-level key)
                if in_validation_section and line and not line[0].isspace():
                    in_validation_section = False

                # Keep line if not in validation section
                if not in_validation_section:
                    new_lines.append(line)

            content = '\n'.join(new_lines).rstrip()

        # Append validation section to end
        validation_yaml = f"\n\n# Validation metadata\nvalidation:\n"
        validation_yaml += f"  tags:\n"
        for tag in validation_tags:
            validation_yaml += f"    - {tag}\n"
        validation_yaml += f"  validated_at: '{datetime.now().isoformat()}'\n"

        # Write back to file
        with open(file_path, 'w') as f:
            f.write(content)
            f.write(validation_yaml)

        return True

    except Exception as e:
        print(f"Error tagging {file_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Tag YAML files with validation results"
    )
    parser.add_argument("data_dir", help="Directory with YAML files")
    parser.add_argument("tags", nargs="+", help="Validation tags to add")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Tagging files in {data_dir}...")
    print(f"Tags: {', '.join(args.tags)}")

    # Find all YAML files
    yaml_files = list(data_dir.glob("*.yaml")) + list(data_dir.glob("*.yml"))

    if not yaml_files:
        print("No YAML files found")
        sys.exit(0)

    # Tag each file
    success_count = 0
    for yaml_file in yaml_files:
        if tag_file(yaml_file, args.tags):
            success_count += 1

    print(f"\n✓ Tagged {success_count}/{len(yaml_files)} files")

    if success_count < len(yaml_files):
        print(f"⚠ Failed to tag {len(yaml_files) - success_count} files", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
