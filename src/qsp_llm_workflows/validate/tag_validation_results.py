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
from pathlib import Path
import yaml
from datetime import datetime

from qsp_llm_workflows.core.validation_utils import load_yaml_directory


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
            # Remove existing validation section (including preceding comment and blank lines)
            lines = content.split('\n')

            # Find where validation section starts
            validation_start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('validation:'):
                    validation_start_idx = i
                    break

            if validation_start_idx is not None:
                # Look backwards to find where to start removing (skip blank lines and comments)
                removal_start_idx = validation_start_idx
                for i in range(validation_start_idx - 1, -1, -1):
                    line = lines[i]
                    # If blank or comment line immediately before, include it in removal
                    if line.strip() == '' or line.strip().startswith('#'):
                        removal_start_idx = i
                    else:
                        break

                # Find where validation section ends
                validation_end_idx = len(lines)
                in_validation = False
                for i in range(validation_start_idx, len(lines)):
                    line = lines[i]
                    if line.strip().startswith('validation:'):
                        in_validation = True
                    elif in_validation and line and not line[0].isspace():
                        # Found next top-level key
                        validation_end_idx = i
                        break

                # Keep lines before and after validation section
                new_lines = lines[:removal_start_idx] + lines[validation_end_idx:]
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


def tag_directory(data_dir: str, validation_tags: list) -> int:
    """
    Tag all YAML files in a directory with validation results.

    Args:
        data_dir: Directory containing YAML files
        validation_tags: List of validation tag strings

    Returns:
        Number of files successfully tagged
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise ValueError(f"Directory not found: {data_dir}")

    # Find all YAML files
    yaml_files = list(data_dir_path.glob("*.yaml")) + list(data_dir_path.glob("*.yml"))

    if not yaml_files:
        return 0

    # Tag each file
    success_count = 0
    for yaml_file in yaml_files:
        if tag_file(yaml_file, validation_tags):
            success_count += 1

    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="Tag YAML files with validation results"
    )
    parser.add_argument("data_dir", help="Directory with YAML files")
    parser.add_argument("tags", nargs="+", help="Validation tags to add")

    args = parser.parse_args()

    print(f"Tagging files in {args.data_dir}...")
    print(f"Tags: {', '.join(args.tags)}")

    try:
        success_count = tag_directory(args.data_dir, args.tags)

        # Count total files
        data_dir_path = Path(args.data_dir)
        total_files = len(list(data_dir_path.glob("*.yaml")) + list(data_dir_path.glob("*.yml")))

        if total_files == 0:
            print("No YAML files found")
            sys.exit(0)

        print(f"\n✓ Tagged {success_count}/{total_files} files")

        if success_count < total_files:
            print(f"⚠ Failed to tag {total_files - success_count} files", file=sys.stderr)
            sys.exit(1)

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
