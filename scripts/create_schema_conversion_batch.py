#!/usr/bin/env python3
"""
Create batch requests for converting YAML files from one schema version to another.

This script reads existing YAML files and generates conversion requests to migrate
them to a new schema structure while preserving all data.

Flow:
- Reads YAML files and strips header fields
- Converts schemas and data to JSON
- LLM receives: old schema (JSON), new schema (JSON), current data (JSON)
- LLM returns: converted data (JSON)
- Unpacking converts JSON back to YAML and adds original header fields

Everything is presented to the LLM in JSON format for consistency and reliability.
"""

import sys
from pathlib import Path

from batch_creator import SchemaConversionBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Check arguments
    if len(sys.argv) < 4:
        print("Usage: create_schema_conversion_batch.py yaml_dir old_schema new_schema [migration_notes_file] [pattern]")
        print("       yaml_dir: Directory containing YAML files to convert")
        print("       old_schema: Path to old schema template file")
        print("       new_schema: Path to new schema template file")
        print("       migration_notes_file: (optional) Path to text file with migration instructions")
        print("       pattern: (optional) Glob pattern for YAML files (default: *.yaml)")
        print("")
        print("Example:")
        print("  python scripts/create_schema_conversion_batch.py \\")
        print("    ../qsp-metadata-storage/parameter_estimates \\")
        print("    templates/parameter_metadata_template.yaml \\")
        print("    templates/parameter_metadata_template_v2.yaml \\")
        print("    templates/migration_notes.txt \\")
        print("    '*_PDAC_*.yaml'")
        sys.exit(1)

    yaml_dir = Path(sys.argv[1])
    old_schema_path = Path(sys.argv[2])
    new_schema_path = Path(sys.argv[3])
    migration_notes_file = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    pattern = sys.argv[5] if len(sys.argv) > 5 else "*.yaml"

    # Read migration notes from file if provided
    migration_notes = ""
    if migration_notes_file and migration_notes_file.exists():
        with open(migration_notes_file, 'r', encoding='utf-8') as f:
            migration_notes = f.read()
    elif migration_notes_file:
        print(f"Warning: Migration notes file not found: {migration_notes_file}")
        print("Proceeding without migration notes...")
        migration_notes = ""

    # Validate inputs
    if not yaml_dir.exists():
        print(f"Error: Directory not found: {yaml_dir}")
        sys.exit(1)

    if not old_schema_path.exists():
        print(f"Error: Old schema file not found: {old_schema_path}")
        sys.exit(1)

    if not new_schema_path.exists():
        print(f"Error: New schema file not found: {new_schema_path}")
        sys.exit(1)

    # Create batch creator and process
    creator = SchemaConversionBatchCreator(base_dir)
    output_path = creator.run(None, yaml_dir, old_schema_path, new_schema_path,
                             migration_notes, pattern)

    print(f"\nSchema conversion batch file created: {output_path}")
    print("")
    if migration_notes:
        print(f"Using migration notes from: {migration_notes_file}")
        print("")
    print("Next steps:")
    print(f"  # Review a request:")
    print(f"  python scripts/inspect_jsonl.py {output_path} 0")
    print("")
    print(f"  # Extract prompt to examine:")
    print(f"  python scripts/extract_prompt.py {output_path} 0")
    print("")
    print(f"  # Process immediately for testing:")
    print(f"  python scripts/upload_immediate.py {output_path}")
    print("")
    print(f"  # Or submit as batch job:")
    print(f"  python scripts/upload_batch.py {output_path}")


if __name__ == "__main__":
    main()
