#!/usr/bin/env python3
"""
Create batch requests for lightweight JSON validation from raw batch results.

This script validates JSON structure and required fields, fixing syntax errors
and structural issues without deep content validation.
"""

import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.batch_creator import JsonValidationBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent  # Project root, where templates/ is located

    # Requires batch results JSONL file and input CSV
    if len(sys.argv) != 3:
        print("Usage: create_json_validation_batch.py batch_results.jsonl input.csv")
        print("       batch_results.jsonl: JSONL file from batch_monitor with raw JSON responses")
        print("       input.csv: Original input CSV with parameter metadata (for header fields)")
        print("")
        print("This script creates lightweight JSON validation requests for parameter extractions")
        print("from raw JSON responses (before unpacking to YAML).")
        print("")
        print("This validates structure and required fields without deep content review.")
        sys.exit(1)

    batch_results = Path(sys.argv[1])
    input_csv = Path(sys.argv[2])

    # Validate input files exist
    if not batch_results.exists():
        print(f"Error: File not found: {batch_results}")
        sys.exit(1)

    if not input_csv.exists():
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)

    # Create batch creator and process
    creator = JsonValidationBatchCreator(base_dir)
    output_path = creator.run(None, batch_results, input_csv)

    print(f"JSON validation batch file created: {output_path}")
    print("")
    print("Next steps:")
    print(f"  # Process immediately for testing:")
    print(f"  python scripts/run/upload_immediate.py {output_path}")
    print("")
    print(f"  # Or submit as batch job:")
    print(f"  python scripts/run/upload_batch.py {output_path}")


if __name__ == "__main__":
    main()
