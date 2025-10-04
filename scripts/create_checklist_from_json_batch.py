#!/usr/bin/env python3
"""
Create batch requests for parameter checklist auditing from raw JSON batch results.

This script reads the raw batch results JSONL file (from batch_monitor) and creates
checklist requests for each JSON response. This allows catching packing errors early
by checking the raw LLM responses before unpacking to YAML.
"""

import sys
from pathlib import Path

from batch_creator import ParameterChecklistFromJsonBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Requires batch results JSONL file and input CSV
    if len(sys.argv) != 3:
        print("Usage: create_checklist_from_json_batch.py batch_results.jsonl input.csv")
        print("       batch_results.jsonl: JSONL file from batch_monitor with raw JSON responses")
        print("       input.csv: Original input CSV with parameter metadata (for header fields)")
        print("")
        print("This script creates checklist batch requests for auditing parameter extractions")
        print("from raw JSON responses (before unpacking to YAML).")
        print("")
        print("This allows catching packing errors early by validating the raw LLM responses.")
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
    creator = ParameterChecklistFromJsonBatchCreator(base_dir)
    output_path = creator.run(None, batch_results, input_csv)  # Pass None for output_path, then both arguments

    print(f"Checklist batch file created: {output_path}")
    print("")
    print("Next steps:")
    print(f"  # Process immediately for testing:")
    print(f"  python scripts/upload_immediate.py {output_path}")
    print("")
    print(f"  # Or submit as batch job:")
    print(f"  python scripts/upload_batch.py {output_path}")


if __name__ == "__main__":
    main()
