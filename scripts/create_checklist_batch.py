#!/usr/bin/env python3
"""
Create batch requests for parameter checklist auditing using the new class-based batch creator system.
"""

import sys
from pathlib import Path

from batch_creator import ParameterChecklistBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Requires input.csv with cancer_type and parameter_name columns
    if len(sys.argv) != 2:
        print("Usage: create_checklist_batch.py input.csv")
        print("       input.csv: CSV with cancer_type and parameter_name columns")
        print("")
        print("This script creates checklist batch requests for auditing parameter extractions.")
        print("It requires:")
        print("  - Parameter definitions in ../qsp-metadata-storage/parameter_estimates/parameter-definitions/")
        print("  - Study YAMLs to audit in ../qsp-metadata-storage/parameter_estimates/")
        sys.exit(1)

    input_csv = Path(sys.argv[1])

    # Validate input file exists
    if not input_csv.exists():
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)

    # Create batch creator and process
    creator = ParameterChecklistBatchCreator(base_dir)
    output_path = creator.run(None, input_csv)  # Use default output path

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