#!/usr/bin/env python3
"""
Create batch requests for quick parameter estimation using the new class-based batch creator system.
"""

import sys
from pathlib import Path

from batch_creator import QuickEstimateBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    # Only requires input.csv - parameter definitions loaded from storage
    if len(sys.argv) != 2:
        print("Usage: create_quick_estimate_batch.py input.csv")
        print("       input.csv: CSV with cancer_type and parameter_name columns")
        print("")
        print("Note: This requires parameter definitions to be created first using:")
        print("      create_parameter_definition_batch.py input.csv params.csv reactions.csv")
        sys.exit(1)

    input_csv = Path(sys.argv[1])

    # Validate input file exists
    if not input_csv.exists():
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)

    # Create batch creator and process
    creator = QuickEstimateBatchCreator(base_dir)
    output_path = creator.run(None, input_csv)  # Use default output path

    print(f"Batch file created: {output_path}")


if __name__ == "__main__":
    main()