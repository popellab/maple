#!/usr/bin/env python3
"""
Create batch requests for quick parameter estimation using the new class-based batch creator system.
"""

import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.batch_creator import QuickEstimateBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    if len(sys.argv) != 2:
        print("Usage: create_quick_estimate_batch.py input.csv")
        print("       input.csv: CSV with parameter metadata columns")
        print("")
        print("Required CSV columns:")
        print("  - cancer_type: Cancer type (e.g., PDAC, NSCLC)")
        print("  - parameter_name: Parameter name")
        print("  - parameter_units: Units for the parameter")
        print("  - parameter_description: Full parameter definition/description")
        print("  - model_context: JSON string with reactions_and_rules")
        print("  - definition_hash: Hash of model context for tracking")
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