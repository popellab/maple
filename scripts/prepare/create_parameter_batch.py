#!/usr/bin/env python3
"""
Create batch requests for parameter extraction using the new class-based batch creator system.

Required CSV columns:
  - cancer_type: Cancer type or indication
  - parameter_name: Name of the parameter
  - parameter_units: Units of measurement
  - parameter_description: Description/definition of the parameter
  - model_context: JSON containing reactions_and_rules with model usage information
  - definition_hash: Hash identifier for the parameter definition
"""

import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.batch_creator import ParameterBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent  # Project root, where templates/ is located

    if len(sys.argv) != 2:
        print("Usage: create_parameter_batch.py input.csv")
        print("")
        print("Required CSV columns:")
        print("  - cancer_type")
        print("  - parameter_name")
        print("  - parameter_units")
        print("  - parameter_description")
        print("  - model_context (JSON)")
        print("  - definition_hash")
        sys.exit(1)

    input_csv = Path(sys.argv[1])

    # Validate input file exists
    if not input_csv.exists():
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)

    # Create batch creator and process
    creator = ParameterBatchCreator(base_dir)
    output_path = creator.run(None, input_csv)  # Use default output path

    print(f"Batch file created: {output_path}")


if __name__ == "__main__":
    main()