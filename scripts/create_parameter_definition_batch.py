#!/usr/bin/env python3
"""
Create batch requests for parameter definition generation using the class-based batch creator system.
"""

import sys
import argparse
from pathlib import Path

from batch_creator import ParameterDefinitionBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Create parameter definition batch requests")
    parser.add_argument("input_csv", help="CSV with cancer_type and parameter_name columns")
    parser.add_argument("params_csv", help="Simbio parameters (Name, Units) - e.g., simbio_parameters.csv")
    parser.add_argument("reactions_csv", help="Model context (Parameter, Reaction, etc.)")
    parser.add_argument("--force", action="store_true",
                       help="Force creation of requests even for parameters with existing definitions")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    params_csv = Path(args.params_csv)
    reactions_csv = Path(args.reactions_csv)

    # Validate input files exist
    for file_path in [input_csv, params_csv, reactions_csv]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    # Create batch creator and process
    creator = ParameterDefinitionBatchCreator(base_dir)

    # Pass skip_existing parameter (opposite of force)
    skip_existing = not args.force
    requests = creator.process(input_csv, params_csv, reactions_csv, skip_existing=skip_existing)

    if requests:
        output_path = creator.get_default_output_path()
        creator.write_batch_file(requests, output_path)
        print(f"Batch file created: {output_path}")
    else:
        print("No requests to create - all parameters already have definitions (use --force to override)")

    print(f"Batch type: {creator.get_batch_type()}")


if __name__ == "__main__":
    main()