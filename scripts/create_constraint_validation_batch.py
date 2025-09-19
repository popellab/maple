#!/usr/bin/env python3
"""
Create batch requests for constraint validation test generation from biological expectations.

This script processes a CSV file containing constraint descriptions and generates
OpenAI batch API requests to create MATLAB unit test-style constraint validation
definitions for QSP model validation.

Input CSV format:
- constraint_id: Unique identifier for the constraint
- constraint_description: Biological expectation or clinical constraint to be formalized
- cancer_type: (optional) Cancer type context
- parameter_context: (optional) Related parameter context

Optional model context CSV format:
- Variable: Model variable name
- Description: Description of the variable
- Units: Units of measurement
- Compartment: Model compartment

Usage:
    python scripts/create_constraint_validation_batch.py input.csv [model_context.csv]

Examples:
    python scripts/create_constraint_validation_batch.py constraints.csv
    python scripts/create_constraint_validation_batch.py constraints.csv model_variables.csv
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path to import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.batch_creator import ConstraintValidationBatchCreator


def main():
    parser = argparse.ArgumentParser(
        description="Create constraint validation batch requests from CSV input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV file with constraint_id and constraint_description columns"
    )

    parser.add_argument(
        "model_context_csv",
        type=Path,
        nargs="?",
        help="Optional CSV file with model variable information"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for batch requests JSONL file (default: batch_jobs/constraint_validation_requests.jsonl)"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input CSV file '{args.input_csv}' not found")
        sys.exit(1)

    # Validate model context file if provided
    if args.model_context_csv and not args.model_context_csv.exists():
        print(f"Error: Model context CSV file '{args.model_context_csv}' not found")
        sys.exit(1)

    # Create batch creator
    creator = ConstraintValidationBatchCreator(project_root)

    # Process and create batch requests
    try:
        print(f"Processing constraint validation input from {args.input_csv}")
        if args.model_context_csv:
            print(f"Using model context from {args.model_context_csv}")

        output_path = creator.run(
            output_path=args.output,
            input_csv=args.input_csv,
            model_context_csv=args.model_context_csv
        )

        print(f"✓ Constraint validation batch requests created: {output_path}")
        print(f"Next steps:")
        print(f"  1. Upload batch: python scripts/upload_batch.py {output_path}")
        print(f"  2. Monitor progress: python scripts/batch_monitor.py <batch_id>")
        print(f"  3. Process results when complete")

    except Exception as e:
        print(f"Error creating batch requests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()