#!/usr/bin/env python3
"""
Create batch requests for test statistic generation from biological expectations.

This script processes a CSV file containing biological expectations and generates
OpenAI batch API requests to create test statistic definitions for QSP model
validation based on literature data.

Input CSV format:
- test_statistic_id: Unique identifier for the test statistic
- model_context: Model structure and variable information
- scenario_context: Experimental scenario and biological context
- required_species: Comma-separated list of model species (e.g., "V_T.TumorVolume", "V_T.T_eff,V_T.T_reg")
- derived_species_description: Biological description of what the derived species represents

Optional model context CSV format:
- Variable: Model variable name
- Description: Description of the variable
- Units: Units of measurement
- Compartment: Model compartment

Usage:
    python scripts/prepare/create_test_statistic_batch.py input.csv [model_context.csv]

Examples:
    python scripts/prepare/create_test_statistic_batch.py test_statistics.csv
    python scripts/prepare/create_test_statistic_batch.py test_statistics.csv model_variables.csv
"""

import argparse
import sys
from pathlib import Path

from qsp_llm_workflows.core.batch_creator import TestStatisticBatchCreator
from qsp_llm_workflows.core.resource_utils import get_package_root


def main():
    parser = argparse.ArgumentParser(
        description="Create test statistic batch requests from CSV input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV file with test_statistic_id, model_context, scenario_context, required_species, and derived_species_description columns"
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
        help="Output path for batch requests JSONL file (default: batch_jobs/test_statistic_requests.jsonl)"
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

    # Get package root directory (contains templates/)
    base_dir = get_package_root()

    # Create batch creator
    creator = TestStatisticBatchCreator(base_dir)

    # Process and create batch requests
    try:
        print(f"Processing test statistic input from {args.input_csv}")
        if args.model_context_csv:
            print(f"Using model context from {args.model_context_csv}")

        output_path = creator.run(
            output_path=args.output,
            input_csv=args.input_csv,
            model_context_csv=args.model_context_csv
        )

        print(f"✓ Test statistic batch requests created: {output_path}")
        print(f"Next steps:")
        print(f"  1. Upload batch: python scripts/run/upload_batch.py {output_path}")
        print(f"  2. Monitor progress: python scripts/run/batch_monitor.py <batch_id>")
        print(f"  3. Process results when complete")

    except Exception as e:
        print(f"Error creating batch requests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()