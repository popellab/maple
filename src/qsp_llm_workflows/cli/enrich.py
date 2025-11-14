#!/usr/bin/env python3
"""
CLI wrapper for CSV enrichment.

Entry point: qsp-enrich-csv
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Enrich CSV files with model context",
        epilog="""
Examples:
    qsp-enrich-csv parameter input.csv model_defs.json PDAC -o output.csv
    qsp-enrich-csv test_statistic input.csv model_context.txt scenario.yaml -o output.csv
        """
    )

    parser.add_argument(
        "type",
        choices=["parameter", "test_statistic"],
        help="Type of enrichment"
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Input CSV file"
    )

    parser.add_argument(
        "context_file",
        type=Path,
        help="Context file (model definitions JSON for parameters, model context text for test statistics)"
    )

    parser.add_argument(
        "additional_arg",
        help="Cancer type for parameters, scenario YAML for test statistics"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output CSV file (default: enriched_<input>.csv)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    if not args.context_file.exists():
        print(f"Error: Context file not found: {args.context_file}", file=sys.stderr)
        sys.exit(1)

    # Import appropriate enrichment module
    if args.type == "parameter":
        from qsp_llm_workflows.prepare.enrich_parameter_csv import main as enrich_main
        cancer_type = args.additional_arg

        # Set up sys.argv for the enrich script
        sys.argv = [
            "enrich_parameter_csv.py",
            str(args.input_csv),
            str(args.context_file),
            cancer_type
        ]

        if args.output:
            sys.argv.extend(["-o", str(args.output)])

        enrich_main()

    else:  # test_statistic
        from qsp_llm_workflows.prepare.enrich_test_statistic_csv import main as enrich_main
        scenario_file = Path(args.additional_arg)

        if not scenario_file.exists():
            print(f"Error: Scenario file not found: {scenario_file}", file=sys.stderr)
            sys.exit(1)

        # Set up sys.argv for the enrich script
        sys.argv = [
            "enrich_test_statistic_csv.py",
            str(args.input_csv),
            str(args.context_file),
            str(scenario_file)
        ]

        if args.output:
            sys.argv.extend(["-o", str(args.output)])

        enrich_main()


if __name__ == "__main__":
    main()
