#!/usr/bin/env python3
"""
CLI wrapper for extraction workflow.

Entry point: qsp-extract
"""
import argparse
import sys
from pathlib import Path

from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Run automated extraction workflow",
        epilog="""
Examples:
    qsp-extract input.csv --type parameter
    qsp-extract input.csv --type test_statistic --immediate
    qsp-extract input.csv --type parameter --timeout 7200 --no-push
        """,
    )

    parser.add_argument("input_csv", type=Path, help="Input CSV file with extraction requests")

    parser.add_argument(
        "--type",
        required=True,
        choices=["parameter", "test_statistic"],
        help="Type of extraction workflow",
    )

    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Use immediate mode (Responses API) instead of batch API",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for batch monitoring (default: 3600)",
    )

    parser.add_argument(
        "--no-push", action="store_true", help="Create branch locally without pushing to remote"
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    # Run workflow
    orchestrator = WorkflowOrchestrator()

    try:
        result = orchestrator.run_extraction_workflow(
            input_csv=str(args.input_csv),
            workflow_type=args.type,
            use_batch_api=not args.immediate,
            timeout=args.timeout,
            push_to_remote=not args.no_push,
        )

        sys.exit(0 if result else 1)

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
