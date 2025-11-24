#!/usr/bin/env python3
"""
CLI wrapper for extraction workflow.

Entry point: qsp-extract
"""
import argparse
import sys
from pathlib import Path

from qsp_llm_workflows.core.config import WorkflowConfig
from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator


def print_progress(message: str):
    """Print progress message to stdout."""
    print(message)


def main():
    parser = argparse.ArgumentParser(
        description="Run automated extraction workflow",
        epilog="""
Examples:
    qsp-extract input.csv --type parameter
    qsp-extract input.csv --type test_statistic --immediate
    qsp-extract input.csv --type parameter --timeout 7200
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
        help="Timeout in seconds for batch monitoring (default: from config or 3600)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    # Load configuration from environment
    try:
        config = WorkflowConfig.from_env()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Make sure OPENAI_API_KEY is set in .env file", file=sys.stderr)
        sys.exit(1)

    # Validate storage directory exists
    if not config.storage_dir.exists():
        print(
            f"Error: Metadata storage directory not found: {config.storage_dir}",
            file=sys.stderr,
        )
        print("Expected qsp-metadata-storage as sibling directory", file=sys.stderr)
        sys.exit(1)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(config)

    try:
        # Run workflow
        print(f"\nStarting {args.type} extraction workflow...")
        print(f"Mode: {'immediate' if args.immediate else 'batch'}")
        print(f"Input: {args.input_csv}")
        print()

        result = orchestrator.run_complete_workflow(
            input_csv=Path(args.input_csv),
            workflow_type=args.type,
            immediate=args.immediate,
            timeout=args.timeout,
            progress_callback=print_progress,
        )

        # Print summary
        print()
        print("=" * 70)
        print("WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"Status: {result.status}")
        print(f"Files extracted: {result.file_count}")
        print(f"Output directory: {result.output_directory}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print()
        print("Next steps:")
        print(f"  1. Review files in: {result.output_directory}")
        print(f"  2. Run validation: qsp-validate {result.output_directory}")
        print(f"  3. If satisfied, commit manually:")
        print(f"       cd {config.storage_dir}")
        print(f"       git add {result.output_directory}")
        print(f'       git commit -m "Add {args.type} extractions"')
        print()

        sys.exit(0)

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
