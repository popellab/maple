#!/usr/bin/env python3
"""
CLI wrapper for extraction workflow.

Entry point: qsp-extract
"""
import argparse
import sys
from pathlib import Path

from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator


def load_api_key() -> str:
    """Load OpenAI API key from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise ValueError("OPENAI_API_KEY not found in .env file")


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

    # Determine directories
    base_dir = Path.cwd()
    storage_dir = base_dir.parent / "qsp-metadata-storage"

    # Validate storage directory exists
    if not storage_dir.exists():
        print(f"Error: Metadata storage directory not found: {storage_dir}", file=sys.stderr)
        print("Expected qsp-metadata-storage as sibling directory", file=sys.stderr)
        sys.exit(1)

    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Run workflow
    orchestrator = WorkflowOrchestrator(base_dir, storage_dir, api_key)

    try:
        result = orchestrator.run_complete_workflow(
            input_csv=Path(args.input_csv),
            workflow_type=args.type,
            timeout=args.timeout,
            push=not args.no_push,
            immediate=args.immediate,
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
