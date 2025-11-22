#!/usr/bin/env python3
"""
CLI wrapper for validation fix workflow.

Entry point: qsp-fix
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
        description="Fix validation errors by re-submitting to OpenAI",
        epilog="""
Examples:
    qsp-fix parameter_estimates --immediate
    qsp-fix test_statistics --timeout 7200
        """,
    )

    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of workflow to fix",
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

    args = parser.parse_args()

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

    # Run validation fix workflow
    orchestrator = WorkflowOrchestrator(base_dir, storage_dir, api_key)

    try:
        result = orchestrator.run_validation_fix_workflow(
            workflow_type=args.workflow_type, use_batch_api=not args.immediate, timeout=args.timeout
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
