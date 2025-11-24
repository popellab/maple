#!/usr/bin/env python3
"""
CLI wrapper for validation fix workflow.

Entry point: qsp-fix
"""
import argparse
import sys
from pathlib import Path


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
        "--dir",
        type=str,
        default=None,
        help="Custom directory containing files to fix (default: ../qsp-metadata-storage/to-review/{workflow_type})",
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

    # Determine data directory
    if args.dir:
        data_dir = Path(args.dir)
    else:
        to_review_subdir = (
            "test_statistics" if args.workflow_type == "test_statistics" else "parameter_estimates"
        )
        data_dir = storage_dir / "to-review" / to_review_subdir

    # Validate data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # TODO: Implement validation fix workflow
    # The validation fix workflow is not yet implemented in the refactored architecture.
    # This would involve:
    # 1. Loading validation reports from output/validation_results/
    # 2. Creating fix batch requests with failed YAMLs + error messages
    # 3. Using the validation_fix_prompt.md template
    # 4. Running through batch or immediate API
    # 5. Unpacking results back to the original directory

    print("Error: Validation fix workflow is not yet implemented", file=sys.stderr)
    print("\nThe qsp-fix command is currently under development.", file=sys.stderr)
    print(
        "For now, you'll need to manually fix validation errors in the YAML files.", file=sys.stderr
    )
    print(f"\nFiles to fix are in: {data_dir}", file=sys.stderr)
    print("Validation reports are in: output/validation_results/", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
