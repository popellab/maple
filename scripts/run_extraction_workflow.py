#!/usr/bin/env python3
"""
Automated extraction workflow CLI.

Runs complete extraction pipeline from batch creation through git commit/push.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from workflow_orchestrator import WorkflowOrchestrator


def load_api_key():
    """Load API key from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment")

    return api_key


def print_progress(message: str):
    """Print progress message."""
    print(message)


def print_summary(results: dict):
    """Print workflow summary."""
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)

    if results["status"] == "success":
        print(f"✓ Status: SUCCESS")
        print(f"✓ Workflow type: {results['workflow_type']}")
        print(f"✓ Files extracted: {results['file_count']}")
        print(f"✓ Duration: {results['duration_seconds']:.1f}s")
        print(f"✓ Review branch: {results['branch_name']}")

        if results.get('pushed'):
            print(f"✓ Pushed to origin/{results['branch_name']}")
        else:
            print(f"✓ Local branch created: {results['branch_name']}")

        print("\nNext steps:")
        print("  1. cd ../qsp-metadata-storage")
        print(f"  2. git checkout {results['branch_name']}")
        print("  3. Review files in to-review/")
        print("  4. Move approved files to appropriate directories")
        print("  5. Merge to main when approved")
    else:
        print(f"✗ Status: FAILED")
        print(f"✗ Error: {results.get('error', 'Unknown error')}")
        print(f"✗ Duration: {results['duration_seconds']:.1f}s")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete extraction workflow with automated validation and git operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parameter extraction
  python scripts/run_extraction_workflow.py input.csv --type parameter

  # Test statistic extraction with custom timeout
  python scripts/run_extraction_workflow.py test_stats.csv --type test_statistic --timeout 7200

  # Quick estimates without validation
  python scripts/run_extraction_workflow.py quick.csv --type quick_estimate --skip-validation

  # Create branch locally without pushing
  python scripts/run_extraction_workflow.py input.csv --type parameter --no-push
        """
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--type",
        choices=["parameter", "test_statistic", "quick_estimate"],
        required=True,
        help="Type of extraction workflow to run"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum seconds to wait for batch completion (default: 3600)"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip checklist validation step"
    )

    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Create branch locally but don't push to remote"
    )

    parser.add_argument(
        "--branch-prefix",
        default="review/batch",
        help="Prefix for review branch name (default: review/batch)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        sys.exit(1)

    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine directories
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    storage_dir = base_dir.parent / "qsp-metadata-storage"

    if not storage_dir.exists():
        print(f"Error: qsp-metadata-storage not found at {storage_dir}")
        print("Expected: ../qsp-metadata-storage relative to qsp-llm-workflows")
        sys.exit(1)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(base_dir, storage_dir, api_key)

    # Print header
    print("\n" + "=" * 70)
    print(f"AUTOMATED EXTRACTION WORKFLOW")
    print("=" * 70)
    print(f"Type: {args.type}")
    print(f"Input: {args.input_csv}")
    print(f"Timeout: {args.timeout}s")
    print(f"Validation: {'Disabled' if args.skip_validation else 'Enabled'}")
    print(f"Push: {'Disabled' if args.no_push else 'Enabled'}")
    print("=" * 70 + "\n")

    # Run workflow
    try:
        results = orchestrator.run_complete_workflow(
            input_csv=args.input_csv,
            workflow_type=args.type,
            timeout=args.timeout,
            skip_validation=args.skip_validation,
            push=not args.no_push,
            branch_prefix=args.branch_prefix,
            progress_callback=print_progress
        )

        print_summary(results)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
