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
        description="Run automated extraction workflow using Pydantic AI",
        epilog="""
Examples:
    qsp-extract input.csv --type parameter --output-dir metadata-storage
    qsp-extract input.csv --type test_statistic --output-dir metadata-storage
    qsp-extract input.csv --type parameter --output-dir metadata-storage --preview-prompts
        """,
    )

    parser.add_argument("input_csv", type=Path, help="Input CSV file with extraction requests")

    parser.add_argument(
        "--type",
        required=True,
        choices=["parameter", "test_statistic", "calibration_target"],
        help="Type of extraction workflow",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for extracted metadata (e.g., metadata-storage)",
    )

    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="high",
        help="Reasoning effort level for OpenAI API (default: high)",
    )

    parser.add_argument(
        "--preview-prompts",
        action="store_true",
        help="Preview prompts without sending to API (saves preview file to batch_jobs/)",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input_csv.exists():
        print(f"Error: Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    # Resolve output directory
    output_dir = args.output_dir.resolve()

    # Validate output directory exists
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        print("Please create the directory or specify an existing path", file=sys.stderr)
        sys.exit(1)

    # Load configuration from environment with explicit storage directory
    try:
        config = WorkflowConfig.from_env(storage_dir=output_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Make sure OPENAI_API_KEY is set in .env file", file=sys.stderr)
        sys.exit(1)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(config)

    try:
        # Run workflow
        if args.preview_prompts:
            print("\n=== PREVIEW MODE ===")
            print(f"Building prompts for {args.type} extraction workflow...")
        else:
            print(f"\nStarting {args.type} extraction workflow (Pydantic AI)...")
        print(f"Input: {args.input_csv}")
        print(f"Reasoning effort: {args.reasoning_effort}")
        print()

        result = orchestrator.run_complete_workflow(
            input_csv=Path(args.input_csv),
            workflow_type=args.type,
            reasoning_effort=args.reasoning_effort,
            progress_callback=print_progress,
            preview_prompts=args.preview_prompts,
        )

        # Print summary
        print()
        print("=" * 70)
        if args.preview_prompts:
            print("PROMPT PREVIEW COMPLETE")
            print("=" * 70)
            print(f"Preview file: {result.output_directory}")
            print(f"Request count: {result.file_count}")
            print()
            print("Next steps:")
            print(f"  1. Review prompts in: {result.output_directory}")
            print("  2. If satisfied, run without --preview-prompts to execute")
        else:
            print("WORKFLOW COMPLETE")
            print("=" * 70)
            print(f"Status: {result.status}")
            print(f"Files extracted: {result.file_count}")
            print(f"Output directory: {result.output_directory}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            print()
            print("Next steps:")
            print(f"  1. Review files in: {result.output_directory}")
            print(f"  2. Run validation: qsp-validate {args.type} --dir {result.output_directory}")
            print("  3. If satisfied, commit manually:")
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
