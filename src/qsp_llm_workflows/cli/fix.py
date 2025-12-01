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
    qsp-fix parameter_estimates --dir metadata-storage/to-review/parameter_estimates --immediate
    qsp-fix test_statistics --dir metadata-storage/to-review/test_statistics --timeout 7200
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
        required=True,
        help="Directory containing files to fix (e.g., metadata-storage/to-review/parameter_estimates)",
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
        "--preview-prompts",
        action="store_true",
        help="Preview prompts without sending to API (saves to batch_jobs/prompt_preview.jsonl)",
    )

    parser.add_argument(
        "--validation-results-dir",
        type=str,
        default=None,
        help="Custom validation results directory (default: output/validation_results)",
    )

    args = parser.parse_args()

    # Determine directories
    base_dir = Path.cwd()
    data_dir = Path(args.dir).resolve()

    # Validate data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Derive storage_dir from data_dir (parent of to-review or the directory itself)
    # Expected structure: metadata-storage/to-review/{workflow_type}
    if data_dir.parent.name == "to-review":
        storage_dir = data_dir.parent.parent
    else:
        storage_dir = data_dir.parent

    # Determine validation results directory
    if args.validation_results_dir:
        validation_results_dir = Path(args.validation_results_dir)
    else:
        validation_results_dir = Path("output/validation_results")

    if not validation_results_dir.exists():
        print("Error: Validation results directory not found", file=sys.stderr)
        print(f"Expected: {validation_results_dir}", file=sys.stderr)
        print("\nRun qsp-validate first to generate validation reports.", file=sys.stderr)
        sys.exit(1)

    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Import workflow components
    from qsp_llm_workflows.core.config import WorkflowConfig
    from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator

    # Determine workflow type (convert CLI format to internal format)
    if args.workflow_type == "test_statistics":
        workflow_type = "test_statistic"
    else:
        workflow_type = "parameter"

    # Create workflow config
    config = WorkflowConfig(
        base_dir=base_dir,
        storage_dir=storage_dir,
        openai_api_key=api_key,
        batch_timeout=args.timeout,
    )

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(config)

    # Run validation fix workflow
    print("=" * 60)
    if args.preview_prompts:
        print("VALIDATION FIX WORKFLOW - PREVIEW MODE")
    else:
        print("VALIDATION FIX WORKFLOW")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Validation results: {validation_results_dir}")
    print(f"Mode: {'Immediate (Responses API)' if args.immediate else 'Batch API'}")
    print()

    result = orchestrator.run_validation_fix_workflow(
        data_dir=data_dir,
        validation_results_dir=validation_results_dir,
        workflow_type=workflow_type,
        immediate=args.immediate,
        timeout=args.timeout,
        progress_callback=print,
        preview_prompts=args.preview_prompts,
    )

    # Check result
    if result.status == "failed":
        print(f"\n✗ Validation fix workflow failed: {result.error}", file=sys.stderr)
        sys.exit(1)

    # Success!
    print()
    print("=" * 60)
    if args.preview_prompts:
        print("PROMPT PREVIEW COMPLETE")
        print("=" * 60)
        print(f"\nPreview file: {result.output_directory}")
        print(f"Fix request count: {result.file_count}")
        print()
        print("Next steps:")
        print(f"  1. Review prompts in: {result.output_directory}")
        print("  2. If satisfied, run without --preview-prompts to execute")
    else:
        print("VALIDATION FIX COMPLETE")
        print("=" * 60)
        print(f"\nFixed {result.file_count} files in {result.output_directory}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print()
        print("Next steps:")
        print("  1. Re-run validation to verify fixes")
        if args.dir:
            print(f"     qsp-validate {args.workflow_type} --dir {args.dir}")
        else:
            print(f"     qsp-validate {args.workflow_type}")
    print()


if __name__ == "__main__":
    main()
