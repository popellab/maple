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
        choices=[
            "parameter",
            "test_statistic",
            "calibration_target",
            "isolated_system_target",
            "submodel_target",
        ],
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
        default="low",
        help="Reasoning effort level for OpenAI API (default: low)",
    )

    parser.add_argument(
        "--preview-prompts",
        action="store_true",
        help="Preview prompts without sending to API (saves preview file to jobs/)",
    )

    parser.add_argument(
        "--model-structure",
        type=Path,
        help="Path to model_structure.json for parameter context and validation (isolated_system_target)",
    )

    parser.add_argument(
        "--model-context",
        type=Path,
        help="Path to model_context.txt with high-level model description (isolated_system_target)",
    )

    parser.add_argument(
        "--previous-extractions",
        type=Path,
        help="Path to directory with previous extractions (submodel_target). "
        "Sources from matching targets will be excluded from new extractions.",
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

    # Validate required options for specific workflow types
    if args.type in ("isolated_system_target", "submodel_target"):
        if not args.model_structure:
            print(
                f"Error: --model-structure is required for {args.type} workflow",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.model_context:
            print(
                f"Error: --model-context is required for {args.type} workflow",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.model_context.exists():
            print(f"Error: Model context file not found: {args.model_context}", file=sys.stderr)
            sys.exit(1)

    # Load configuration from environment with explicit storage directory
    try:
        config = WorkflowConfig.from_env(storage_dir=output_dir)

        # Add model_structure_file if provided
        if args.model_structure:
            if not args.model_structure.exists():
                print(
                    f"Error: Model structure file not found: {args.model_structure}",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Validate previous-extractions if provided
            previous_extractions_dir = None
            if hasattr(args, "previous_extractions") and args.previous_extractions:
                if not args.previous_extractions.exists():
                    print(
                        f"Error: Previous extractions directory not found: {args.previous_extractions}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                previous_extractions_dir = args.previous_extractions
            # Create new config with model files (config is frozen)
            config = WorkflowConfig(
                base_dir=config.base_dir,
                storage_dir=config.storage_dir,
                openai_api_key=config.openai_api_key,
                openai_model=config.openai_model,
                reasoning_effort=config.reasoning_effort,
                model_structure_file=args.model_structure,
                model_context_file=args.model_context,
                previous_extractions_dir=previous_extractions_dir,
            )
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
            if result.error:
                print(f"Error: {result.error}")
            print(f"Files extracted: {result.file_count}")
            print(f"Output directory: {result.output_directory}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            print()
            if result.status == "success":
                print("Next steps:")
                print(f"  1. Review files in: {result.output_directory}")
                print(
                    f"  2. Run validation: qsp-validate {args.type} --dir {result.output_directory}"
                )
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
