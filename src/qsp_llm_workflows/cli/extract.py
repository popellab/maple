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
        "--model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model to use (default: gpt-5.1)",
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

    # Validate optional paths
    model_structure_file = None
    model_context_file = None
    previous_extractions_dir = None

    if args.model_structure:
        if not args.model_structure.exists():
            print(f"Error: Model structure file not found: {args.model_structure}", file=sys.stderr)
            sys.exit(1)
        model_structure_file = args.model_structure.resolve()

    if args.model_context:
        model_context_file = args.model_context.resolve()

    if args.previous_extractions:
        previous_extractions_dir = args.previous_extractions.resolve()
        if not previous_extractions_dir.exists():
            print(
                f"Error: Previous extractions directory not found: {previous_extractions_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load API key from environment, then create single config with all values
    try:
        env_config = WorkflowConfig.from_env(storage_dir=output_dir)

        config = WorkflowConfig(
            base_dir=env_config.base_dir,
            storage_dir=env_config.storage_dir,
            openai_api_key=env_config.openai_api_key,
            openai_model=args.model,
            reasoning_effort=args.reasoning_effort,
            model_structure_file=model_structure_file,
            model_context_file=model_context_file,
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
        print(f"Model: {args.model}")
        print(f"Reasoning effort: {args.reasoning_effort}")
        print()

        result = orchestrator.run_complete_workflow(
            input_csv=Path(args.input_csv),
            workflow_type=args.type,
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
