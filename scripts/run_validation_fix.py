#!/usr/bin/env python3
"""
Automated validation fix workflow.

Creates batch requests to fix validation errors, uploads to OpenAI API,
monitors completion, and unpacks fixed YAMLs (overwriting originals).
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
import subprocess
import time

# Add lib directory to path
lib_dir = Path(__file__).parent / "lib"
sys.path.insert(0, str(lib_dir))

# Import after adding to path
from batch_creator import ValidationFixBatchCreator
from workflow_orchestrator import WorkflowOrchestrator


def load_api_key():
    """Load API key from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment")

    return api_key


def print_progress(message: str):
    """Print progress message."""
    print(message)


def extract_prompt_to_file(batch_file: Path, base_dir: Path, index: int = 0) -> Path:
    """
    Extract a prompt from batch file and write to scratch directory.

    Args:
        batch_file: Path to batch JSONL file
        base_dir: Base directory of the project
        index: Index of request to extract (default: 0)

    Returns:
        Path to the extracted prompt file
    """
    import json

    # Create scratch directory in current project
    scratch_dir = base_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Read the specified request
    with open(batch_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                request = json.loads(line)
                break
        else:
            raise ValueError(f"Request index {index} not found in batch file")

    # Extract prompt
    prompt = request['body']['input']
    custom_id = request.get('custom_id', f'request_{index}')

    # Write to file
    prompt_file = scratch_dir / f"validation_fix_prompt_{custom_id}.txt"
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)

    return prompt_file


def unpack_results(results_file: Path, output_dir: Path, template_path: Path):
    """
    Unpack batch results to YAML files.

    Args:
        results_file: Path to batch results JSONL
        output_dir: Directory to write YAML files (will overwrite)
        template_path: Path to template for schema reference
    """
    print(f"\nUnpacking results to: {output_dir}")

    script_dir = Path(__file__).parent
    unpack_script = script_dir / "process" / "unpack_results.py"

    result = subprocess.run(
        [
            sys.executable, str(unpack_script),
            str(results_file),
            str(output_dir)
            # No input CSV needed for validation fixes
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"Unpacking failed:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix validation errors by sending failed YAMLs back to OpenAI for correction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix test statistics with immediate processing (faster)
  python scripts/run_validation_fix.py test_statistics --immediate

  # Fix parameter estimates with Batch API
  python scripts/run_validation_fix.py parameter_estimates

  # Fix with custom timeout for Batch API
  python scripts/run_validation_fix.py test_statistics --timeout 7200

Workflow:
  1. Loads validation JSON reports from output/validation_results/
  2. Creates fix batch requests for failed files
  3. Uploads to OpenAI API (or processes immediately with --immediate)
  4. Monitors until completion
  5. Unpacks fixed YAMLs (overwrites originals in to-review/)
  6. Prompts to re-run validation

Note: Original files are backed up in git history before overwriting.
        """
    )

    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of workflow to fix"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum seconds to wait for batch completion (default: 3600)"
    )

    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Use immediate Responses API instead of Batch API (faster, good for testing)"
    )

    args = parser.parse_args()

    # Determine directories and template based on workflow type
    validation_dir = Path("output/validation_results")

    if args.workflow_type == "test_statistics":
        yaml_dir = Path("../qsp-metadata-storage/to-review/test_statistics")
        template = Path("templates/test_statistic_template.yaml")
    else:  # parameter_estimates
        yaml_dir = Path("../qsp-metadata-storage/to-review/parameter_estimates")
        template = Path("templates/parameter_metadata_template.yaml")

    # Validate inputs
    if not validation_dir.exists():
        print(f"Error: Validation directory not found: {validation_dir}")
        sys.exit(1)

    if not yaml_dir.exists():
        print(f"Error: YAML directory not found: {yaml_dir}")
        sys.exit(1)

    if not template.exists():
        print(f"Error: Template not found: {template}")
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

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(base_dir, storage_dir, api_key)

    # Print header
    print("\n" + "=" * 70)
    print("VALIDATION FIX WORKFLOW")
    print("=" * 70)
    print(f"Workflow type: {args.workflow_type}")
    print(f"Validation results: {validation_dir}")
    print(f"YAML directory: {yaml_dir}")
    print(f"Template: {template}")
    print(f"Mode: {'Immediate (Responses API)' if args.immediate else f'Batch API (timeout: {args.timeout}s)'}")
    print("=" * 70 + "\n")

    try:
        # Step 1: Create fix batch
        print_progress("Step 1: Creating fix batch...")
        creator = ValidationFixBatchCreator(base_dir)
        batch_file = creator.run(
            None,  # Use default output path
            validation_dir,
            yaml_dir,
            template
        )

        if not batch_file or not batch_file.exists():
            print("No validation errors found to fix. Exiting.")
            sys.exit(0)

        # Step 1.5: Extract sample prompt for review
        print("\nExtracting sample prompt for review...")
        prompt_file = extract_prompt_to_file(batch_file, base_dir, index=0)
        print(f"Sample prompt written to: {prompt_file}")
        print("\nPlease review the prompt to ensure it looks correct.")

        response = input("\nProceed with batch submission? [y/N]: ")
        if response.lower() != 'y':
            print("\nBatch submission cancelled.")
            print(f"Batch file saved to: {batch_file}")
            print(f"Sample prompt saved to: {prompt_file}")
            sys.exit(0)

        # Step 2 & 3: Process batch (immediate or batch API)
        if args.immediate:
            print_progress("\nStep 2: Processing requests immediately (Responses API)...")
            results_file = orchestrator.process_immediate(batch_file, progress_callback=print_progress)
            print_progress(f"✓ Requests completed: {results_file}")
        else:
            print_progress("\nStep 2: Uploading batch...")
            batch_id = orchestrator.upload_batch(batch_file, progress_callback=print_progress)
            print_progress(f"✓ Batch uploaded: {batch_id}")

            print_progress("\nStep 3: Monitoring batch completion...")
            results_file = orchestrator.monitor_batch(batch_id, timeout=args.timeout, progress_callback=print_progress)
            print_progress(f"✓ Batch completed: {results_file}")

        # Step 3.5: Check for existing changes and commit if needed
        print("\nStep 3.5: Checking for existing changes in qsp-metadata-storage...")
        orchestrator.check_and_commit_existing_changes(print_progress)

        # Step 4: Unpack results (overwrites originals)
        print("\nStep 4: Unpacking fixed YAMLs...")
        print("⚠  WARNING: This will overwrite files in:")
        print(f"   {yaml_dir}")
        print("   Original files are preserved in git history.")

        response = input("\nProceed with unpacking? [y/N]: ")
        if response.lower() != 'y':
            print("Unpacking cancelled. Results saved to:")
            print(f"  {results_file}")
            sys.exit(0)

        unpack_results(results_file, yaml_dir, template)
        print("✓ Results unpacked")

        # Step 5: Print summary and next steps
        print("\n" + "=" * 70)
        print("VALIDATION FIX SUMMARY")
        print("=" * 70)
        print("✓ Status: SUCCESS")
        if not args.immediate:
            print(f"✓ Batch ID: {batch_id}")
        print(f"✓ Fixed YAMLs written to: {yaml_dir}")
        print("\nNext steps:")
        print("  1. Re-run validation to verify fixes:")
        print(f"     python scripts/validate/run_all_validations.py {args.workflow_type}")
        print("\n  2. If validation passes, review and merge")
        print("  3. If failures persist, investigate manually")
        print("=" * 70)

        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Validation fix workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
