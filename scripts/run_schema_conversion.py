#!/usr/bin/env python3
"""
Automated schema conversion workflow CLI.

Scans metadata directories for files with outdated schemas,
converts them to latest versions, and commits to review branch.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

# Add lib directory to path
sys.path.insert(0, str(Path(__file__).parent / "lib"))

from schema_version_detector import SchemaVersionDetector
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


def print_summary(results: dict):
    """Print workflow summary."""
    print("\n" + "=" * 70)
    print("SCHEMA CONVERSION SUMMARY")
    print("=" * 70)

    if results["status"] == "success":
        print(f"✓ Status: SUCCESS")
        print(f"✓ Files converted: {results['file_count']}")
        print(f"✓ Duration: {results['duration_seconds']:.1f}s")
        print(f"✓ Review branch: {results['branch_name']}")

        if results.get('pushed'):
            print(f"✓ Pushed to origin/{results['branch_name']}")
        else:
            print(f"✓ Local branch created: {results['branch_name']}")

        print("\nNext steps:")
        print("  1. cd ../qsp-metadata-storage")
        print(f"  2. git checkout {results['branch_name']}")
        print("  3. Review converted files in to-review/")
        print("  4. Move approved files to replace originals")
        print("  5. Delete originals and commit")
        print("  6. Merge to main when approved")
    else:
        print(f"✗ Status: FAILED")
        print(f"✗ Error: {results.get('error', 'Unknown error')}")
        print(f"✗ Duration: {results['duration_seconds']:.1f}s")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Automated schema conversion for outdated metadata files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan and convert all outdated files
  python scripts/run_schema_conversion.py

  # Convert only parameters
  python scripts/run_schema_conversion.py --only parameter

  # Convert only test statistics
  python scripts/run_schema_conversion.py --only test_statistic

  # Scan without converting (dry run)
  python scripts/run_schema_conversion.py --dry-run

  # Convert with custom timeout
  python scripts/run_schema_conversion.py --timeout 7200

  # Convert locally without pushing
  python scripts/run_schema_conversion.py --no-push
        """
    )

    parser.add_argument(
        "--only",
        choices=["parameter", "test_statistic", "quick_estimate"],
        help="Only convert specific metadata type"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan for outdated files without converting"
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
        default="schema-conversion",
        help="Prefix for review branch name (default: schema-conversion)"
    )

    args = parser.parse_args()

    # Determine directories
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    storage_dir = base_dir.parent / "qsp-metadata-storage"

    if not storage_dir.exists():
        print(f"Error: qsp-metadata-storage not found at {storage_dir}")
        print("Expected: ../qsp-metadata-storage relative to qsp-llm-workflows")
        sys.exit(1)

    # Create detector
    detector = SchemaVersionDetector(base_dir, storage_dir)

    # Scan for outdated files
    print("\n" + "=" * 70)
    print("SCANNING FOR OUTDATED SCHEMAS")
    print("=" * 70)

    metadata_types = [args.only] if args.only else None
    scan_results = detector.scan_all_directories(metadata_types)

    # Print summary
    detector.print_summary(scan_results)

    # If dry run or no files found, exit
    total_files = sum(len(files) for files in scan_results.values())
    if total_files == 0:
        sys.exit(0)

    if args.dry_run:
        print("\nDry run complete. Use 'run_schema_conversion.py' without --dry-run to convert.")
        sys.exit(0)

    # Load API key
    try:
        api_key = load_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create orchestrator
    orchestrator = WorkflowOrchestrator(base_dir, storage_dir, api_key)

    # Print header
    print("\n" + "=" * 70)
    print(f"AUTOMATED SCHEMA CONVERSION WORKFLOW")
    print("=" * 70)
    print(f"Files to convert: {total_files}")
    print(f"Timeout: {args.timeout}s")
    print(f"Validation: {'Disabled' if args.skip_validation else 'Enabled'}")
    print(f"Push: {'Disabled' if args.no_push else 'Enabled'}")
    print("=" * 70 + "\n")

    # Run conversion workflow for each metadata type
    try:
        all_results = []

        for metadata_type, files in scan_results.items():
            if not files:
                continue

            print(f"\n{'='*70}")
            print(f"Converting {metadata_type.upper()}: {len(files)} files")
            print(f"{'='*70}\n")

            # Group files by version transition
            version_groups = {}
            for file_path, from_ver, to_ver in files:
                key = (from_ver, to_ver)
                if key not in version_groups:
                    version_groups[key] = []
                version_groups[key].append(file_path)

            # Convert each version group
            for (from_ver, to_ver), file_paths in version_groups.items():
                print(f"\nConverting {len(file_paths)} files: {from_ver} → {to_ver}")

                results = orchestrator.run_schema_conversion_workflow(
                    files_to_convert=file_paths,
                    metadata_type=metadata_type,
                    from_version=from_ver,
                    to_version=to_ver,
                    timeout=args.timeout,
                    skip_validation=args.skip_validation,
                    push=not args.no_push,
                    branch_prefix=args.branch_prefix,
                    progress_callback=print_progress
                )

                all_results.append(results)

        # Print final summary for all conversions
        print("\n" + "=" * 70)
        print("ALL CONVERSIONS COMPLETE")
        print("=" * 70)

        total_converted = sum(r.get('file_count', 0) for r in all_results)
        print(f"✓ Total files converted: {total_converted}")

        branches = [r.get('branch_name') for r in all_results if r.get('branch_name')]
        if branches:
            print(f"✓ Review branches created:")
            for branch in branches:
                print(f"    - {branch}")

        print("\nNext steps:")
        print("  1. cd ../qsp-metadata-storage")
        for branch in branches:
            print(f"  2. git checkout {branch}")
            print(f"  3. Review and approve conversions")
            print(f"  4. Merge to main")

        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
