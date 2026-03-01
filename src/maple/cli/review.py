#!/usr/bin/env python3
"""
CLI entry point for qsp-review command.

Provides scientific review of parameter extractions and test statistics
using Claude Code in headless mode.
"""

import argparse
import sys
from pathlib import Path

from maple.core.scientific_review import ScientificReviewer


def main():
    """Main entry point for qsp-review command."""
    parser = argparse.ArgumentParser(
        description="Review QSP extractions for scientific soundness using Claude Code",
        epilog="""
Examples:
    # Review a single file
    qsp-review parameter_estimates --file path/to/k_growth.yaml

    # Review all YAMLs in a directory (in parallel)
    qsp-review parameter_estimates --dir metadata-storage/to-review/parameter_estimates

    # Limit parallel workers
    qsp-review test_statistics --dir path/to/files --workers 2

This command uses Claude Code in headless mode to:
1. Review the extraction against a scientific soundness rubric (5 dimensions)
2. Recommend up to 2 low-complexity prompt improvements based on issues found
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of extraction to review",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Single YAML file to review",
    )
    group.add_argument(
        "--dir",
        type=str,
        help="Directory of YAML files to review (runs in parallel)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers when using --dir (default: 4)",
    )

    parser.add_argument(
        "--recommendations-file",
        type=str,
        default="review_recommendations.md",
        help="Path to save prompt improvement recommendations (default: review_recommendations.md)",
    )

    args = parser.parse_args()

    # Check for Claude Code CLI
    import shutil

    if not shutil.which("claude"):
        print("Error: Claude Code CLI not found.")
        print("Please install it from https://claude.ai/code")
        sys.exit(1)

    reviewer = ScientificReviewer(
        workflow_type=args.workflow_type,
        recommendations_file=args.recommendations_file,
    )

    if args.file:
        yaml_path = Path(args.file)
        if not yaml_path.exists():
            print(f"Error: File not found: {yaml_path}")
            sys.exit(1)
        success = reviewer.review_file(yaml_path)
        sys.exit(0 if success else 1)
    else:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)
        results = reviewer.review_directory(dir_path, max_workers=args.workers)
        # Exit with failure if any reviews failed
        all_passed = all(results.values())
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
