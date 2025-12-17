#!/usr/bin/env python3
"""
CLI entry point for qsp-review command.

Provides scientific review of parameter extractions and test statistics
using Claude Code in headless mode.
"""

import argparse
import sys
from pathlib import Path

from qsp_llm_workflows.core.scientific_review import ScientificReviewer


def main():
    """Main entry point for qsp-review command."""
    parser = argparse.ArgumentParser(
        description="Review QSP extractions for scientific soundness using Claude Code",
        epilog="""
Examples:
    qsp-review parameter_estimates --file metadata-storage/to-review/parameter_estimates/k_growth.yaml
    qsp-review test_statistics --file metadata-storage/to-review/test_statistics/tumor_volume.yaml
    qsp-review test_statistics --file path/to/file.yaml --species-units-file species_units.json

This command uses Claude Code in headless mode to:
1. Review the extraction against a scientific soundness rubric
2. Run automated validation and suggest intelligent fixes
3. Recommend prompt improvements based on patterns observed
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of extraction to review",
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="YAML file to review",
    )

    parser.add_argument(
        "--species-units-file",
        type=str,
        default=None,
        help="Path to species_units.json for model output code validation (test statistics only)",
    )

    parser.add_argument(
        "--recommendations-file",
        type=str,
        default="review_recommendations.md",
        help="Path to save prompt improvement recommendations (default: review_recommendations.md)",
    )

    args = parser.parse_args()

    yaml_path = Path(args.file)
    if not yaml_path.exists():
        print(f"Error: File not found: {yaml_path}")
        sys.exit(1)

    # Check for Claude Code CLI
    import shutil

    if not shutil.which("claude"):
        print("Error: Claude Code CLI not found.")
        print("Please install it from https://claude.ai/code")
        sys.exit(1)

    reviewer = ScientificReviewer(
        workflow_type=args.workflow_type,
        species_units_file=args.species_units_file,
        recommendations_file=args.recommendations_file,
    )

    success = reviewer.review_file(yaml_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
