#!/usr/bin/env python3
"""
CLI wrapper for model structure export.

Entry point: qsp-export-model
"""
import argparse
import sys
from pathlib import Path

from maple.core.model_structure_exporter import ModelStructureExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export model structure from SimBiology model",
        epilog="""
Examples:
    # Export model structure (for LLM query tools and validation)
    qsp-export-model --matlab-model model.m --output model_structure.json

    # Export from SimBiology project
    qsp-export-model --simbiology-project model.sbproj --output model_structure.json
        """,
    )

    # Create mutually exclusive group for model source
    model_group = parser.add_mutually_exclusive_group(required=True)

    model_group.add_argument(
        "--matlab-model",
        type=Path,
        help="Path to MATLAB model script (e.g., your_model_file.m)",
    )

    model_group.add_argument(
        "--simbiology-project",
        type=Path,
        help="Path to SimBiology project file (e.g., your_model.sbproj)",
    )

    parser.add_argument(
        "--output", required=True, type=Path, help="Output JSON file for model structure"
    )

    args = parser.parse_args()

    # Determine model file and type
    if args.matlab_model:
        model_file = args.matlab_model
        model_type = "matlab_script"
    else:
        model_file = args.simbiology_project
        model_type = "simbiology_project"

    # Validate input
    if not model_file.exists():
        print(f"Error: Model file not found: {model_file}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Exporting model structure from {model_file}...")

        exporter = ModelStructureExporter(str(model_file), model_type=model_type)
        exporter.export_to_json(str(args.output))

        print(f"✓ Model structure exported to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
