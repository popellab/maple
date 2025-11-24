#!/usr/bin/env python3
"""
CLI wrapper for model definition export.

Entry point: qsp-export-model
"""
import argparse
import sys
from pathlib import Path

from qsp_llm_workflows.core.model_definition_exporter import ModelDefinitionExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export model definitions from SimBiology model",
        epilog="""
Examples:
    # Export from MATLAB script
    qsp-export-model --matlab-model ../your-model-repo/scripts/your_model_file.m --output model_defs.json

    # Export from SimBiology project
    qsp-export-model --simbiology-project ../your-model-repo/models/your_model.sbproj --output model_defs.json
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
        "--output", required=True, type=Path, help="Output JSON file for model definitions"
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
        print(f"Exporting model definitions from {model_file}...")

        exporter = ModelDefinitionExporter(str(model_file), model_type=model_type)
        exporter.export_to_json(str(args.output))

        print(f"✓ Model definitions exported to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
