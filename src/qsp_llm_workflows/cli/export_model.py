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
        description="Export model definitions from SimBiology MATLAB model",
        epilog="""
Examples:
    qsp-export-model --matlab-model ../qspio-pdac/immune_oncology_model_PDAC.m --output model_defs.json
        """
    )

    parser.add_argument(
        "--matlab-model",
        required=True,
        type=Path,
        help="Path to MATLAB model file (e.g., immune_oncology_model_PDAC.m)"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file for model definitions"
    )

    args = parser.parse_args()

    # Validate input
    if not args.matlab_model.exists():
        print(f"Error: MATLAB model file not found: {args.matlab_model}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Exporting model definitions from {args.matlab_model}...")

        exporter = ModelDefinitionExporter(str(args.matlab_model))
        exporter.export_to_json(str(args.output))

        print(f"✓ Model definitions exported to {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
