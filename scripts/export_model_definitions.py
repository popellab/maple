#!/usr/bin/env python3
"""
Export model definitions from SimBiology MATLAB model.

Usage:
    python scripts/export_model_definitions.py \\
        --matlab-model ../qspio-pdac/immune_oncology_model_PDAC.m \\
        --output batch_jobs/input_data/model_definitions.json
"""

import sys
import json
import argparse
from pathlib import Path

# Add lib directory to path
lib_dir = Path(__file__).parent / "lib"
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from model_definition_exporter import ModelDefinitionExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export model definitions from SimBiology MATLAB model"
    )
    parser.add_argument(
        "--matlab-model",
        required=True,
        help="Path to MATLAB model file (e.g., ../qspio-pdac/immune_oncology_model_PDAC.m)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path (e.g., batch_jobs/input_data/model_definitions.json)"
    )

    args = parser.parse_args()

    # Resolve paths
    matlab_model = Path(args.matlab_model).resolve()
    output_file = Path(args.output).resolve()

    # Validate MATLAB model exists
    if not matlab_model.exists():
        print(f"Error: MATLAB model file not found: {matlab_model}")
        sys.exit(1)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL DEFINITION EXPORT")
    print("=" * 70)
    print(f"MATLAB model: {matlab_model}")
    print(f"Output file:  {output_file}")
    print("=" * 70 + "\n")

    # Export definitions
    try:
        exporter = ModelDefinitionExporter(matlab_model)
        definitions = exporter.export_definitions()

        # Write to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(definitions, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Successfully exported {len(definitions)} parameter definitions")
        print(f"✓ Saved to: {output_file}")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
