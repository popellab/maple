#!/usr/bin/env python3
"""
Validate a YAML file against the SubmodelTarget schema.

Usage:
    python scripts/validate_submodel_target.py --model-structure path/to/model_structure.json path/to/target.yaml

    # Or set MODEL_STRUCTURE_PATH environment variable:
    export MODEL_STRUCTURE_PATH=path/to/model_structure.json
    python scripts/validate_submodel_target.py path/to/target.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration.submodel_target import (
    SubmodelTarget,
)
from qsp_llm_workflows.core.model_structure import ModelStructure


def _load_reference_db(yaml_path: str) -> dict:
    """Load reference values from YAML and return name -> value mapping."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return {v["name"]: float(v["value"]) for v in data["values"]}


def validate_yaml(
    yaml_path: str,
    model_structure: ModelStructure,
    reference_db: dict | None = None,
) -> bool:
    """
    Validate a YAML file against SubmodelTarget schema.

    Args:
        yaml_path: Path to the YAML file
        model_structure: ModelStructure instance for unit validation
        reference_db: Reference values database (optional)

    Returns:
        True if valid, False otherwise
    """
    path = Path(yaml_path)

    if not path.exists():
        print(f"Error: File not found: {yaml_path}")
        return False

    # Load YAML
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax:\n{e}")
        return False

    # Build validation context
    context = {"model_structure": model_structure}
    if reference_db is not None:
        context["reference_db"] = reference_db

    # Validate against model with model_structure context
    try:
        target = SubmodelTarget.model_validate(
            data,
            context=context,
        )
        print(f"✓ Valid: {path.name}")
        print(f"  target_id: {target.target_id}")
        print(f"  inputs: {len(target.inputs)}")
        print(f"  parameters: {[p.name for p in target.calibration.parameters]}")
        print(f"  model type: {target.calibration.model.type}")
        return True
    except ValidationError as e:
        print(f"✗ Validation failed: {path.name}\n")
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            print(f"  [{loc}]")
            print(f"    {error['msg']}")
            if error.get("input"):
                input_str = str(error["input"])[:80]
                print(f"    Input: {input_str}{'...' if len(str(error['input'])) > 80 else ''}")
            print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate SubmodelTarget YAML files",
        epilog="Model structure is required for unit validation against the QSP model.",
    )
    parser.add_argument(
        "--model-structure",
        type=str,
        default=os.environ.get("MODEL_STRUCTURE_PATH"),
        help="Path to model_structure.json (or set MODEL_STRUCTURE_PATH env var)",
    )
    parser.add_argument(
        "--reference-values",
        type=str,
        default=os.environ.get("REFERENCE_VALUES_PATH"),
        help="Path to reference_values.yaml (or set REFERENCE_VALUES_PATH env var). "
        "Auto-discovered next to model_structure.json if not specified.",
    )
    parser.add_argument(
        "yaml_files",
        nargs="+",
        help="YAML file(s) to validate",
    )
    args = parser.parse_args()

    # Require model_structure
    if not args.model_structure:
        print("Error: --model-structure is required (or set MODEL_STRUCTURE_PATH env var)")
        print("This is needed to validate parameter units against the QSP model.")
        sys.exit(1)

    model_structure_path = Path(args.model_structure)
    if not model_structure_path.exists():
        print(f"Error: Model structure file not found: {args.model_structure}")
        sys.exit(1)

    # Load model structure
    try:
        model_structure = ModelStructure.from_json(str(model_structure_path))
        print(f"Loaded model structure: {len(model_structure.parameters)} parameters")
        print()
    except Exception as e:
        print(f"Error loading model structure: {e}")
        sys.exit(1)

    # Load reference database (auto-discover next to model_structure if not specified)
    reference_db = None
    ref_path = args.reference_values
    if not ref_path:
        auto_path = model_structure_path.parent / "reference_values.yaml"
        if auto_path.exists():
            ref_path = str(auto_path)

    if ref_path:
        try:
            reference_db = _load_reference_db(ref_path)
            print(f"Loaded reference database: {len(reference_db)} values")
            print()
        except Exception as e:
            print(f"Warning: Could not load reference database: {e}")
            print()

    # Validate each YAML file
    results = []
    for yaml_path in args.yaml_files:
        valid = validate_yaml(yaml_path, model_structure, reference_db=reference_db)
        results.append((yaml_path, valid))
        if len(args.yaml_files) > 1:
            print()

    # Summary for multiple files
    if len(results) > 1:
        print("-" * 40)
        valid_count = sum(1 for _, v in results if v)
        print(f"Summary: {valid_count}/{len(results)} files valid")

    # Exit with error if any failed
    sys.exit(0 if all(v for _, v in results) else 1)


if __name__ == "__main__":
    main()
