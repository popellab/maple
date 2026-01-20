#!/usr/bin/env python3
"""
Validate a YAML file against the SubmodelTarget schema.

Usage:
    python scripts/validate_submodel_target.py path/to/target.yaml
"""

import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration.submodel_target import (
    SubmodelTarget,
)


def validate_yaml(yaml_path: str) -> bool:
    """
    Validate a YAML file against SubmodelTarget schema.

    Args:
        yaml_path: Path to the YAML file

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

    # Validate against model
    try:
        target = SubmodelTarget.model_validate(data)
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
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    yaml_paths = sys.argv[1:]
    results = []

    for yaml_path in yaml_paths:
        valid = validate_yaml(yaml_path)
        results.append((yaml_path, valid))
        if len(yaml_paths) > 1:
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
