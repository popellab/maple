#!/usr/bin/env python3
"""
Validate calibration target YAML files against the CalibrationTarget schema.

Usage:
    # Validate specific files:
    python scripts/validate_calibration_target.py \
        --species-units path/to/species_units.json \
        target1.yaml target2.yaml

    # Validate all YAMLs in a directory:
    python scripts/validate_calibration_target.py \
        --species-units path/to/species_units.json \
        calibration_targets/

    # Auto-discover species_units.json and reference_values.yaml:
    export SPECIES_UNITS_PATH=path/to/species_units.json
    python scripts/validate_calibration_target.py calibration_targets/

    # Skip DOI validation (faster, for offline use):
    python scripts/validate_calibration_target.py --skip-doi calibration_targets/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from qsp_llm_workflows.core.calibration import CalibrationTarget


def _load_species_units(json_path: str) -> dict:
    """Load species_units.json and convert to validator-expected format."""
    with open(json_path) as f:
        raw = json.load(f)
    # CalibrationTarget expects {species_name: {"units": str, "description": str}}
    return {name: {"units": units, "description": ""} for name, units in raw.items()}


def _load_reference_db(yaml_path: str) -> dict:
    """Load reference values from YAML and return name -> value mapping."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return {v["name"]: float(v["value"]) for v in data["values"]}


def validate_yaml(
    yaml_path: str,
    species_units: dict,
    reference_db: dict | None = None,
    skip_doi: bool = False,
) -> bool:
    """
    Validate a YAML file against CalibrationTarget schema.

    Returns:
        True if valid, False otherwise
    """
    path = Path(yaml_path)

    if not path.exists():
        print(f"Error: File not found: {yaml_path}")
        return False

    if path.suffix not in (".yaml", ".yml"):
        return True  # Skip non-YAML files silently

    # Load YAML
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"✗ YAML parse error: {path.name}\n  {e}")
        return False

    if data is None:
        print(f"✗ Empty file: {path.name}")
        return False

    # Build validation context
    context = {"species_units": species_units}
    if reference_db is not None:
        context["reference_db"] = reference_db
    if skip_doi:
        context["skip_doi_validation"] = True

    # Validate
    try:
        target = CalibrationTarget.model_validate(data, context=context)
        target_id = getattr(target, "calibration_target_id", "unknown")
        obs = target.observable
        print(f"✓ {path.name}")
        print(f"    target_id: {target_id}")
        print(f"    units: {obs.units}, support: {obs.support}")
        print(f"    species: {obs.species}")
        if obs.experimental_denominator:
            print(f"    exp_denom: {obs.experimental_denominator[:80]}")
        if obs.unmodeled_denominator_components:
            print(f"    unmodeled: {obs.unmodeled_denominator_components[:80].strip()}")
        return True
    except ValidationError as e:
        print(f"✗ {path.name}\n")
        for error in e.errors():
            loc = " → ".join(str(x) for x in error["loc"])
            print(f"  [{loc}]")
            print(f"    {error['msg']}")
            if error.get("input"):
                input_str = str(error["input"])[:100]
                print(f"    Input: {input_str}{'...' if len(str(error['input'])) > 100 else ''}")
            print()
        return False


def collect_yaml_files(paths: list[str]) -> list[str]:
    """Expand directories to YAML files, keep explicit file paths."""
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(str(f) for f in path.rglob("*.yaml")))
            files.extend(sorted(str(f) for f in path.rglob("*.yml")))
        elif path.is_file():
            files.append(str(path))
        else:
            print(f"Warning: {p} is not a file or directory, skipping")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Validate CalibrationTarget YAML files against the Pydantic schema.",
        epilog="Requires species_units.json for unit validation. "
        "Reference values are auto-discovered if adjacent to species_units.",
    )
    parser.add_argument(
        "--species-units",
        type=str,
        default=os.environ.get("SPECIES_UNITS_PATH"),
        help="Path to species_units.json (or set SPECIES_UNITS_PATH env var)",
    )
    parser.add_argument(
        "--reference-values",
        type=str,
        default=os.environ.get("REFERENCE_VALUES_PATH"),
        help="Path to reference_values.yaml (or set REFERENCE_VALUES_PATH env var)",
    )
    parser.add_argument(
        "--skip-doi",
        action="store_true",
        help="Skip DOI resolution checks (faster, for offline use)",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="YAML file(s) or directory(ies) to validate",
    )
    args = parser.parse_args()

    # Require species_units
    if not args.species_units:
        print("Error: --species-units is required (or set SPECIES_UNITS_PATH env var)")
        sys.exit(1)

    species_units_path = Path(args.species_units)
    if not species_units_path.exists():
        print(f"Error: species_units file not found: {args.species_units}")
        sys.exit(1)

    # Load species units
    try:
        species_units = _load_species_units(str(species_units_path))
        print(f"Loaded species_units: {len(species_units)} species")
    except Exception as e:
        print(f"Error loading species_units: {e}")
        sys.exit(1)

    # Load reference database (auto-discover next to species_units if not specified)
    reference_db = None
    ref_path = args.reference_values
    if not ref_path:
        auto_path = species_units_path.parent / "reference_values.yaml"
        if auto_path.exists():
            ref_path = str(auto_path)

    if ref_path:
        try:
            reference_db = _load_reference_db(ref_path)
            print(f"Loaded reference_db: {len(reference_db)} values")
        except Exception as e:
            print(f"Warning: Could not load reference database: {e}")

    print()

    # Collect and validate files
    yaml_files = collect_yaml_files(args.paths)
    if not yaml_files:
        print("No YAML files found")
        sys.exit(1)

    results = []
    for yaml_path in yaml_files:
        valid = validate_yaml(
            yaml_path, species_units, reference_db=reference_db, skip_doi=args.skip_doi
        )
        results.append((yaml_path, valid))
        print()

    # Summary
    valid_count = sum(1 for _, v in results if v)
    total = len(results)
    print("-" * 50)
    if valid_count == total:
        print(f"All {total} files passed validation")
    else:
        print(f"{valid_count}/{total} files passed, {total - valid_count} failed")

    sys.exit(0 if all(v for _, v in results) else 1)


if __name__ == "__main__":
    main()
