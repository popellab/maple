#!/usr/bin/env python3
"""
Enrich test statistic CSV with scenario context and validate code/units.

This script validates user-provided compute_test_statistic functions
and enriches the CSV with scenario context for LLM extraction.

Validation checks:
1. output_unit is a valid Pint unit string
2. model_output_code parses and has correct signature (time, species_dict, ureg)
3. Code executes with Pint-wrapped mock data
4. Output has correct unit dimensionality
"""

import argparse
import ast
import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pint
import yaml

from qsp_llm_workflows.core.unit_registry import create_unit_registry


class CodeUnitValidator:
    """
    Validates compute_test_statistic function code with Pint unit checking.

    This is used during CSV enrichment to validate user-provided code
    before sending to the LLM for literature extraction.
    """

    REQUIRED_FUNCTION_NAME = "compute_test_statistic"
    REQUIRED_PARAMS = ["time", "species_dict", "ureg"]

    def __init__(self, code: str, output_unit: str, species_units: dict[str, str]):
        """
        Initialize validator.

        Args:
            code: Python code defining compute_test_statistic
            output_unit: Expected output unit (Pint-parseable string)
            species_units: Mapping of species name -> unit string
        """
        self.code = code
        self.output_unit = output_unit
        self.species_units = species_units
        self.errors: list[str] = []
        self.ureg = create_unit_registry()

    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code fences if present."""
        if not code:
            return ""
        code = re.sub(r"^```python?\s*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)
        return code.strip()

    def validate(self) -> bool:
        """
        Run all validations.

        Returns:
            True if valid, False otherwise. Errors stored in self.errors.
        """
        self.errors = []
        code = self._clean_code_block(self.code)

        # 1. Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
            return False

        # 2. Find function definition
        func_def = self._find_function_def(tree)
        if func_def is None:
            self.errors.append(f"Code must define a function named '{self.REQUIRED_FUNCTION_NAME}'")
            return False

        # 3. Check signature
        if not self._validate_signature(func_def):
            return False

        # 4. Extract and validate species accessed
        accessed_species = self._extract_accessed_species(tree)
        if not accessed_species:
            self.errors.append("Code does not access any species from species_dict")
            return False

        missing_units = [s for s in accessed_species if s not in self.species_units]
        if missing_units:
            self.errors.append(
                f"Species accessed in code but not in species_units: {missing_units}"
            )
            return False

        # 5. Validate output_unit is parseable
        try:
            expected_unit = self.ureg.parse_expression(self.output_unit)
        except pint.UndefinedUnitError as e:
            self.errors.append(f"Invalid output_unit '{self.output_unit}': {e}")
            return False

        # 6. Execute with Pint and check output units
        if not self._validate_execution(code, accessed_species, expected_unit):
            return False

        return True

    def _find_function_def(self, tree: ast.AST) -> Optional[ast.FunctionDef]:
        """Find the compute_test_statistic function definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == self.REQUIRED_FUNCTION_NAME:
                return node
        return None

    def _validate_signature(self, func_def: ast.FunctionDef) -> bool:
        """Validate function has correct signature."""
        arg_names = [arg.arg for arg in func_def.args.args]
        if arg_names != self.REQUIRED_PARAMS:
            self.errors.append(
                f"Function must have signature ({', '.join(self.REQUIRED_PARAMS)}), "
                f"got ({', '.join(arg_names)})"
            )
            return False
        return True

    def _extract_accessed_species(self, tree: ast.AST) -> list[str]:
        """Extract species names accessed via species_dict['...']."""
        species = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == "species_dict":
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        species.append(node.slice.value)
        return list(set(species))

    def _validate_execution(
        self, code: str, accessed_species: list[str], expected_unit: pint.Unit
    ) -> bool:
        """Execute code with Pint-wrapped mock data and validate output."""
        try:
            # Create mock time array (in days)
            mock_time = np.linspace(0, 60, 100) * self.ureg.day

            # Create mock species_dict with Pint quantities
            mock_species_dict = {}
            for species in accessed_species:
                unit_str = self.species_units[species]
                unit = self.ureg.parse_expression(unit_str)
                mock_values = np.abs(np.sin(np.linspace(0, 2 * np.pi, 100)) + 1.5) * 100
                mock_species_dict[species] = mock_values * unit

            # Execute code
            local_namespace = {"np": np, "numpy": np}
            exec(code, local_namespace)

            func = local_namespace[self.REQUIRED_FUNCTION_NAME]
            result = func(mock_time, mock_species_dict, self.ureg)

            # Check result is a Pint Quantity
            if not isinstance(result, pint.Quantity):
                self.errors.append(
                    f"Function returned {type(result).__name__}, expected pint.Quantity. "
                    "Ensure the function returns a value with units attached."
                )
                return False

            # Check dimensionality matches
            if result.dimensionality != expected_unit.dimensionality:
                self.errors.append(
                    f"Output unit mismatch: got {result.units} "
                    f"(dimensionality: {result.dimensionality}), "
                    f"expected {self.output_unit} "
                    f"(dimensionality: {expected_unit.dimensionality})"
                )
                return False

            # Check for NaN or Inf
            if np.isnan(result.magnitude):
                self.errors.append("Function returned NaN")
                return False
            if np.isinf(result.magnitude):
                self.errors.append("Function returned Inf")
                return False

            return True

        except Exception as e:
            self.errors.append(f"Code execution failed: {e}")
            return False


def load_species_units(species_file: Path) -> dict[str, str]:
    """
    Load species units from a CSV or JSON file.

    Args:
        species_file: Path to species file (CSV from model export or JSON)

    Returns:
        Dict mapping species name -> unit string

    Raises:
        ValueError: If file format is not supported
    """
    if species_file.suffix == ".csv":
        df = pd.read_csv(species_file)
        species_units = {}
        for _, row in df.iterrows():
            name = row["Name"]
            units = row["Units"] if pd.notna(row["Units"]) else "dimensionless"
            compartment = row.get("Compartment", "")

            species_units[name] = units
            if pd.notna(compartment) and compartment:
                qualified_name = f"{compartment}.{name}"
                species_units[qualified_name] = units

        return species_units

    elif species_file.suffix == ".json":
        import json

        with open(species_file) as f:
            data = json.load(f)

        species_units = {}
        for name, info in data.items():
            if isinstance(info, dict):
                species_units[name] = info.get("units", "dimensionless")
            elif isinstance(info, str):
                species_units[name] = info

        return species_units

    else:
        raise ValueError(f"Unsupported species file format: {species_file.suffix}")


def enrich_test_statistic_csv(
    partial_csv: Path,
    scenario_yaml: Path,
    species_file: Path,
    output_path: Path,
) -> None:
    """
    Enrich test statistic CSV with scenario context and validate code/units.

    Args:
        partial_csv: CSV with test_statistic_id, output_unit, model_output_code
        scenario_yaml: YAML with scenario_context
        species_file: CSV or JSON with species units
        output_path: Output CSV path

    Raises:
        ValueError: If validation fails or required columns are missing
    """
    # Load scenario context
    with open(scenario_yaml, "r") as f:
        scenario = yaml.safe_load(f)

    scenario_context = scenario["scenario_context"].strip()
    context_hash = hashlib.md5(scenario_context.encode()).hexdigest()[:8]

    # Load species units
    species_units = load_species_units(species_file)
    print(f"Loaded units for {len(species_units)} species")

    # Load CSV
    df = pd.read_csv(partial_csv)

    # Check required columns
    required_cols = ["test_statistic_id", "output_unit", "model_output_code"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate each row
    validation_errors = []

    for idx, row in df.iterrows():
        test_stat_id = row["test_statistic_id"]
        print(f"Validating {test_stat_id}...")

        validator = CodeUnitValidator(
            code=row["model_output_code"],
            output_unit=row["output_unit"],
            species_units=species_units,
        )

        if validator.validate():
            print("  ✓ Valid")
        else:
            for err in validator.errors:
                validation_errors.append(f"{test_stat_id}: {err}")
                print(f"  ✗ {err}")

    if validation_errors:
        print(f"\nValidation failed with {len(validation_errors)} error(s)")
        raise ValueError(f"Found {len(validation_errors)} validation error(s)")

    # Add enriched columns
    df["scenario_context"] = scenario_context
    df["context_hash"] = context_hash

    # Reorder columns
    df = df[
        [
            "test_statistic_id",
            "output_unit",
            "model_output_code",
            "scenario_context",
            "context_hash",
        ]
    ]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nEnriched {len(df)} test statistics → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich test statistic CSV with scenario context and validate units",
        epilog="""
Example:
    python -m qsp_llm_workflows.prepare.enrich_test_statistic_csv \\
        input.csv scenario.yaml species.csv -o output.csv
        """,
    )
    parser.add_argument(
        "partial_csv",
        type=Path,
        help="CSV with test_statistic_id, output_unit, model_output_code",
    )
    parser.add_argument(
        "scenario_yaml",
        type=Path,
        help="Scenario YAML with scenario_context field",
    )
    parser.add_argument(
        "species_file",
        type=Path,
        help="Species units file (CSV from model export or JSON)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Validate inputs exist
    if not args.partial_csv.exists():
        print(f"Error: Input file not found: {args.partial_csv}", file=sys.stderr)
        sys.exit(1)

    if not args.scenario_yaml.exists():
        print(f"Error: Scenario file not found: {args.scenario_yaml}", file=sys.stderr)
        sys.exit(1)

    if not args.species_file.exists():
        print(f"Error: Species file not found: {args.species_file}", file=sys.stderr)
        sys.exit(1)

    try:
        enrich_test_statistic_csv(
            args.partial_csv,
            args.scenario_yaml,
            args.species_file,
            args.output,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
