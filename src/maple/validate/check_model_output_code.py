#!/usr/bin/env python3
"""
Validate model_output_code for test statistics.

Validates that the compute_test_statistic function:
1. Is properly defined with the correct signature (time, species_dict, ureg)
2. Accepts Pint-wrapped time array and species_dict arguments
3. Returns a Pint Quantity when called with mock data

This validation only applies to test statistics (not parameter estimates).
"""
import json
import re
from pathlib import Path

import numpy as np
import pint

from maple.validate.validator import Validator
from maple.core.validation_utils import load_yaml_directory, ValidationReport
from maple.core.unit_registry import ureg


def load_species_units(species_units_file: str | Path) -> dict[str, str]:
    """
    Load species units from JSON file.

    Args:
        species_units_file: Path to species_units.json (from qsp-export-model)

    Returns:
        Dict mapping species name -> unit string
        Example: {'V_T.CD8': 'cell', 'TGFb': 'nanomolarity'}
    """
    with open(species_units_file) as f:
        return json.load(f)


class ModelOutputCodeValidator(Validator):
    """
    Validate model_output_code for test statistics.

    Checks that compute_test_statistic(time, species_dict, ureg) -> pint.Quantity is properly defined.

    If species_units_file is provided, mock data will use correct units from the model.
    Otherwise, falls back to dimensionless units (may cause validation failures for
    code that performs unit conversions).
    """

    def __init__(self, data_dir: str, species_units_file: str | None = None, **kwargs):
        """
        Initialize validator.

        Args:
            data_dir: Directory containing test statistic YAML files
            species_units_file: Optional path to species_units.json (from qsp-export-model).
                              If provided, mock data will use correct units from the model.
                              If not provided, falls back to dimensionless units.
        """
        super().__init__(data_dir, **kwargs)
        self.ureg = ureg
        self.species_units: dict[str, str] = {}

        if species_units_file:
            self.species_units = load_species_units(species_units_file)
            print(f"Loaded units for {len(self.species_units)} species from {species_units_file}")

    @property
    def name(self) -> str:
        return "Model Output Code Validation"

    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code fences if present."""
        if not code:
            return None

        # Remove ```python and ``` markers
        code = re.sub(r"^```python?\s*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)

        return code.strip()

    def _extract_model_output_code(self, data: dict) -> str | None:
        """Extract model_output_code from test statistic YAML header field."""
        code = data.get("model_output_code")
        if not code:
            return None

        return self._clean_code_block(code)

    def _get_required_species(self, data: dict) -> list[str]:
        """Get required_species from test statistic YAML."""
        required_species = data.get("required_species", [])
        if isinstance(required_species, str):
            # Handle comma-separated string format
            return [s.strip() for s in required_species.split(",") if s.strip()]
        return required_species

    def validate_file(self, _filepath: str, data: dict) -> tuple[bool, str]:
        """
        Validate model_output_code for a single test statistic file.

        Returns:
            (is_valid, message) tuple
        """
        # Skip non-test-statistic files
        if "test_statistic_id" not in data:
            return (True, "Skipped (not a test statistic)")

        # Extract model_output_code
        code = self._extract_model_output_code(data)
        if not code:
            return (False, "Missing model_output_code field")

        # Check that function is defined
        if "def compute_test_statistic" not in code:
            return (False, "Function 'compute_test_statistic' not defined in code")

        # Check function signature has correct arguments (time, species_dict, ureg)
        signature_pattern = (
            r"def\s+compute_test_statistic\s*\(\s*(\w+)\s*[,:]\s*.*?(\w+)\s*[,:]\s*.*?(\w+)\s*[,:)]"
        )
        match = re.search(signature_pattern, code)
        if not match:
            return (
                False,
                "Function signature doesn't match expected pattern: "
                "compute_test_statistic(time, species_dict, ureg)",
            )

        arg1, arg2, arg3 = match.groups()
        if arg1 != "time":
            return (
                False,
                f"First argument should be 'time', got '{arg1}'",
            )
        if arg2 != "species_dict":
            return (
                False,
                f"Second argument should be 'species_dict', got '{arg2}'",
            )
        if arg3 != "ureg":
            return (
                False,
                f"Third argument should be 'ureg', got '{arg3}'",
            )

        # Get required species for mock data
        required_species = self._get_required_species(data)
        if not required_species:
            return (False, "No required_species defined - cannot validate code execution")

        # Get expected output unit (required field)
        output_unit = data.get("output_unit")
        if not output_unit:
            return (False, "Missing required output_unit field")

        # Try executing the code with mock data
        try:
            # Create mock time array with Pint units (days)
            # Use 0-60 days to cover common time ranges in test statistics
            mock_time = np.linspace(0, 60, 200) * self.ureg.day

            # Create mock species_dict with required species as Pint quantities
            mock_species_dict = {}
            for species in required_species:
                # Look up unit from species_units if available
                unit_str = self.species_units.get(species, "dimensionless")
                try:
                    unit = self.ureg.parse_expression(unit_str)
                except pint.UndefinedUnitError:
                    # Fall back to dimensionless if unit string is invalid
                    unit = self.ureg.dimensionless

                # Species starting with V_ are compartment time-series, others are scalars
                if species.startswith("V_"):
                    # Generate plausible mock data (positive values with some variation)
                    mock_data = np.abs(np.sin(mock_time.magnitude / 10) + 1.5) * 100
                    mock_species_dict[species] = mock_data * unit
                else:
                    # Scalar parameter (e.g., initial_tumour_diameter)
                    mock_species_dict[species] = 10.0 * unit

            # Execute the code to define the function
            local_namespace = {"np": np, "numpy": np}
            exec(code, local_namespace)

            # Check function was defined
            if "compute_test_statistic" not in local_namespace:
                return (False, "Function 'compute_test_statistic' not found after execution")

            func = local_namespace["compute_test_statistic"]

            # Call the function with mock data and ureg
            result = func(mock_time, mock_species_dict, self.ureg)

            # Validate return type is Pint Quantity
            if not isinstance(result, pint.Quantity):
                return (
                    False,
                    f"Function returned {type(result).__name__}, expected pint.Quantity",
                )

            # Check for NaN or Inf in magnitude
            if np.isnan(result.magnitude):
                return (False, "Function returned NaN")
            if np.isinf(result.magnitude):
                return (False, "Function returned Inf")

            # Validate output unit dimensionality matches declared output_unit
            try:
                expected_unit = self.ureg.parse_expression(output_unit)
                if result.dimensionality != expected_unit.dimensionality:
                    return (
                        False,
                        f"Unit mismatch: function returned {result.units:~} "
                        f"but output_unit declares {expected_unit:~}",
                    )
            except pint.UndefinedUnitError:
                return (False, f"Cannot parse declared output_unit: {output_unit}")

            # Format result for display
            result_str = f"{result.magnitude:.6g} {result.units:~}"

            return (True, f"Valid (returned {result_str})")

        except Exception as e:
            return (False, f"Code execution failed: {str(e)}")

    def validate(self) -> ValidationReport:
        """Run validation on all YAML files in directory."""
        report = ValidationReport(self.name)

        # Load all YAML files
        yaml_files = load_yaml_directory(self.data_dir)

        if not yaml_files:
            report.add_warning("NO_FILES", "No YAML files found in directory")
            return report

        for file_info in yaml_files:
            filepath = file_info["filepath"]
            filename = file_info["filename"]
            data = file_info["data"]

            # Skip non-test-statistic files silently
            if "test_statistic_id" not in data:
                continue

            is_valid, message = self.validate_file(filepath, data)

            if is_valid:
                report.add_pass(filename, message)
            else:
                report.add_fail(filename, message)

        return report
