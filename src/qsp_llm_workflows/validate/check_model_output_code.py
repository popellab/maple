#!/usr/bin/env python3
"""
Validate model_output.code for test statistics.

Validates that the compute_test_statistic function:
1. Is properly defined with the correct signature
2. Accepts time (numpy array) and species_dict (dict) arguments
3. Returns a scalar float value when called with mock data

This validation only applies to test statistics (not parameter estimates).
"""
import re
import numpy as np

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


class ModelOutputCodeValidator(Validator):
    """
    Validate model_output.code for test statistics.

    Checks that compute_test_statistic(time, species_dict) -> float is properly defined.
    """

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)

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
        """Extract model_output.code from test statistic YAML."""
        model_output = data.get("model_output")
        if not model_output:
            return None

        code = model_output.get("code")
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
        Validate model_output.code for a single test statistic file.

        Returns:
            (is_valid, message) tuple
        """
        # Skip non-test-statistic files
        if "test_statistic_id" not in data:
            return (True, "Skipped (not a test statistic)")

        # Extract model_output.code
        code = self._extract_model_output_code(data)
        if not code:
            return (False, "Missing model_output.code field")

        # Check that function is defined
        if "def compute_test_statistic" not in code:
            return (False, "Function 'compute_test_statistic' not defined in code")

        # Check function signature has correct arguments
        signature_pattern = r"def\s+compute_test_statistic\s*\(\s*(\w+)\s*[,:]\s*.*?(\w+)\s*[,:)]"
        match = re.search(signature_pattern, code)
        if not match:
            return (
                False,
                "Function signature doesn't match expected pattern: "
                "compute_test_statistic(time, species_dict)",
            )

        arg1, arg2 = match.groups()
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

        # Get required species for mock data
        required_species = self._get_required_species(data)
        if not required_species:
            return (False, "No required_species defined - cannot validate code execution")

        # Try executing the code with mock data
        try:
            # Create mock time array
            mock_time = np.linspace(0, 14, 100)

            # Create mock species_dict with required species
            mock_species_dict = {}
            for species in required_species:
                # Generate plausible mock data (positive values with some variation)
                mock_species_dict[species] = np.abs(np.sin(mock_time) + 1) * 100

            # Execute the code to define the function
            local_namespace = {"np": np, "numpy": np}
            exec(code, local_namespace)

            # Check function was defined
            if "compute_test_statistic" not in local_namespace:
                return (False, "Function 'compute_test_statistic' not found after execution")

            func = local_namespace["compute_test_statistic"]

            # Call the function with mock data
            result = func(mock_time, mock_species_dict)

            # Validate return type is scalar
            if isinstance(result, np.ndarray):
                if result.ndim == 0:
                    # 0-dimensional array is ok, extract scalar
                    result = float(result)
                else:
                    return (
                        False,
                        f"Function returned array with shape {result.shape}, expected scalar",
                    )

            if not isinstance(result, (int, float, np.floating, np.integer)):
                return (
                    False,
                    f"Function returned {type(result).__name__}, expected scalar float",
                )

            # Check for NaN or Inf
            if np.isnan(result):
                return (False, "Function returned NaN")
            if np.isinf(result):
                return (False, "Function returned Inf")

            return (True, f"Valid (returned {result:.6g})")

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
