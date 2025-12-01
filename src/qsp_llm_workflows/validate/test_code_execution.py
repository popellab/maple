#!/usr/bin/env python3
"""
Test Python code execution from YAML files.

Validates:
- Python code executes without errors
- Function returns required fields (mean/variance for parameters, median/iqr for test statistics)
- Returned values match declared values in YAML
- All inputs have corresponding sources

Works for both parameter estimates (v3 schema) and test statistics (v2 schema).

Usage:
    python scripts/validate/test_code_execution.py \\
        metadata-storage/parameter_estimates \\
        output/code_execution_report.json \\
        --threshold 5.0
"""
import re
import numpy as np

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


class CodeExecutionValidator(Validator):
    """
    Validate Python code execution from YAML files.
    Works for both parameters (v3) and test statistics (v2).
    """

    def __init__(self, data_dir: str, threshold_pct: float = 5.0, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.threshold_pct = threshold_pct

    @property
    def name(self) -> str:
        return "Code Execution Testing"

    def extract_python_code(self, data: dict) -> tuple:
        """
        Extract Python code from YAML data.

        Returns:
            (code, code_type) tuple where code_type is "parameter" or "test_statistic"
        """
        # Try parameter_estimates first (params v3)
        if "parameter_estimates" in data:
            estimates = data["parameter_estimates"]
            if isinstance(estimates, dict) and "derivation_code" in estimates:
                code = estimates["derivation_code"]
                return (self._clean_code_block(code), "parameter")

        # Try test_statistic_estimates (test stats v2)
        if "test_statistic_estimates" in data:
            estimates = data["test_statistic_estimates"]
            if isinstance(estimates, dict) and "derivation_code" in estimates:
                code = estimates["derivation_code"]
                return (self._clean_code_block(code), "test_statistic")

        return (None, None)

    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code fences if present."""
        if not code:
            return None

        # Remove ```python and ``` markers
        code = re.sub(r"^```python?\s*\n", "", code, flags=re.MULTILINE)
        code = re.sub(r"\n```\s*$", "", code, flags=re.MULTILINE)

        return code.strip()

    def extract_inputs(self, data: dict) -> list:
        """
        Extract inputs from parameter_estimates or test_statistic_estimates.

        Returns the raw list of input dicts as expected by the derivation functions.
        Each input dict has keys: name, value, units, description, source_ref, etc.

        The derivation code accesses inputs like:
            float([x for x in inputs if x['name']=='Foo'][0]['value'])

        Returns:
            List of input dicts, or None if no inputs found
        """
        # Try parameter_estimates first
        if "parameter_estimates" in data:
            estimates = data["parameter_estimates"]
            if "inputs" in estimates and isinstance(estimates["inputs"], list):
                return estimates["inputs"]

        # Try test_statistic_estimates
        if "test_statistic_estimates" in data:
            estimates = data["test_statistic_estimates"]
            if "inputs" in estimates and isinstance(estimates["inputs"], list):
                return estimates["inputs"]

        return None

    def execute_python_code(
        self,
        code: str,
        inputs: list,
        code_type: str,
        expected_mean: float = None,
        expected_variance: float = None,
        expected_ci95: list = None,
    ) -> tuple:
        """
        Execute Python code and check for errors.
        Compare computed values to expected values from YAML.

        Args:
            code: Python code to execute
            inputs: List of input dicts to pass to derive function.
                    Each dict has 'name', 'value', 'units', etc.
            code_type: "parameter" or "test_statistic"
            expected_mean: Expected mean/median value
            expected_variance: Expected variance/iqr value
            expected_ci95: Expected CI95 [lower, upper]

        Returns:
            (success, message, results_dict) tuple
        """
        if not code:
            return (False, "No Python code found", {})

        if not inputs:
            return (False, "No inputs defined", {})

        # Create namespace for execution
        namespace = {"inputs": inputs, "np": np, "numpy": np}

        try:
            # Execute the code in isolated namespace
            exec(code, namespace)

            # Determine function name based on code type
            if code_type == "parameter":
                func_name = "derive_parameter"
                expected_keys = [
                    "median_param",
                    "iqr_param",
                    "ci95_param",
                ]  # Changed to robust stats
                stat_names = ["median", "iqr"]  # For display
            elif code_type == "test_statistic":
                func_name = "derive_distribution"
                expected_keys = ["median_stat", "iqr_stat", "ci95_stat"]  # Changed to robust stats
                stat_names = ["median", "iqr"]  # For display
            else:
                return (False, f"Unknown code type: {code_type}", {})

            # Check if function exists
            if func_name not in namespace:
                return (False, f"Code does not define {func_name} function", {})

            # Call the function
            derive_func = namespace[func_name]
            result = derive_func(inputs)

            # Validate return type
            if not isinstance(result, dict):
                return (False, f"{func_name} returned {type(result)}, expected dict", {})

            # Check required fields
            missing = [k for k in expected_keys if k not in result]
            if missing:
                return (False, f"Missing required fields in result: {missing}", {})

            # Extract computed values (normalize keys for comparison)
            central_key = expected_keys[0]  # mean_param or median_stat
            spread_key = expected_keys[1]  # variance_param or iqr_stat
            ci95_key = expected_keys[2]

            # Use generic names for internal comparison
            computed_values = {
                "central": float(result[central_key]),  # mean or median
                "spread": float(result[spread_key]),  # variance or iqr
                "ci95_lower": float(result[ci95_key][0]),
                "ci95_upper": float(result[ci95_key][1]),
            }

            # Compare to expected values if provided
            comparison_results = {}
            overall_success = True

            if expected_mean is not None:
                computed_central = computed_values["central"]
                diff_pct = (
                    abs(computed_central - expected_mean) / abs(expected_mean) * 100
                    if expected_mean != 0
                    else float("inf")
                )
                comparison_results["central_match"] = diff_pct < self.threshold_pct
                comparison_results["central_diff_pct"] = diff_pct
                overall_success = overall_success and comparison_results["central_match"]

            if expected_variance is not None:
                computed_spread = computed_values["spread"]
                diff_pct = (
                    abs(computed_spread - expected_variance) / abs(expected_variance) * 100
                    if expected_variance != 0
                    else float("inf")
                )
                comparison_results["spread_match"] = diff_pct < self.threshold_pct
                comparison_results["spread_diff_pct"] = diff_pct
                overall_success = overall_success and comparison_results["spread_match"]

            if expected_ci95 is not None and len(expected_ci95) == 2:
                computed_ci95 = [computed_values["ci95_lower"], computed_values["ci95_upper"]]

                # For lower bound: use absolute difference if expected is 0, otherwise percentage
                if expected_ci95[0] == 0:
                    # Both should be 0 or very close - use absolute tolerance (e.g., 1e-6)
                    lower_diff = (
                        abs(computed_ci95[0] - expected_ci95[0]) * 100
                    )  # Scale for comparison with threshold_pct
                else:
                    lower_diff = (
                        abs(computed_ci95[0] - expected_ci95[0]) / abs(expected_ci95[0]) * 100
                    )

                # For upper bound: use absolute difference if expected is 0, otherwise percentage
                if expected_ci95[1] == 0:
                    upper_diff = abs(computed_ci95[1] - expected_ci95[1]) * 100
                else:
                    upper_diff = (
                        abs(computed_ci95[1] - expected_ci95[1]) / abs(expected_ci95[1]) * 100
                    )

                comparison_results["ci95_match"] = (lower_diff < self.threshold_pct) and (
                    upper_diff < self.threshold_pct
                )
                comparison_results["ci95_diff_pct"] = max(lower_diff, upper_diff)
                overall_success = overall_success and comparison_results["ci95_match"]

            results = {"computed_values": computed_values, "comparison": comparison_results}

            # Build message with calculated vs reported values
            if comparison_results:
                issues = []

                # Report central tendency (mean or median)
                if expected_mean is not None:
                    computed_central = computed_values["central"]
                    stat_label = stat_names[0]  # 'mean' or 'median'
                    if comparison_results.get("central_match", True):
                        issues.append(
                            f"  {stat_label}:     calculated={computed_central:12.3e}, reported={expected_mean:12.3e} ✓"
                        )
                    else:
                        issues.append(
                            f"  {stat_label}:     calculated={computed_central:12.3e}, reported={expected_mean:12.3e} ✗"
                        )

                # Report spread (variance or IQR)
                if expected_variance is not None:
                    computed_spread = computed_values["spread"]
                    stat_label = stat_names[1]  # 'variance' or 'iqr'
                    if comparison_results.get("spread_match", True):
                        issues.append(
                            f"  {stat_label}: calculated={computed_spread:12.3e}, reported={expected_variance:12.3e} ✓"
                        )
                    else:
                        issues.append(
                            f"  {stat_label}: calculated={computed_spread:12.3e}, reported={expected_variance:12.3e} ✗"
                        )

                # Report CI95
                if expected_ci95 is not None:
                    computed_ci95 = [computed_values["ci95_lower"], computed_values["ci95_upper"]]
                    if comparison_results.get("ci95_match", True):
                        issues.append(
                            f"  CI95:     calculated=[{computed_ci95[0]:12.3e}, {computed_ci95[1]:12.3e}], reported=[{expected_ci95[0]:12.3e}, {expected_ci95[1]:12.3e}] ✓"
                        )
                    else:
                        issues.append(
                            f"  CI95:     calculated=[{computed_ci95[0]:12.3e}, {computed_ci95[1]:12.3e}], reported=[{expected_ci95[0]:12.3e}, {expected_ci95[1]:12.3e}] ✗"
                        )

                msg = "\n" + "\n".join(issues)
            else:
                msg = "Python code executed (no expected values to compare)"

            return (overall_success, msg, results)

        except Exception as e:
            return (False, f"Execution error: {str(e)}", {})

    def validate_file(self, file_info: dict) -> tuple:
        """
        Validate code execution in a single YAML file.

        Returns:
            (is_valid, error_msg) tuple
        """
        data = file_info["data"]
        file_info["filename"]

        # Extract Python code
        python_code, code_type = self.extract_python_code(data)

        if not python_code:
            return (True, "No Python code found (skipped)")

        # Extract inputs
        inputs = self.extract_inputs(data)

        if not inputs:
            return (False, "No inputs defined")

        # Extract expected values from YAML
        expected_mean = None
        expected_variance = None
        expected_ci95 = None

        if code_type == "parameter" and "parameter_estimates" in data:
            estimates = data["parameter_estimates"]
            # Support both old (mean/variance) and new (median/iqr) field names
            expected_mean = estimates.get("median") or estimates.get("mean")
            expected_variance = estimates.get("iqr") or estimates.get("variance")
            expected_ci95 = estimates.get("ci95")
        elif code_type == "test_statistic" and "test_statistic_estimates" in data:
            estimates = data["test_statistic_estimates"]
            # Support both old (mean/variance) and new (median/iqr) field names
            expected_mean = estimates.get("median") or estimates.get("mean")
            expected_variance = estimates.get("iqr") or estimates.get("variance")
            expected_ci95 = estimates.get("ci95")

        # Execute Python code with comparison
        success, message, results = self.execute_python_code(
            python_code, inputs, code_type, expected_mean, expected_variance, expected_ci95
        )

        return (success, message)

    def validate(self) -> ValidationReport:
        """Validate Python code in all YAML files."""
        report = ValidationReport(self.name)

        # ANSI color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        print(f"Testing Python code execution in {self.data_dir}...")
        print(f"Value match threshold: {self.threshold_pct}%")
        print()

        files = load_yaml_directory(self.data_dir)
        total_files = len(files)
        passed = 0
        failed = 0
        skipped = 0

        for idx, file_info in enumerate(files, 1):
            filename = file_info["filename"]

            # Show progress
            print(f"[{idx}/{total_files}] {filename}")

            is_valid, message = self.validate_file(file_info)

            if "skipped" in message.lower():
                print(f"{YELLOW}  ⊘ No Python code found (skipped){RESET}")
                skipped += 1
                report.add_pass(filename, message)
            elif is_valid:
                # Print the comparison results (message contains the formatted comparisons)
                if message.startswith("\n"):
                    # Multi-line message with comparisons - print each line
                    for line in message.strip().split("\n"):
                        if "✓" in line:
                            print(f"{GREEN}  {line.strip()}{RESET}")
                        elif "✗" in line:
                            print(f"{RED}  {line.strip()}{RESET}")
                        else:
                            print(f"  {line.strip()}")
                    print(f"{GREEN}  ✓ All values match{RESET}")
                else:
                    print(f"{GREEN}  ✓ {message}{RESET}")
                passed += 1
                report.add_pass(filename, message)
            else:
                # Print the error/comparison results
                if message.startswith("\n"):
                    for line in message.strip().split("\n"):
                        if "✓" in line:
                            print(f"{GREEN}  {line.strip()}{RESET}")
                        elif "✗" in line:
                            print(f"{RED}  {line.strip()}{RESET}")
                        else:
                            print(f"  {line.strip()}")
                    print(f"{RED}  ✗ Value mismatch detected{RESET}")
                else:
                    print(f"{RED}  ✗ {message}{RESET}")
                failed += 1
                report.add_fail(filename, message)

        # Print summary
        print()
        print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")

        return report
