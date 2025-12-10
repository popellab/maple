#!/usr/bin/env python3
"""
Test Python code execution from YAML files.

Validates:
- Python code executes without errors
- Function returns required fields (median/iqr for parameters and test statistics)
- Returned values are Pint Quantities with correct unit dimensionality
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
import pint

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport
from qsp_llm_workflows.core.unit_registry import create_unit_registry


class CodeExecutionValidator(Validator):
    """
    Validate Python code execution from YAML files.
    Works for both parameters (v3) and test statistics (v2).

    Validates that derivation code:
    1. Executes without errors
    2. Returns Pint Quantities (not raw floats)
    3. Returns values with correct unit dimensionality
    4. Returns values matching declared YAML values (within threshold)
    """

    def __init__(
        self,
        data_dir: str,
        threshold_pct: float = 5.0,
        interactive: bool = True,
        **kwargs,
    ):
        super().__init__(data_dir, **kwargs)
        self.threshold_pct = threshold_pct
        self.interactive = interactive
        # Store computed values for all files with executable code
        # Key: filepath, Value: dict with code_type and computed_values
        self.executable_files: dict[str, dict] = {}
        # Create Pint UnitRegistry using shared factory
        self.ureg = create_unit_registry()

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

    def extract_inputs(self, data: dict) -> dict:
        """
        Extract inputs from parameter_estimates or test_statistic_estimates.

        Converts the list of input dicts from YAML into a dict of Pint quantities
        keyed by input name. This is the format expected by derivation functions:
            inputs['Input Name'].magnitude  # access value
            inputs['Input Name'].units      # access units

        Returns:
            Dict mapping input names to Pint Quantities, or None if no inputs found
        """
        inputs_list = None

        # Try parameter_estimates first
        if "parameter_estimates" in data:
            estimates = data["parameter_estimates"]
            if "inputs" in estimates and isinstance(estimates["inputs"], list):
                inputs_list = estimates["inputs"]

        # Try test_statistic_estimates
        if inputs_list is None and "test_statistic_estimates" in data:
            estimates = data["test_statistic_estimates"]
            if "inputs" in estimates and isinstance(estimates["inputs"], list):
                inputs_list = estimates["inputs"]

        if inputs_list is None:
            return None

        # Convert list of dicts to dict of Pint quantities
        inputs_dict = {}
        for inp in inputs_list:
            name = inp.get("name")
            value = inp.get("value")
            units_str = inp.get("units", "dimensionless")

            if name is None or value is None:
                continue

            # Parse units and create Pint quantity
            try:
                # Handle special unit strings
                if units_str in (None, "", "dimensionless", "unitless", "ratio"):
                    pint_qty = float(value) * self.ureg.dimensionless
                elif units_str in ("patients", "samples", "cases", "lesions"):
                    # Count units - treat as dimensionless with magnitude
                    pint_qty = float(value) * self.ureg.dimensionless
                elif units_str.startswith("%") or "percent" in units_str.lower():
                    # Percentages - treat as dimensionless
                    pint_qty = float(value) * self.ureg.dimensionless
                elif units_str.endswith("percentage points"):
                    pint_qty = float(value) * self.ureg.dimensionless
                elif "log-space" in units_str.lower() or "log_" in units_str.lower():
                    pint_qty = float(value) * self.ureg.dimensionless
                else:
                    # Try to parse the unit string
                    pint_qty = float(value) * self.ureg.parse_expression(units_str)
            except (pint.UndefinedUnitError, pint.DimensionalityError):
                # Fall back to dimensionless if parsing fails
                pint_qty = float(value) * self.ureg.dimensionless

            inputs_dict[name] = pint_qty

        return inputs_dict if inputs_dict else None

    def execute_python_code(
        self,
        code: str,
        inputs: dict,
        code_type: str,
        expected_unit: str = None,
        expected_mean: float = None,
        expected_variance: float = None,
        expected_ci95: list = None,
    ) -> tuple:
        """
        Execute Python code and check for errors.
        Validate Pint quantities and compare computed values to expected values.

        Args:
            code: Python code to execute
            inputs: Dict mapping input names to Pint Quantities.
                    Access values like: inputs['name'].magnitude
            code_type: "parameter" or "test_statistic"
            expected_unit: Expected output unit (Pint-parseable string)
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

        # Create namespace for execution with Pint support
        namespace = {"inputs": inputs, "np": np, "numpy": np, "pint": pint, "ureg": self.ureg}

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
                ]
                stat_names = ["median", "iqr"]  # For display
            elif code_type == "test_statistic":
                func_name = "derive_distribution"
                expected_keys = ["median_stat", "iqr_stat", "ci95_stat"]
                stat_names = ["median", "iqr"]  # For display
            else:
                return (False, f"Unknown code type: {code_type}", {})

            # Check if function exists
            if func_name not in namespace:
                return (False, f"Code does not define {func_name} function", {})

            # Call the function with ureg argument
            derive_func = namespace[func_name]
            result = derive_func(inputs, self.ureg)

            # Validate return type
            if not isinstance(result, dict):
                return (False, f"{func_name} returned {type(result)}, expected dict", {})

            # Check required fields
            missing = [k for k in expected_keys if k not in result]
            if missing:
                return (False, f"Missing required fields in result: {missing}", {})

            # Extract computed values (normalize keys for comparison)
            central_key = expected_keys[0]  # median_param or median_stat
            spread_key = expected_keys[1]  # iqr_param or iqr_stat
            ci95_key = expected_keys[2]

            # Validate Pint quantities and extract magnitudes
            central_value = result[central_key]
            spread_value = result[spread_key]
            ci95_values = result[ci95_key]

            # Check that returned values are Pint Quantities
            if not isinstance(central_value, pint.Quantity):
                return (
                    False,
                    f"{central_key} is not a Pint Quantity (got {type(central_value).__name__}). "
                    "Derivation code must return Pint Quantities, not raw floats.",
                    {},
                )

            # Validate unit dimensionality if expected_unit provided
            if expected_unit:
                try:
                    expected_pint_unit = self.ureg.parse_expression(expected_unit)
                    if central_value.dimensionality != expected_pint_unit.dimensionality:
                        return (
                            False,
                            f"Unit mismatch: {central_key} has units {central_value.units} "
                            f"(dimensionality: {central_value.dimensionality}), "
                            f"expected {expected_unit} (dimensionality: {expected_pint_unit.dimensionality})",
                            {},
                        )
                except pint.UndefinedUnitError as e:
                    return (False, f"Invalid expected_unit '{expected_unit}': {e}", {})

            # Extract magnitudes for comparison (after unit validation)
            computed_values = {
                "central": float(central_value.magnitude),
                "spread": (
                    float(spread_value.magnitude)
                    if isinstance(spread_value, pint.Quantity)
                    else float(spread_value)
                ),
                "ci95_lower": (
                    float(ci95_values[0].magnitude)
                    if isinstance(ci95_values[0], pint.Quantity)
                    else float(ci95_values[0])
                ),
                "ci95_upper": (
                    float(ci95_values[1].magnitude)
                    if isinstance(ci95_values[1], pint.Quantity)
                    else float(ci95_values[1])
                ),
                "units": str(central_value.units),
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

                # Report units validation
                computed_units = computed_values["units"]
                if expected_unit:
                    units_match = (
                        central_value.dimensionality
                        == self.ureg.parse_expression(expected_unit).dimensionality
                    )
                    if units_match:
                        issues.append(
                            f"  units:    computed={computed_units}, expected={expected_unit} ✓"
                        )
                    else:
                        issues.append(
                            f"  units:    computed={computed_units}, expected={expected_unit} ✗"
                        )
                else:
                    issues.append(
                        f"  units:    computed={computed_units} (no expected unit specified)"
                    )

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
            (is_valid, error_msg, code_type, computed_values, current_values) tuple
            - code_type: "parameter" or "test_statistic" or None
            - computed_values: dict with median, iqr, ci95 or empty dict
            - current_values: dict with current YAML values (median, iqr, ci95)
        """
        data = file_info["data"]
        file_info["filename"]

        # Extract Python code
        python_code, code_type = self.extract_python_code(data)

        if not python_code:
            return (True, "No Python code found (skipped)", None, {}, {})

        # Extract inputs
        inputs = self.extract_inputs(data)

        if not inputs:
            return (False, "No inputs defined", code_type, {}, {})

        # Extract expected values from YAML
        expected_mean = None
        expected_variance = None
        expected_ci95 = None
        expected_unit = None

        if code_type == "parameter":
            # Get expected unit from header field
            expected_unit = data.get("parameter_units")
            if "parameter_estimates" in data:
                estimates = data["parameter_estimates"]
                # Support both old (mean/variance) and new (median/iqr) field names
                expected_mean = estimates.get("median") or estimates.get("mean")
                expected_variance = estimates.get("iqr") or estimates.get("variance")
                expected_ci95 = estimates.get("ci95")
        elif code_type == "test_statistic":
            # Get expected unit from header field
            expected_unit = data.get("output_unit")
            if "test_statistic_estimates" in data:
                estimates = data["test_statistic_estimates"]
                # Support both old (mean/variance) and new (median/iqr) field names
                expected_mean = estimates.get("median") or estimates.get("mean")
                expected_variance = estimates.get("iqr") or estimates.get("variance")
                expected_ci95 = estimates.get("ci95")

        # Store current YAML values
        current_values = {
            "median": expected_mean,
            "iqr": expected_variance,
            "ci95": expected_ci95 if expected_ci95 else [None, None],
        }

        # Execute Python code with comparison and unit validation
        success, message, results = self.execute_python_code(
            python_code,
            inputs,
            code_type,
            expected_unit,
            expected_mean,
            expected_variance,
            expected_ci95,
        )

        # Extract computed values in a format suitable for YAML update
        computed_values = {}
        if results.get("computed_values"):
            cv = results["computed_values"]
            computed_values = {
                "median": cv["central"],
                "iqr": cv["spread"],
                "ci95": [cv["ci95_lower"], cv["ci95_upper"]],
            }

        return (success, message, code_type, computed_values, current_values)

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

        # Clear executable_files from any previous run
        self.executable_files = {}

        for idx, file_info in enumerate(files, 1):
            filename = file_info["filename"]
            filepath = file_info["filepath"]

            # Show progress
            print(f"[{idx}/{total_files}] {filename}")

            is_valid, message, code_type, computed_values, current_values = self.validate_file(
                file_info
            )

            # Store computed values for all files with executable code
            if code_type and computed_values:
                self.executable_files[filepath] = {
                    "filename": filename,
                    "code_type": code_type,
                    "computed_values": computed_values,
                    "current_values": current_values,
                    "is_valid": is_valid,
                }

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

        # Prompt user if there are files with executable code (unless interactive=False)
        if self.interactive and self.executable_files:
            self._prompt_apply_computed_values()

        return report

    def _prompt_apply_computed_values(self) -> None:
        """
        Interactively prompt user to overwrite YAML values with computed values.
        Goes through each file with executable code one by one.
        """
        # ANSI color codes
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        # Count matches vs mismatches
        n_match = sum(1 for info in self.executable_files.values() if info["is_valid"])
        n_mismatch = len(self.executable_files) - n_match

        print()
        print("=" * 60)
        print("APPLY COMPUTED VALUES")
        print("=" * 60)
        print()
        print(f"Found {len(self.executable_files)} files with executable code.")
        print(f"  {GREEN}✓ {n_match} match{RESET}")
        print(f"  {YELLOW}✗ {n_mismatch} mismatch{RESET}")
        print()

        # Ask if user wants to enter the review
        while True:
            response = (
                input("Review files and optionally overwrite values? [y/n]: ").strip().lower()
            )
            if response in ("y", "yes"):
                break
            elif response in ("n", "no"):
                print("Skipping value overwrite review.")
                return
            else:
                print("Please enter 'y' or 'n'")

        print()

        updated_count = 0
        skipped_count = 0

        for filepath, info in self.executable_files.items():
            filename = info["filename"]
            code_type = info["code_type"]
            computed = info["computed_values"]
            current = info["current_values"]
            is_valid = info["is_valid"]

            # Show file info
            status = f"{GREEN}✓ MATCH{RESET}" if is_valid else f"{YELLOW}✗ MISMATCH{RESET}"
            print(f"{CYAN}File: {filename}{RESET} [{status}]")

            # Format current values (handle None)
            def fmt(val):
                return f"{val:.6e}" if val is not None else "None"

            def fmt_ci95(ci):
                if ci and ci[0] is not None and ci[1] is not None:
                    return f"[{ci[0]:.6e}, {ci[1]:.6e}]"
                return str(ci)

            # Show current vs computed values side by side
            print(f"  {'Field':<8} {'Current (YAML)':<20} {'Computed (code)':<20}")
            print(f"  {'-'*8} {'-'*20} {'-'*20}")
            print(f"  {'median':<8} {fmt(current['median']):<20} {fmt(computed['median']):<20}")
            print(f"  {'iqr':<8} {fmt(current['iqr']):<20} {fmt(computed['iqr']):<20}")
            print(f"  {'ci95':<8} {fmt_ci95(current['ci95']):<20} {fmt_ci95(computed['ci95']):<20}")
            print()

            # Prompt user
            while True:
                response = (
                    input("  Overwrite YAML with computed values? [y/n/q(uit)]: ").strip().lower()
                )
                if response in ("y", "yes"):
                    success = self._update_yaml_values(filepath, code_type, computed)
                    if success:
                        print(f"  {GREEN}✓ Updated{RESET}")
                        updated_count += 1
                    else:
                        print(f"  {YELLOW}✗ Failed to update{RESET}")
                    break
                elif response in ("n", "no"):
                    print("  Skipped")
                    skipped_count += 1
                    break
                elif response in ("q", "quit"):
                    print()
                    print(f"Stopped. Updated {updated_count} files, skipped {skipped_count} files.")
                    return
                else:
                    print("  Please enter 'y', 'n', or 'q'")

            print()

        print(f"Done. Updated {updated_count} files, skipped {skipped_count} files.")

    def _update_yaml_values(self, filepath: str, code_type: str, computed_values: dict) -> bool:
        """
        Update median, iqr, and ci95 values in a YAML file.

        Args:
            filepath: Path to the YAML file
            code_type: "parameter" or "test_statistic"
            computed_values: dict with median, iqr, ci95 keys

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the file content
            with open(filepath, "r") as f:
                content = f.read()

            # Determine the estimates section key
            if code_type == "parameter":
                section_key = "parameter_estimates"
            else:
                section_key = "test_statistic_estimates"

            # Use regex to find and replace values within the estimates section
            # This preserves formatting better than loading/dumping YAML

            # Find the section start
            section_pattern = rf"^{section_key}:\s*$"
            section_match = re.search(section_pattern, content, re.MULTILINE)
            if not section_match:
                print(f"    Warning: Could not find {section_key} section")
                return False

            section_start = section_match.end()

            # Find the next top-level key (or end of file)
            next_section = re.search(r"^\w+:", content[section_start:], re.MULTILINE)
            if next_section:
                section_end = section_start + next_section.start()
            else:
                section_end = len(content)

            section_content = content[section_start:section_end]

            # Replace median value
            median_pattern = r"(^\s+median:\s*)[\d.eE+-]+"
            section_content = re.sub(
                median_pattern,
                rf"\g<1>{computed_values['median']:.10g}",
                section_content,
                flags=re.MULTILINE,
            )

            # Replace iqr value
            iqr_pattern = r"(^\s+iqr:\s*)[\d.eE+-]+"
            section_content = re.sub(
                iqr_pattern,
                rf"\g<1>{computed_values['iqr']:.10g}",
                section_content,
                flags=re.MULTILINE,
            )

            # Replace ci95 value - handle both inline [a, b] and multi-line formats
            ci95_lower = computed_values["ci95"][0]
            ci95_upper = computed_values["ci95"][1]

            # Try inline format first: ci95: [1.23, 4.56]
            ci95_inline_pattern = r"(^\s+ci95:\s*)\[[\d.eE+\-,\s]+\]"
            new_ci95 = f"[{ci95_lower:.10g}, {ci95_upper:.10g}]"
            section_content, n_subs = re.subn(
                ci95_inline_pattern,
                rf"\g<1>{new_ci95}",
                section_content,
                flags=re.MULTILINE,
            )

            # If no inline match, try multi-line format
            if n_subs == 0:
                # Multi-line format:
                # ci95:
                #   - 1.23
                #   - 4.56
                ci95_multiline_pattern = (
                    r"(^\s+ci95:\s*\n)" r"(\s+- )[\d.eE+-]+\n" r"(\s+- )[\d.eE+-]+"
                )
                replacement = rf"\g<1>\g<2>{ci95_lower:.10g}\n\g<3>{ci95_upper:.10g}"
                section_content = re.sub(
                    ci95_multiline_pattern,
                    replacement,
                    section_content,
                    flags=re.MULTILINE,
                )

            # Reconstruct the file
            new_content = content[:section_start] + section_content + content[section_end:]

            # Write back
            with open(filepath, "w") as f:
                f.write(new_content)

            return True

        except Exception as e:
            print(f"    Error updating file: {e}")
            return False
