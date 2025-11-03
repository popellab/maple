#!/usr/bin/env python3
"""
Validation runner wrapper for automated workflow integration.

Runs workflow-specific validation suites and provides unified summary format.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ValidationRunner:
    """Runs validation suites and formats results for workflow reporting."""

    def __init__(self, base_dir: Path):
        """
        Initialize validation runner.

        Args:
            base_dir: Base directory of qsp-llm-workflows
        """
        self.base_dir = Path(base_dir)
        self.validate_dir = self.base_dir / "scripts" / "validate"

    def run_parameter_validations(self,
                                  data_dir: Path,
                                  template: Path,
                                  output_dir: Path,
                                  timeout: int = 600) -> Dict[str, Any]:
        """
        Run parameter-specific validation suite.

        Args:
            data_dir: Directory with parameter YAML files (to-review/)
            template: Path to parameter template
            output_dir: Output directory for validation results
            timeout: Timeout in seconds (default: 600)

        Returns:
            Dictionary with validation results
        """
        script_path = self.validate_dir / "run_all_validations.py"

        result = subprocess.run(
            ["python3", str(script_path), str(data_dir), str(template), str(output_dir)],
            cwd=self.base_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return self._parse_validation_results(output_dir, result.returncode == 0)

    def run_test_statistic_validations(self,
                                      data_dir: Path,
                                      template: Path,
                                      output_dir: Path,
                                      timeout: int = 600) -> Dict[str, Any]:
        """
        Run test statistic validation suite (uses same validators as parameters).

        Args:
            data_dir: Directory with test statistic YAML files (to-review/)
            template: Path to test statistic template
            output_dir: Output directory for validation results
            timeout: Timeout in seconds (default: 600)

        Returns:
            Dictionary with validation results
        """
        # Use same validation suite as parameters (validators are now schema-agnostic)
        script_path = self.validate_dir / "run_all_validations.py"

        result = subprocess.run(
            ["python3", str(script_path), str(data_dir), str(template), str(output_dir)],
            cwd=self.base_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return self._parse_validation_results(output_dir, result.returncode == 0)

    def run_validation(self,
                      workflow_type: str,
                      data_dir: Path,
                      template: Path,
                      output_dir: Path,
                      timeout: int = 600) -> Dict[str, Any]:
        """
        Run workflow-appropriate validation suite.

        Args:
            workflow_type: Type of workflow (parameter/test_statistic/quick_estimate)
            data_dir: Directory with YAML files (not used - determined by script)
            template: Path to template file (not used - determined by script)
            output_dir: Output directory for validation results (not used - goes to output/validation_results)
            timeout: Timeout in seconds

        Returns:
            Dictionary with validation results
        """
        # Map workflow type to validation type
        workflow_type_map = {
            "parameter": "parameter_estimates",
            "test_statistic": "test_statistics",
            "quick_estimate": "quick_estimates"
        }
        validation_type = workflow_type_map.get(workflow_type)

        if not validation_type:
            return {
                "workflow_type": workflow_type,
                "status": "error",
                "message": f"Unknown workflow type: {workflow_type}"
            }

        if workflow_type == "quick_estimate":
            # Quick estimates don't have comprehensive validation yet
            return {
                "workflow_type": workflow_type,
                "status": "skipped",
                "message": "Quick estimate validation not implemented"
            }

        # Run validation script (it determines paths internally)
        script_path = self.validate_dir / "run_all_validations.py"

        result = subprocess.run(
            ["python3", str(script_path), validation_type],
            cwd=self.base_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Validation outputs to output/validation_results
        default_output_dir = self.base_dir / "output" / "validation_results"

        return self._parse_validation_results(default_output_dir, result.returncode == 0)

    def _parse_validation_results(self, output_dir: Path, success: bool) -> Dict[str, Any]:
        """
        Parse validation results from output directory.

        Args:
            output_dir: Directory containing validation JSON files
            success: Whether validation script succeeded

        Returns:
            Dictionary with parsed validation summary
        """
        summary_file = output_dir / "master_validation_summary.json"

        if not summary_file.exists():
            return {
                "status": "error" if not success else "completed",
                "message": "Validation summary not found",
                "success": success
            }

        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            # Extract key metrics
            validations = summary.get('validations', [])
            total = len(validations)
            passed = sum(1 for v in validations if v.get('success', False))
            failed = total - passed

            return {
                "status": "completed",
                "success": success,
                "total_validations": total,
                "passed": passed,
                "failed": failed,
                "validations": validations,
                "details_path": str(summary_file)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to parse validation summary: {e}",
                "success": False
            }

    def format_summary_for_commit(self, validation_results: Optional[Dict[str, Any]]) -> str:
        """
        Format validation summary for git commit message.

        Args:
            validation_results: Results from run_validation()

        Returns:
            Formatted string for commit message
        """
        if not validation_results:
            return "Validation: Not run"

        status = validation_results.get("status", "unknown")

        if status == "skipped":
            return f"Validation: Skipped ({validation_results.get('message', 'N/A')})"

        if status == "error":
            return f"Validation: Error - {validation_results.get('message', 'Unknown error')}"

        if status == "completed":
            total = validation_results.get("total_validations", 0)
            passed = validation_results.get("passed", 0)
            failed = validation_results.get("failed", 0)

            lines = [
                f"Validation: {passed}/{total} checks passed"
            ]

            if failed > 0:
                lines.append(f"  ⚠ {failed} validation(s) failed - review recommended")

                # List failed validations
                validations = validation_results.get("validations", [])
                for v in validations:
                    if not v.get('success', False):
                        lines.append(f"    - {v.get('name', 'Unknown')}")

            return "\n".join(lines)

        return "Validation: Unknown status"
