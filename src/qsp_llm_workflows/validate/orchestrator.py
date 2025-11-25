"""
Validation orchestrator using Chain of Responsibility pattern.

Executes validators in sequence, collects results, and generates reports.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.validate.validation_utils import ValidationReport


class ValidationResult:
    """Result of validation orchestration with metadata."""

    def __init__(self, reports: List[ValidationReport], data_dir: str, model_name: str):
        """
        Create validation result from reports.

        Args:
            reports: List of ValidationReport objects from validators
            data_dir: Directory that was validated
            model_name: Name of Pydantic model used for validation
        """
        self.reports = reports
        self.data_dir = data_dir
        self.model_name = model_name
        self.timestamp = datetime.now().isoformat()

    @property
    def has_failures(self) -> bool:
        """Check if any validations failed."""
        return any(len(report.failed) > 0 for report in self.reports)

    @property
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return not self.has_failures

    def get_passed_validation_names(self) -> List[str]:
        """
        Get list of validation names that passed.

        Returns:
            List of validation names (for tagging)
        """
        return [report.name for report in self.reports if len(report.failed) == 0]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with validation results
        """
        return {
            "timestamp": self.timestamp,
            "data_dir": self.data_dir,
            "model": self.model_name,
            "validations": [
                {
                    "name": report.name,
                    "success": len(report.failed) == 0,
                    "summary": report.get_summary(),
                }
                for report in self.reports
            ],
        }


class ValidationOrchestrator:
    """
    Orchestrates validation workflow using Chain of Responsibility pattern.

    Executes validators in sequence, handles result collection, and generates reports.
    """

    def __init__(self, data_dir: str, output_dir: Path, model_name: str):
        """
        Initialize validation orchestrator.

        Args:
            data_dir: Directory containing YAML files to validate
            output_dir: Directory for validation reports
            model_name: Name of Pydantic model (for reporting)
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_validator(self, validator: Validator) -> ValidationReport:
        """
        Run a single validator and handle errors.

        Args:
            validator: Validator instance

        Returns:
            ValidationReport (may contain error information)
        """
        print(f"\n{'='*60}")
        print(f"Running: {validator.name}")
        print(f"{'='*60}")

        try:
            report = validator.validate()
            return report
        except Exception as e:
            # Create error report
            report = ValidationReport(validator.name)
            report.add_fail("VALIDATION_ERROR", f"Validator failed: {str(e)}")
            return report

    def save_report(self, report: ValidationReport, filename: str):
        """
        Save validation report to JSON file.

        Args:
            report: ValidationReport to save
            filename: Output filename (without extension)
        """
        output_file = self.output_dir / f"{filename}.json"

        report_data = {
            "summary": report.get_summary(),
            "passed": report.passed,
            "failed": report.failed,
            "warnings": report.warnings,
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

    def print_report_summary(self, report: ValidationReport):
        """
        Print summary of validation report with detailed failure information.

        Args:
            report: ValidationReport to summarize
        """
        summary = report.get_summary()

        if "total" in summary:
            print(f"  Total: {summary['total']}")
            print(f"  Passed: {summary['passed']}")
            print(f"  Failed: {summary['failed']}")

            # Print detailed failure information
            if report.failed:
                print("\n  Failures:")
                for failure in report.failed:
                    item = failure.get("item", "Unknown")
                    reason = failure.get("reason", "No reason provided")
                    print(f"    ✗ {item}")
                    print(f"      {reason}")
                print()  # Blank line after failures

        elif "n_comparisons" in summary:
            print(f"  Comparisons: {summary['n_comparisons']}")
            if "pearson_r" in summary:
                print(f"  Correlation: {summary['pearson_r']:.3f}")

    def run_all_validations(self, validators: List[Validator]) -> ValidationResult:
        """
        Run all validators and return aggregated results.

        Args:
            validators: List of Validator instances to execute

        Returns:
            ValidationResult with aggregated results
        """
        print(f"\n{'#'*60}")
        print("# VALIDATION ORCHESTRATOR")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Data directory: {self.data_dir}")
        print(f"# Model: {self.model_name}")
        print(f"# Output directory: {self.output_dir}")
        print(f"# Validators: {len(validators)}")
        print(f"{'#'*60}")

        reports = []

        # Execute validators in sequence
        for validator in validators:
            report = self.run_validator(validator)
            reports.append(report)

            # Save report to JSON
            filename = validator.name.lower().replace(" ", "_").replace("-", "_")
            self.save_report(report, filename)

            # Print summary
            self.print_report_summary(report)

        # Create aggregated result
        result = ValidationResult(reports, self.data_dir, self.model_name)

        # Save master summary
        summary_path = self.output_dir / "master_validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def print_master_summary(self, result: ValidationResult):
        """
        Print master validation summary.

        Args:
            result: ValidationResult to summarize
        """
        print(f"\n{'='*60}")
        print("MASTER VALIDATION SUMMARY")
        print(f"{'='*60}")

        for report in result.reports:
            success = len(report.failed) == 0
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\n{report.name}: {status}")

            summary = report.get_summary()
            if "total" in summary:
                print(f"  Total: {summary['total']}")
                print(f"  Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
                print(f"  Failed: {summary['failed']}")
            elif "n_comparisons" in summary:
                print(f"  Comparisons: {summary['n_comparisons']}")
                if "pearson_r" in summary:
                    print(f"  Correlation: {summary['pearson_r']:.3f}")

        print(f"\n{'='*60}")
        print(f"Master summary saved to: {self.output_dir}/master_validation_summary.json")
        print(f"Individual reports in: {self.output_dir}")
        print(f"{'='*60}\n")
