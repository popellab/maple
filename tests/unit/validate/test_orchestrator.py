"""
Tests for ValidationOrchestrator.
"""

import json
import tempfile
from pathlib import Path

import yaml

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.validate.orchestrator import ValidationOrchestrator, ValidationResult
from qsp_llm_workflows.core.validation_utils import ValidationReport
from qsp_llm_workflows.validate.tag_validation_results import tag_files_individually


class PassingValidator(Validator):
    """Mock validator that always passes."""

    @property
    def name(self) -> str:
        return "Passing Validation"

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name)
        report.add_pass("file1.yaml", "All good")
        report.add_pass("file2.yaml", "All good")
        return report


class FailingValidator(Validator):
    """Mock validator that always fails."""

    @property
    def name(self) -> str:
        return "Failing Validation"

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name)
        report.add_pass("file1.yaml", "This one passed")
        report.add_fail("file2.yaml", "This one failed")
        return report


class ErrorValidator(Validator):
    """Mock validator that raises an error."""

    @property
    def name(self) -> str:
        return "Error Validation"

    def validate(self) -> ValidationReport:
        raise ValueError("Validator encountered an error")


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult from reports."""
        report1 = ValidationReport("Test1")
        report1.add_pass("file1.yaml", "OK")

        report2 = ValidationReport("Test2")
        report2.add_fail("file2.yaml", "Failed")

        result = ValidationResult([report1, report2], "/path/to/data", "TestModel")

        assert result.data_dir == "/path/to/data"
        assert result.model_name == "TestModel"
        assert len(result.reports) == 2
        assert result.timestamp is not None

    def test_has_failures_true(self):
        """Test has_failures returns True when there are failures."""
        report1 = ValidationReport("Test1")
        report1.add_pass("file1.yaml", "OK")

        report2 = ValidationReport("Test2")
        report2.add_fail("file2.yaml", "Failed")

        result = ValidationResult([report1, report2], "/data", "Model")

        assert result.has_failures is True
        assert result.all_passed is False

    def test_has_failures_false(self):
        """Test has_failures returns False when all pass."""
        report1 = ValidationReport("Test1")
        report1.add_pass("file1.yaml", "OK")

        report2 = ValidationReport("Test2")
        report2.add_pass("file2.yaml", "OK")

        result = ValidationResult([report1, report2], "/data", "Model")

        assert result.has_failures is False
        assert result.all_passed is True

    def test_get_passed_validation_names(self):
        """Test getting names of passed validations."""
        report1 = ValidationReport("Schema Validation")
        report1.add_pass("file1.yaml", "OK")

        report2 = ValidationReport("DOI Validation")
        report2.add_fail("file2.yaml", "Failed")

        report3 = ValidationReport("Code Execution")
        report3.add_pass("file3.yaml", "OK")

        result = ValidationResult([report1, report2, report3], "/data", "Model")

        passed_names = result.get_passed_validation_names()
        assert len(passed_names) == 2
        assert "Schema Validation" in passed_names
        assert "Code Execution" in passed_names
        assert "DOI Validation" not in passed_names

    def test_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        report1 = ValidationReport("Test1")
        report1.add_pass("file1.yaml", "OK")

        report2 = ValidationReport("Test2")
        report2.add_fail("file2.yaml", "Failed")

        result = ValidationResult([report1, report2], "/data", "TestModel")

        result_dict = result.to_dict()

        assert result_dict["data_dir"] == "/data"
        assert result_dict["model"] == "TestModel"
        assert "timestamp" in result_dict
        assert len(result_dict["validations"]) == 2

        # Check first validation (passed)
        val1 = result_dict["validations"][0]
        assert val1["name"] == "Test1"
        assert val1["success"] is True
        assert val1["summary"]["passed"] == 1
        assert val1["summary"]["failed"] == 0

        # Check second validation (failed)
        val2 = result_dict["validations"][1]
        assert val2["name"] == "Test2"
        assert val2["success"] is False
        assert val2["summary"]["passed"] == 0
        assert val2["summary"]["failed"] == 1


class TestValidationOrchestrator:
    """Test ValidationOrchestrator class."""

    def test_orchestrator_creation(self):
        """Test creating ValidationOrchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            assert orchestrator.data_dir == "/data"
            assert orchestrator.output_dir == output_dir
            assert orchestrator.model_name == "TestModel"
            assert output_dir.exists()  # Should create output directory

    def test_run_single_passing_validator(self):
        """Test running a single validator that passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validator = PassingValidator("/data")
            report = orchestrator.run_validator(validator)

            assert report.name == "Passing Validation"
            assert len(report.passed) == 2
            assert len(report.failed) == 0

    def test_run_single_failing_validator(self):
        """Test running a validator that fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validator = FailingValidator("/data")
            report = orchestrator.run_validator(validator)

            assert report.name == "Failing Validation"
            assert len(report.passed) == 1
            assert len(report.failed) == 1

    def test_run_validator_with_error(self):
        """Test running a validator that raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validator = ErrorValidator("/data")
            report = orchestrator.run_validator(validator)

            # Should return error report instead of crashing
            assert report.name == "Error Validation"
            assert len(report.failed) == 1
            assert "Validator encountered an error" in report.failed[0]["reason"]

    def test_save_report(self):
        """Test saving validation report to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            report = ValidationReport("Test Validation")
            report.add_pass("file1.yaml", "OK")
            report.add_fail("file2.yaml", "Failed")

            orchestrator.save_report(report, "test_validation")

            # Check file exists
            output_file = output_dir / "test_validation.json"
            assert output_file.exists()

            # Check content
            with open(output_file) as f:
                data = json.load(f)

            assert data["summary"]["name"] == "Test Validation"
            assert data["summary"]["passed"] == 1
            assert data["summary"]["failed"] == 1
            assert len(data["passed"]) == 1
            assert len(data["failed"]) == 1

    def test_run_all_validations(self):
        """Test running multiple validators."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validators = [
                PassingValidator("/data"),
                FailingValidator("/data"),
                PassingValidator("/data"),
            ]

            result = orchestrator.run_all_validations(validators)

            # Check result
            assert isinstance(result, ValidationResult)
            assert len(result.reports) == 3
            assert result.has_failures is True  # One validator failed
            assert result.all_passed is False

            # Check that reports were saved
            assert (output_dir / "passing_validation.json").exists()
            assert (output_dir / "failing_validation.json").exists()
            assert (output_dir / "master_validation_summary.json").exists()

    def test_run_all_validations_all_pass(self):
        """Test running multiple validators that all pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validators = [
                PassingValidator("/data"),
                PassingValidator("/data"),
            ]

            result = orchestrator.run_all_validations(validators)

            assert result.has_failures is False
            assert result.all_passed is True

    def test_get_passed_validation_names_for_tagging(self):
        """Test getting passed validation names for file tagging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validators = [
                PassingValidator("/data"),
                FailingValidator("/data"),
            ]

            result = orchestrator.run_all_validations(validators)

            passed_names = result.get_passed_validation_names()
            assert len(passed_names) == 1
            assert "Passing Validation" in passed_names
            assert "Failing Validation" not in passed_names

    def test_master_summary_structure(self):
        """Test structure of master validation summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            orchestrator = ValidationOrchestrator("/data", output_dir, "TestModel")

            validators = [PassingValidator("/data"), FailingValidator("/data")]

            orchestrator.run_all_validations(validators)

            # Check master summary file
            summary_file = output_dir / "master_validation_summary.json"
            assert summary_file.exists()

            with open(summary_file) as f:
                summary = json.load(f)

            assert summary["data_dir"] == "/data"
            assert summary["model"] == "TestModel"
            assert "timestamp" in summary
            assert len(summary["validations"]) == 2

            # Check validation entries
            assert summary["validations"][0]["name"] == "Passing Validation"
            assert summary["validations"][0]["success"] is True
            assert summary["validations"][1]["name"] == "Failing Validation"
            assert summary["validations"][1]["success"] is False

    def test_get_per_file_tags(self):
        """Test getting per-file validation tags based on which validations each file passed."""
        # Create reports where different files pass different validations
        report1 = ValidationReport("Schema Validation")
        report1.add_pass("file1.yaml", "OK")
        report1.add_pass("file2.yaml", "OK")
        report1.add_fail("file3.yaml", "Failed schema")

        report2 = ValidationReport("Code Execution")
        report2.add_pass("file1.yaml", "OK")
        report2.add_fail("file2.yaml", "Code error")
        report2.add_pass("file3.yaml", "OK")

        result = ValidationResult([report1, report2], "/data", "Model")
        file_tags = result.get_per_file_tags()

        # file1 passed both validations
        assert len(file_tags["file1.yaml"]) == 2

        # file2 passed schema only
        assert "schema_validation" in file_tags["file2.yaml"]
        assert "code_execution" not in file_tags["file2.yaml"]

        # file3 passed code execution only
        assert "code_execution" in file_tags["file3.yaml"]
        assert "schema_validation" not in file_tags["file3.yaml"]


class TestTagFilesIndividually:
    """Test tag_files_individually function."""

    def test_tags_files_with_different_tags(self):
        """Test that files are tagged with their individual validation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.yaml"
            file2 = Path(tmpdir) / "file2.yaml"
            file1.write_text("parameter_name: test1\n")
            file2.write_text("parameter_name: test2\n")

            file_tags = {
                "file1.yaml": ["schema_validation", "code_execution"],
                "file2.yaml": ["schema_validation"],
            }

            count = tag_files_individually(tmpdir, file_tags)
            assert count == 2

            # Check file1 has 2 tags, file2 has 1 tag
            with open(file1) as f:
                content1 = yaml.safe_load(f)
            assert len(content1["validation"]["tags"]) == 2

            with open(file2) as f:
                content2 = yaml.safe_load(f)
            assert len(content2["validation"]["tags"]) == 1
