"""
Tests for base Validator class.
"""

import pytest
from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.validate.validation_utils import ValidationReport


class MockValidator(Validator):
    """Mock validator for testing base class interface."""

    @property
    def name(self) -> str:
        return "Mock Validation"

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name)
        report.add_pass("test_file.yaml", "Validation passed")
        return report


class TestValidatorInterface:
    """Test base Validator class interface."""

    def test_validator_requires_data_dir(self):
        """Test that validator requires data_dir parameter."""
        validator = MockValidator("/path/to/data")
        assert validator.data_dir == "/path/to/data"

    def test_validator_accepts_kwargs(self):
        """Test that validator accepts additional configuration."""
        validator = MockValidator(
            "/path/to/data", rate_limit=2.0, timeout=30, custom_option="value"
        )
        assert validator.config["rate_limit"] == 2.0
        assert validator.config["timeout"] == 30
        assert validator.config["custom_option"] == "value"

    def test_validator_has_name_property(self):
        """Test that validator has name property."""
        validator = MockValidator("/path/to/data")
        assert validator.name == "Mock Validation"

    def test_validator_has_validate_method(self):
        """Test that validator has validate method returning ValidationReport."""
        validator = MockValidator("/path/to/data")
        report = validator.validate()

        assert isinstance(report, ValidationReport)
        assert report.name == "Mock Validation"
        assert len(report.passed) == 1
        assert len(report.failed) == 0

    def test_validator_repr(self):
        """Test validator string representation."""
        validator = MockValidator("/path/to/data")
        repr_str = repr(validator)

        assert "MockValidator" in repr_str
        assert "/path/to/data" in repr_str

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented by subclasses."""

        class IncompleteValidator(Validator):
            """Validator missing required abstract methods."""

            pass

        with pytest.raises(TypeError):
            IncompleteValidator("/path/to/data")

    def test_name_must_be_implemented(self):
        """Test that name property must be implemented."""

        class ValidatorWithoutName(Validator):
            """Validator missing name property."""

            def validate(self) -> ValidationReport:
                return ValidationReport("Test")

        with pytest.raises(TypeError):
            ValidatorWithoutName("/path/to/data")

    def test_validate_must_be_implemented(self):
        """Test that validate method must be implemented."""

        class ValidatorWithoutValidate(Validator):
            """Validator missing validate method."""

            @property
            def name(self) -> str:
                return "Test"

        with pytest.raises(TypeError):
            ValidatorWithoutValidate("/path/to/data")
