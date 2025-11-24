"""
Tests for individual validators.

Tests core functionality of each validator to ensure they work with base Validator class.
"""

import tempfile


from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.validate.check_schema_compliance import SchemaValidator
from qsp_llm_workflows.validate.test_code_execution import CodeExecutionValidator
from qsp_llm_workflows.validate.check_text_snippets import TextSnippetValidator
from qsp_llm_workflows.validate.check_source_references import SourceReferenceValidator
from qsp_llm_workflows.validate.check_doi_validity import DOIValidator
from qsp_llm_workflows.validate.check_value_consistency import ValueConsistencyChecker
from qsp_llm_workflows.validate.check_duplicate_primary_sources import (
    DuplicatePrimarySourceChecker,
)
from qsp_llm_workflows.validate.check_snippet_sources_manual_verify import (
    SnippetSourceManualVerifier,
)
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata
from qsp_llm_workflows.core.validation_utils import ValidationReport


class TestSchemaValidator:
    """Test SchemaValidator inherits from Validator and works correctly."""

    def test_inherits_from_validator(self):
        """Test SchemaValidator inherits from Validator base class."""
        validator = SchemaValidator("/data", model_class=ParameterMetadata)
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test validator has correct name property."""
        validator = SchemaValidator("/data", model_class=ParameterMetadata)
        assert validator.name == "Template Compliance Validation"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SchemaValidator(tmpdir, model_class=ParameterMetadata)
            report = validator.validate()
            assert isinstance(report, ValidationReport)
            assert report.name == "Template Compliance Validation"

    def test_accepts_model_class_parameter(self):
        """Test validator accepts model_class parameter."""
        validator = SchemaValidator("/data", model_class=ParameterMetadata)
        assert validator.model_class == ParameterMetadata


class TestCodeExecutionValidator:
    """Test CodeExecutionValidator."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = CodeExecutionValidator("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = CodeExecutionValidator("/data")
        assert validator.name == "Code Execution Testing"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = CodeExecutionValidator(tmpdir, threshold_pct=5.0)
            report = validator.validate()
            assert isinstance(report, ValidationReport)


class TestTextSnippetValidator:
    """Test TextSnippetValidator."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = TextSnippetValidator("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = TextSnippetValidator("/data")
        assert validator.name == "Text Snippet Validation"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()
            assert isinstance(report, ValidationReport)


class TestSourceReferenceValidator:
    """Test SourceReferenceValidator."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = SourceReferenceValidator("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = SourceReferenceValidator("/data")
        assert validator.name == "Source Reference Validation"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SourceReferenceValidator(tmpdir)
            report = validator.validate()
            assert isinstance(report, ValidationReport)


class TestDOIValidator:
    """Test DOIValidator."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = DOIValidator("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = DOIValidator("/data")
        assert validator.name == "DOI Resolution Validation"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DOIValidator(tmpdir, rate_limit=1.0)
            report = validator.validate()
            assert isinstance(report, ValidationReport)

    def test_accepts_rate_limit_parameter(self):
        """Test validator accepts rate_limit parameter."""
        validator = DOIValidator("/data", rate_limit=2.0)
        assert validator.rate_limit == 2.0


class TestValueConsistencyChecker:
    """Test ValueConsistencyChecker."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = ValueConsistencyChecker("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = ValueConsistencyChecker("/data")
        assert validator.name == "Value Consistency Checking"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ValueConsistencyChecker(tmpdir)
            report = validator.validate()
            assert isinstance(report, ValidationReport)


class TestDuplicatePrimarySourceChecker:
    """Test DuplicatePrimarySourceChecker."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = DuplicatePrimarySourceChecker("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = DuplicatePrimarySourceChecker("/data")
        assert validator.name == "Duplicate Primary Sources Check"

    def test_validate_returns_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DuplicatePrimarySourceChecker(tmpdir)
            report = validator.validate()
            assert isinstance(report, ValidationReport)


class TestSnippetSourceManualVerifier:
    """Test SnippetSourceManualVerifier."""

    def test_inherits_from_validator(self):
        """Test inherits from Validator base class."""
        validator = SnippetSourceManualVerifier("/data")
        assert isinstance(validator, Validator)

    def test_has_name_property(self):
        """Test has correct name property."""
        validator = SnippetSourceManualVerifier("/data")
        assert validator.name == "Manual Snippet Source Verification"

    def test_has_validate_method(self):
        """Test validator has validate() method."""
        validator = SnippetSourceManualVerifier("/data")
        assert hasattr(validator, "validate")
        assert callable(validator.validate)

    def test_has_verify_interactive_method(self):
        """Test validator still has verify_interactive() method."""
        validator = SnippetSourceManualVerifier("/data")
        assert hasattr(validator, "verify_interactive")
        assert callable(validator.verify_interactive)
