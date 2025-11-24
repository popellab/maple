"""
Tests for individual validators.

Tests core validation logic of each validator.
"""

import tempfile
from pathlib import Path
import yaml

from qsp_llm_workflows.validate.check_schema_compliance import SchemaValidator
from qsp_llm_workflows.validate.test_code_execution import CodeExecutionValidator
from qsp_llm_workflows.validate.check_text_snippets import TextSnippetValidator
from qsp_llm_workflows.validate.check_source_references import SourceReferenceValidator
from qsp_llm_workflows.validate.check_doi_validity import DOIValidator
from qsp_llm_workflows.validate.check_value_consistency import ValueConsistencyChecker
from qsp_llm_workflows.validate.check_duplicate_primary_sources import (
    DuplicatePrimarySourceChecker,
)
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata


class TestSchemaValidator:
    """Test SchemaValidator validates schema compliance."""

    def test_passes_valid_yaml(self):
        """Test validator passes valid YAML that matches Pydantic schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            # Create complete valid parameter metadata
            data = {
                "mathematical_role": "Growth rate parameter",
                "parameter_range": "positive_reals",
                "study_overview": "Test study",
                "study_design": "Experimental",
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5],
                    "units": "1/day",
                    "derivation_code": "import numpy as np\nresult = {'mean': 1.0, 'variance': 0.25}",
                    "inputs": [
                        {
                            "name": "test_input",
                            "value": 1.0,
                            "units": "dimensionless",
                            "description": "Test input description",
                            "source_ref": "src1",
                            "value_table_or_section": "Table 1",
                            "value_snippet": "The value was 1.0",
                            "units_table_or_section": "Methods section",
                            "units_snippet": "dimensionless",
                        }
                    ],
                },
                "key_assumptions": [],
                "derivation_explanation": "Test explanation",
                "key_study_limitations": "No major limitations",
                "primary_data_sources": [
                    {
                        "source_tag": "src1",
                        "title": "Test",
                        "first_author": "Smith",
                        "year": 2020,
                        "doi": "10.1234/test",
                    }
                ],
                "secondary_data_sources": [],
                "methodological_sources": [],
                "biological_relevance": {
                    "species_match": {"value": 1.0, "justification": "Same species"},
                    "system_match": {"value": 1.0, "justification": "Same system"},
                    "overall_confidence": {"value": 0.8, "justification": "High confidence"},
                    "indication_match": {"value": 1.0, "justification": "Same indication"},
                    "regimen_match": {"value": 1.0, "justification": "Same regimen"},
                    "biomarker_population_match": {
                        "value": 0.8,
                        "justification": "Similar population",
                    },
                    "stage_burden_match": {"value": 0.9, "justification": "Similar stage"},
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = SchemaValidator(tmpdir, model_class=ParameterMetadata)
            report = validator.validate()

            assert len(report.failed) == 0
            assert len(report.passed) == 1

    def test_fails_invalid_yaml(self):
        """Test validator fails YAML with invalid schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            # Missing required fields
            data = {
                "parameter_estimates": {
                    "median": 1.0,
                    # Missing iqr, ci95, derivation_code, inputs
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = SchemaValidator(tmpdir, model_class=ParameterMetadata)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "test.yaml" in report.failed[0]["item"]

    def test_fails_wrong_ci95_format(self):
        """Test validator fails when ci95 is not a 2-element list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5, 2.0],  # Wrong: should be 2 elements
                    "derivation_code": "print('test')",
                    "inputs": [],
                },
                "primary_data_sources": [],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = SchemaValidator(tmpdir, model_class=ParameterMetadata)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "ci95" in report.failed[0]["reason"]


class TestCodeExecutionValidator:
    """Test CodeExecutionValidator executes and validates code."""

    def test_passes_valid_python_code(self):
        """Test validator passes valid Python code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5],
                    "derivation_code": """
import numpy as np
def derive_parameter(inputs):
    return {
        "median_param": 1.0,
        "iqr_param": 0.5,
        "ci95_param": [0.5, 1.5]
    }
""",
                    "inputs": [
                        {
                            "name": "input1",
                            "value": 1.0,
                            "units": "dimensionless",
                            "description": "Test input",
                            "source_ref": "src1",
                        }
                    ],
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = CodeExecutionValidator(tmpdir)
            report = validator.validate()

            assert len(report.failed) == 0

    def test_fails_syntax_error_code(self):
        """Test validator fails code with syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5],
                    "derivation_code": "def bad_syntax(\nprint('missing parenthesis')",
                    "inputs": [
                        {
                            "name": "input1",
                            "value": 1.0,
                            "units": "dimensionless",
                            "description": "Test input",
                            "source_ref": "src1",
                        }
                    ],
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = CodeExecutionValidator(tmpdir)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "error" in report.failed[0]["reason"].lower()


class TestTextSnippetValidator:
    """Test TextSnippetValidator checks values in snippets."""

    def test_passes_when_value_in_snippet(self):
        """Test validator passes when declared value is in snippet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "units": "1/day",
                            "value_snippet": "The growth rate was 0.5 per day",
                            "source_ref": "src1",
                        }
                    ]
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()

            assert len(report.failed) == 0
            assert len(report.passed) > 0

    def test_fails_when_value_not_in_snippet(self):
        """Test validator fails when declared value is not in snippet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "units": "1/day",
                            "value_snippet": "The growth rate was 1.5 per day",  # Wrong value
                            "source_ref": "src1",
                        }
                    ]
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "0.5" in report.failed[0]["reason"]


class TestSourceReferenceValidator:
    """Test SourceReferenceValidator checks source reference integrity."""

    def test_passes_when_all_refs_defined(self):
        """Test validator passes when all source_refs have definitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "source_ref": "smith2020",
                        }
                    ]
                },
                "primary_data_sources": [
                    {
                        "source_tag": "smith2020",
                        "title": "Test Paper",
                        "first_author": "Smith",
                        "year": 2020,
                        "doi": "10.1234/test",
                    }
                ],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = SourceReferenceValidator(tmpdir)
            report = validator.validate()

            assert len(report.failed) == 0

    def test_fails_when_ref_undefined(self):
        """Test validator fails when source_ref has no definition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "source_ref": "missing_source",  # Not defined
                        }
                    ]
                },
                "primary_data_sources": [],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = SourceReferenceValidator(tmpdir)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "missing_source" in report.failed[0]["reason"]


class TestDOIValidator:
    """Test DOIValidator validates DOI format and structure."""

    def test_passes_empty_directory(self):
        """Test validator passes when no DOIs to validate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DOIValidator(tmpdir, rate_limit=0.1)
            report = validator.validate()

            # Should pass with no files
            assert len(report.failed) == 0

    def test_is_url_detection(self):
        """Test validator can distinguish URLs from DOIs."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        assert validator.is_url("https://example.com") is True
        assert validator.is_url("http://example.com") is True
        assert validator.is_url("www.example.com") is True
        assert validator.is_url("10.1234/test") is False
        assert validator.is_url("doi.org/10.1234/test") is False


class TestValueConsistencyChecker:
    """Test ValueConsistencyChecker checks value consistency."""

    def test_initializes_collections(self):
        """Test validator initializes data structures correctly."""
        validator = ValueConsistencyChecker("/tmp")

        assert hasattr(validator, "context_groups")
        assert hasattr(validator, "legacy_values")
        assert hasattr(validator, "all_files")

    def test_passes_empty_directory(self):
        """Test validator passes when no files to check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ValueConsistencyChecker(tmpdir)
            report = validator.validate()

            # Should pass with no files
            assert len(report.failed) == 0


class TestDuplicatePrimarySourceChecker:
    """Test DuplicatePrimarySourceChecker detects duplicates."""

    def test_normalizes_doi(self):
        """Test validator normalizes DOIs correctly."""
        validator = DuplicatePrimarySourceChecker("/tmp")

        assert validator.normalize_doi("10.1234/TEST") == "10.1234/test"
        assert validator.normalize_doi("https://doi.org/10.1234/Test") == "10.1234/test"
        assert validator.normalize_doi("  10.1234/TEST  ") == "10.1234/test"

    def test_passes_non_review_directory(self):
        """Test validator skips validation when not in to-review directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DuplicatePrimarySourceChecker(tmpdir)
            report = validator.validate()

            # Should skip validation for non to-review directories
            assert len(report.failed) == 0
