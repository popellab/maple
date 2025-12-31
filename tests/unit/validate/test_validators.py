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
                "parameter_units": "1 / day",
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5],
                    "derivation_code": """
import numpy as np
def derive_parameter(inputs, ureg):
    return {
        "median_param": 1.0 / ureg.day,
        "iqr_param": 0.5 / ureg.day,
        "ci95_param": [0.5 / ureg.day, 1.5 / ureg.day]
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

            validator = CodeExecutionValidator(tmpdir, interactive=False)
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

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "error" in report.failed[0]["reason"].lower()

    def test_fails_when_returning_raw_floats_not_pint(self):
        """Test validator fails when code returns raw floats instead of Pint Quantities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_units": "1 / day",
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5],
                    "derivation_code": """
def derive_parameter(inputs, ureg):
    # Wrong: returning raw floats instead of Pint Quantities
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

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "pint" in report.failed[0]["reason"].lower()

    def test_fails_when_unit_dimensionality_mismatch(self):
        """Test validator fails when returned units don't match expected dimensionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_units": "nanomolar",  # Expects concentration
                "parameter_estimates": {
                    "median": 1.0,
                    "iqr": 0.5,
                    "ci95": [0.5, 1.5],
                    "derivation_code": """
def derive_parameter(inputs, ureg):
    # Wrong: returning rate (1/day) when concentration (nanomolar) expected
    return {
        "median_param": 1.0 / ureg.day,
        "iqr_param": 0.5 / ureg.day,
        "ci95_param": [0.5 / ureg.day, 1.5 / ureg.day]
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

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            report = validator.validate()

            assert len(report.failed) == 1
            assert "mismatch" in report.failed[0]["reason"].lower()

    def test_passes_test_statistic_with_pint_quantities(self):
        """Test validator passes test statistic derivation with Pint Quantities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "output_unit": "nanomolar",
                "test_statistic_estimates": {
                    "median": 50.0,
                    "iqr": 10.0,
                    "ci95": [40.0, 60.0],
                    "derivation_code": """
def derive_distribution(inputs, ureg):
    return {
        "median_stat": 50.0 * ureg.nanomolar,
        "iqr_stat": 10.0 * ureg.nanomolar,
        "ci95_stat": [40.0 * ureg.nanomolar, 60.0 * ureg.nanomolar]
    }
""",
                    "inputs": [
                        {
                            "name": "concentration",
                            "value": 50.0,
                            "units": "nanomolar",
                            "description": "Measured concentration",
                            "source_ref": "src1",
                        }
                    ],
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            report = validator.validate()

            assert len(report.failed) == 0

    def test_passes_dimensionless_parameter(self):
        """Test validator passes dimensionless parameters (ratios, fractions)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_units": "dimensionless",
                "parameter_estimates": {
                    "median": 0.5,
                    "iqr": 0.1,
                    "ci95": [0.3, 0.7],
                    "derivation_code": """
def derive_parameter(inputs, ureg):
    # Dimensionless ratio
    return {
        "median_param": 0.5 * ureg.dimensionless,
        "iqr_param": 0.1 * ureg.dimensionless,
        "ci95_param": [0.3 * ureg.dimensionless, 0.7 * ureg.dimensionless]
    }
""",
                    "inputs": [
                        {
                            "name": "fraction",
                            "value": 0.5,
                            "units": "dimensionless",
                            "description": "Fraction",
                            "source_ref": "src1",
                        }
                    ],
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            report = validator.validate()

            assert len(report.failed) == 0

    def test_stores_computed_values_for_executable_files(self):
        """Test that computed values are stored for all files with executable code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_units": "1 / day",
                "parameter_estimates": {
                    "median": 99.0,  # Wrong value - should be 2.0
                    "iqr": 99.0,  # Wrong value - should be 1.0
                    "ci95": [99.0, 99.0],  # Wrong values - should be [1.0, 3.0]
                    "derivation_code": """
def derive_parameter(inputs, ureg):
    return {
        "median_param": 2.0 / ureg.day,
        "iqr_param": 1.0 / ureg.day,
        "ci95_param": [1.0 / ureg.day, 3.0 / ureg.day]
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

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            validator.validate()

            # Check that computed values were stored
            assert len(validator.executable_files) == 1
            filepath = str(yaml_file)
            assert filepath in validator.executable_files

            stored = validator.executable_files[filepath]
            assert stored["code_type"] == "parameter"
            assert stored["computed_values"]["median"] == 2.0
            assert stored["computed_values"]["iqr"] == 1.0
            assert stored["computed_values"]["ci95"] == [1.0, 3.0]
            assert stored["is_valid"] is False  # Values don't match

            # Check that current YAML values were also stored
            assert stored["current_values"]["median"] == 99.0
            assert stored["current_values"]["iqr"] == 99.0
            assert stored["current_values"]["ci95"] == [99.0, 99.0]

    def test_update_yaml_values_inline_ci95(self):
        """Test _update_yaml_values correctly updates YAML with inline ci95 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            # Write YAML with inline ci95 format
            content = """parameter_estimates:
  inputs:
    - name: input1
      value: 1.0
  derivation_code: |
    def derive_parameter(inputs):
        return {"median_param": 1.0, "iqr_param": 0.5, "ci95_param": [0.5, 1.5]}
  median: 1.0
  iqr: 0.5
  ci95: [0.5, 1.5]
  units: per_day
"""
            with open(yaml_file, "w") as f:
                f.write(content)

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            computed = {"median": 2.5, "iqr": 1.2, "ci95": [0.8, 4.2]}

            success = validator._update_yaml_values(str(yaml_file), "parameter", computed)
            assert success is True

            # Read back and verify
            with open(yaml_file, "r") as f:
                updated = f.read()

            assert "median: 2.5" in updated
            assert "iqr: 1.2" in updated
            assert "[0.8, 4.2]" in updated

    def test_update_yaml_values_multiline_ci95(self):
        """Test _update_yaml_values correctly updates YAML with multiline ci95 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            # Write YAML with multiline ci95 format
            content = """parameter_estimates:
  inputs:
    - name: input1
      value: 1.0
  derivation_code: |
    def derive_parameter(inputs):
        return {"median_param": 1.0, "iqr_param": 0.5, "ci95_param": [0.5, 1.5]}
  median: 1.0
  iqr: 0.5
  ci95:
    - 0.5
    - 1.5
  units: per_day
"""
            with open(yaml_file, "w") as f:
                f.write(content)

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            computed = {"median": 3.0, "iqr": 2.0, "ci95": [1.5, 4.5]}

            success = validator._update_yaml_values(str(yaml_file), "parameter", computed)
            assert success is True

            # Read back and verify
            with open(yaml_file, "r") as f:
                updated = f.read()

            assert "median: 3" in updated
            assert "iqr: 2" in updated
            assert "- 1.5" in updated
            assert "- 4.5" in updated

    def test_update_yaml_values_test_statistic(self):
        """Test _update_yaml_values works for test_statistic_estimates section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            content = """test_statistic_estimates:
  inputs:
    - name: input1
      value: 1.0
  derivation_code: |
    def derive_distribution(inputs):
        return {"median_stat": 1.0, "iqr_stat": 0.5, "ci95_stat": [0.5, 1.5]}
  median: 1.0
  iqr: 0.5
  ci95: [0.5, 1.5]
  units: mm3
"""
            with open(yaml_file, "w") as f:
                f.write(content)

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            computed = {"median": 100.0, "iqr": 50.0, "ci95": [25.0, 175.0]}

            success = validator._update_yaml_values(str(yaml_file), "test_statistic", computed)
            assert success is True

            # Read back and verify
            with open(yaml_file, "r") as f:
                updated = f.read()

            assert "median: 100" in updated
            assert "iqr: 50" in updated
            assert "[25, 175]" in updated

    def test_update_yaml_preserves_other_content(self):
        """Test that updating values doesn't corrupt other YAML content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            content = """# Header comment
parameter_name: k_growth

parameter_estimates:
  inputs:
    - name: input1
      value: 1.0
      description: "Important input"
  derivation_code: |
    def derive_parameter(inputs):
        return {"median_param": 1.0, "iqr_param": 0.5, "ci95_param": [0.5, 1.5]}
  median: 1.0
  iqr: 0.5
  ci95: [0.5, 1.5]
  units: per_day

key_assumptions:
  - number: 1
    text: "Important assumption"

# Footer comment
"""
            with open(yaml_file, "w") as f:
                f.write(content)

            validator = CodeExecutionValidator(tmpdir, interactive=False)
            computed = {"median": 5.0, "iqr": 2.0, "ci95": [2.0, 8.0]}

            success = validator._update_yaml_values(str(yaml_file), "parameter", computed)
            assert success is True

            # Read back and verify structure is preserved
            with open(yaml_file, "r") as f:
                updated = f.read()

            # Check values were updated
            assert "median: 5" in updated
            assert "iqr: 2" in updated

            # Check other content is preserved
            assert "# Header comment" in updated
            assert "parameter_name: k_growth" in updated
            assert 'description: "Important input"' in updated
            assert "key_assumptions:" in updated
            assert '"Important assumption"' in updated
            assert "# Footer comment" in updated


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

    def test_handles_scientific_notation_in_snippets(self):
        """Test validator handles scientific notation in snippets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "concentration",
                            "value": 1.5e-6,
                            "units": "M",
                            "value_snippet": "The concentration was 1.5 × 10⁻⁶ M",
                            "source_ref": "src1",
                        }
                    ]
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()

            # Should pass - validator handles scientific notation
            assert len(report.failed) == 0

    def test_multiple_inputs_mixed_results(self):
        """Test validator with multiple inputs where some pass and some fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.234,
                            "units": "1/day",
                            "value_snippet": "The growth rate was 0.234 per day",  # Correct
                            "source_ref": "src1",
                        },
                        {
                            "name": "clearance",
                            "value": 67.89,
                            "units": "L/h",
                            "value_snippet": "Clearance was measured at 15.3 L/h",  # Wrong - different value
                            "source_ref": "src1",
                        },
                        {
                            "name": "volume",
                            "value": 400.0,
                            "units": "mL",
                            "value_snippet": "The volume was 400.0 mL",  # Correct
                            "source_ref": "src1",
                        },
                    ]
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()

            # Should have 1 failure (clearance) and 2 passes
            assert len(report.failed) == 1
            assert len(report.passed) == 2
            # Check that clearance is the one that failed
            assert "clearance" in report.failed[0]["item"]

    def test_passes_when_no_value_snippet(self):
        """Test validator skips inputs without value_snippet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "units": "1/day",
                            # No value_snippet field
                            "source_ref": "src1",
                        }
                    ]
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()

            # Should pass - no snippet to validate
            assert len(report.failed) == 0

    def test_skips_empty_snippet(self):
        """Test validator skips validation when snippet is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "units": "1/day",
                            "value_snippet": "",  # Empty snippet
                            "source_ref": "src1",
                        }
                    ]
                },
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            validator = TextSnippetValidator(tmpdir)
            report = validator.validate()

            # Empty snippets are skipped (falsy value), so no failures or passes
            assert len(report.failed) == 0
            assert len(report.passed) == 0


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

    def test_is_url_handles_different_protocols(self):
        """Test validator detects various URL protocols."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        # URLs with different protocols
        assert validator.is_url("ftp://files.example.com") is True
        assert validator.is_url("https://www.ncbi.nlm.nih.gov") is True
        assert validator.is_url("http://arxiv.org/abs/1234.5678") is True

        # DOI URLs (with https://doi.org/) are still URLs
        assert validator.is_url("https://doi.org/10.1234/test") is True

        # Bare DOI strings (without https://) with doi.org
        assert validator.is_url("doi.org/10.1234/test") is False

        # Pure DOIs (just the identifier)
        assert validator.is_url("10.1056/NEJMoa1200690") is False
        assert validator.is_url("10.1038/nature12345") is False

    def test_is_url_handles_edge_cases(self):
        """Test validator handles edge cases in URL detection."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        # Empty or None
        assert validator.is_url("") is False
        assert validator.is_url(None) is False

        # Whitespace
        assert validator.is_url("  https://example.com  ") is True
        assert validator.is_url("  10.1234/test  ") is False

    def test_fuzzy_match_exact_match(self):
        """Test fuzzy matching with exact strings."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        assert validator.fuzzy_match("test", "test") is True
        assert validator.fuzzy_match("Test Title", "Test Title") is True

    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy matching is case insensitive."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        assert validator.fuzzy_match("Test", "TEST") is True
        assert validator.fuzzy_match("Test Title", "test title") is True

    def test_fuzzy_match_whitespace_normalized(self):
        """Test fuzzy matching normalizes whitespace."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        assert validator.fuzzy_match("  test  ", "test") is True
        assert validator.fuzzy_match("test title", "test  title") is True

    def test_fuzzy_match_similar_strings(self):
        """Test fuzzy matching with similar but not identical strings."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        # Similar strings should match with default threshold (0.8)
        assert validator.fuzzy_match("A study of cancer cells", "A study of cancer cell") is True

        # Very different strings should not match
        assert validator.fuzzy_match("Cancer study", "Completely different title") is False

    def test_fuzzy_match_with_custom_threshold(self):
        """Test fuzzy matching with custom similarity threshold."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        str1 = "Testing fuzzy matching"
        str2 = "Testing fuzzy"

        # Should match with lower threshold
        assert validator.fuzzy_match(str1, str2, threshold=0.7) is True

        # Should not match with higher threshold
        assert validator.fuzzy_match(str1, str2, threshold=0.95) is False

    def test_fuzzy_match_handles_empty_strings(self):
        """Test fuzzy matching handles empty or None strings."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        assert validator.fuzzy_match("", "test") is False
        assert validator.fuzzy_match("test", "") is False
        assert validator.fuzzy_match("", "") is False
        assert validator.fuzzy_match(None, "test") is False
        assert validator.fuzzy_match("test", None) is False

    def test_collect_sources_from_primary(self):
        """Test collecting sources from primary_data_sources."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        data = {
            "primary_data_sources": [
                {
                    "source_tag": "src1",
                    "title": "Test Paper",
                    "first_author": "Smith",
                    "year": 2020,
                    "doi": "10.1234/test",
                }
            ]
        }

        sources = validator.collect_sources(data)
        assert len(sources) == 1
        assert sources[0][0] == "src1"
        assert sources[0][1]["doi"] == "10.1234/test"

    def test_collect_sources_from_secondary(self):
        """Test collecting sources from secondary_data_sources."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        data = {
            "secondary_data_sources": [
                {
                    "source_tag": "src1",
                    "title": "Reference Book",
                    "first_author": "Jones",
                    "year": 2019,
                    "doi_or_url": "10.5678/ref",
                }
            ]
        }

        sources = validator.collect_sources(data)
        assert len(sources) == 1
        assert sources[0][0] == "src1"

    def test_collect_sources_skips_missing_doi(self):
        """Test collector skips sources without DOI/URL."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        data = {
            "primary_data_sources": [
                {
                    "source_tag": "src1",
                    "title": "Paper without DOI",
                    "first_author": "Smith",
                    "year": 2020,
                    # No doi field
                }
            ]
        }

        sources = validator.collect_sources(data)
        assert len(sources) == 0

    def test_collect_sources_handles_empty_data(self):
        """Test collector handles data without source sections."""
        validator = DOIValidator("/tmp", rate_limit=0.1)

        data = {
            "parameter_estimates": {
                "median": 1.0,
                "iqr": 0.5,
                "ci95": [0.5, 1.5],
            }
        }

        sources = validator.collect_sources(data)
        assert len(sources) == 0


class TestValueConsistencyChecker:
    """Test ValueConsistencyChecker checks value consistency."""

    def test_initializes_collections(self):
        """Test validator initializes data structures correctly."""
        validator = ValueConsistencyChecker("/tmp")

        assert hasattr(validator, "derivation_groups")
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
