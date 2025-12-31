"""
Unit tests for ValidationFixPromptBuilder.

Tests the validation fix batch creation logic in isolation.
"""

import json
import pytest
from unittest.mock import patch

from qsp_llm_workflows.prepare.create_validation_fix_batch import ValidationFixPromptBuilder
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic


class TestValidationFixPromptBuilder:
    """Test ValidationFixPromptBuilder."""

    @pytest.fixture
    def validation_report_data(self):
        """Create sample validation report data."""
        return {
            "summary": {
                "name": "Code Execution Testing",
                "total": 2,
                "passed": 0,
                "failed": 2,
                "warnings": 0,
                "pass_rate": 0.0,
            },
            "passed": [],
            "failed": [
                {
                    "item": "k_APC_mat_PDAC_8d98da17_deriv001.yaml",
                    "reason": "Execution error: string indices must be integers, not 'str'",
                },
                {
                    "item": "k_C1_growth_PDAC_04e798b1_deriv001.yaml",
                    "reason": "Execution error: string indices must be integers, not 'str'",
                },
            ],
            "warnings": [],
        }

    @pytest.fixture
    def setup_validation_reports(self, tmp_path, validation_report_data):
        """Setup validation reports directory with test data."""
        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()

        # Create individual validation report
        report_file = validation_dir / "code_execution_testing.json"
        with open(report_file, "w") as f:
            json.dump(validation_report_data, f)

        # Create master summary (should be skipped)
        master_file = validation_dir / "master_validation_summary.json"
        with open(master_file, "w") as f:
            json.dump({"timestamp": "2025-11-24", "validations": []}, f)

        return validation_dir

    @pytest.fixture
    def setup_yaml_files(self, tmp_path):
        """Setup YAML files to fix."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        yaml1 = data_dir / "k_APC_mat_PDAC_8d98da17_deriv001.yaml"
        yaml1.write_text(
            """schema_version: v3
parameter_name: k_APC_mat
parameter_units: per_day
parameter_definition: APC maturation rate
cancer_type: PDAC
tags: []
derivation_id: deriv001
derivation_timestamp: '2025-01-01T00:00:00'
model_context:
  reactions: []
  rules: []
context_hash: 8d98da17
mathematical_role: Rate parameter
parameter_range: positive_reals
study_overview: Test study
study_design: Test design
parameter_estimates:
  inputs: []
  derivation_code: 'median = 1.5'
  median: 1.5
  iqr: 0.1
  ci95: [1.4, 1.6]
  units: per_day
key_assumptions: []
derivation_explanation: Simple calculation
key_study_limitations: None
primary_data_sources: []
secondary_data_sources: []
biological_relevance:
  species_match:
    value: 1.0
    justification: Human
  system_match:
    value: 1.0
    justification: Match
  overall_confidence:
    value: 1.0
    justification: High
  indication_match:
    value: 1.0
    justification: Match
  regimen_match:
    value: 1.0
    justification: Match
  biomarker_population_match:
    value: 1.0
    justification: Match
  stage_burden_match:
    value: 1.0
    justification: Match
"""
        )

        yaml2 = data_dir / "k_C1_growth_PDAC_04e798b1_deriv001.yaml"
        yaml2.write_text(
            """schema_version: v3
parameter_name: k_C1_growth
parameter_units: per_day
parameter_definition: C1 cell growth rate
cancer_type: PDAC
tags: []
derivation_id: deriv001
derivation_timestamp: '2025-01-01T00:00:00'
model_context:
  reactions: []
  rules: []
context_hash: 04e798b1
mathematical_role: Growth rate parameter
parameter_range: positive_reals
study_overview: Test study
study_design: Test design
parameter_estimates:
  inputs: []
  derivation_code: 'median = 0.5'
  median: 0.5
  iqr: 0.05
  ci95: [0.45, 0.55]
  units: per_day
key_assumptions: []
derivation_explanation: Simple calculation
key_study_limitations: None
primary_data_sources: []
secondary_data_sources: []
biological_relevance:
  species_match:
    value: 1.0
    justification: Human
  system_match:
    value: 1.0
    justification: Match
  overall_confidence:
    value: 1.0
    justification: High
  indication_match:
    value: 1.0
    justification: Match
  regimen_match:
    value: 1.0
    justification: Match
  biomarker_population_match:
    value: 1.0
    justification: Match
  stage_burden_match:
    value: 1.0
    justification: Match
"""
        )

        return data_dir

    def test_load_validation_reports(
        self, tmp_path, setup_validation_reports, validation_report_data
    ):
        """Test loading validation reports and grouping errors by file."""
        # Setup
        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(tmp_path / "data"),
            validation_results_dir=str(setup_validation_reports),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        # Execute
        errors_by_file = creator.load_validation_reports()

        # Verify
        assert len(errors_by_file) == 2
        assert "k_APC_mat_PDAC_8d98da17_deriv001.yaml" in errors_by_file
        assert "k_C1_growth_PDAC_04e798b1_deriv001.yaml" in errors_by_file

        # Check error formatting includes validator name
        error1 = errors_by_file["k_APC_mat_PDAC_8d98da17_deriv001.yaml"][0]
        assert error1.startswith("[Code Execution Testing]")
        assert "string indices must be integers" in error1

    def test_load_validation_reports_skips_master_summary(self, tmp_path, setup_validation_reports):
        """Test that master summary is skipped when loading reports."""
        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(tmp_path / "data"),
            validation_results_dir=str(setup_validation_reports),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        errors_by_file = creator.load_validation_reports()

        # Should only have errors from code_execution_testing.json, not master summary
        assert len(errors_by_file) == 2

    def test_load_validation_reports_handles_paths_in_filenames(self, tmp_path):
        """Test that file paths in validation reports are stripped to filename."""
        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()

        # Create report with path in filename
        report_data = {
            "summary": {"name": "Test Validator"},
            "failed": [
                {
                    "item": "/full/path/to/file.yaml",
                    "reason": "Test error",
                }
            ],
        }

        report_file = validation_dir / "test_validator.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f)

        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(tmp_path / "data"),
            validation_results_dir=str(validation_dir),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        errors_by_file = creator.load_validation_reports()

        # Should extract just the filename
        assert "file.yaml" in errors_by_file
        assert "/full/path/to/file.yaml" not in errors_by_file

    @patch("qsp_llm_workflows.prepare.create_validation_fix_batch.build_validation_fix_prompt")
    def test_create_batch_requests(
        self,
        mock_build_prompt,
        tmp_path,
        setup_validation_reports,
        setup_yaml_files,
    ):
        """Test creating batch requests from validation errors."""
        # Setup
        mock_build_prompt.return_value = "assembled prompt"

        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(setup_yaml_files),
            validation_results_dir=str(setup_validation_reports),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        # Execute
        requests = creator.create_batch_requests()

        # Verify
        assert len(requests) == 2
        assert all("custom_id" in req for req in requests)
        assert all("prompt" in req for req in requests)
        assert all("pydantic_model" in req for req in requests)

        # Check custom IDs (should have "fix_" prefix)
        custom_ids = [req["custom_id"] for req in requests]
        assert "fix_k_APC_mat_PDAC_8d98da17_deriv001" in custom_ids
        assert "fix_k_C1_growth_PDAC_04e798b1_deriv001" in custom_ids

        # Verify prompt assembly was called with correct arguments
        assert mock_build_prompt.call_count == 2

    def test_create_batch_requests_no_errors(self, tmp_path, setup_yaml_files):
        """Test creating batch requests when no validation errors exist."""
        # Setup validation dir with no error reports
        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()

        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(setup_yaml_files),
            validation_results_dir=str(validation_dir),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        # Execute
        requests = creator.create_batch_requests()

        # Verify
        assert len(requests) == 0

    @patch("qsp_llm_workflows.prepare.create_validation_fix_batch.build_validation_fix_prompt")
    def test_create_batch_requests_skips_files_without_errors(
        self, mock_build_prompt, tmp_path, setup_validation_reports, setup_yaml_files
    ):
        """Test that files without validation errors are skipped."""
        # Setup mocks
        mock_build_prompt.return_value = "assembled prompt"

        # Add a YAML file that's not in validation reports
        extra_file = setup_yaml_files / "k_extra_param_PDAC_12345678_deriv001.yaml"
        extra_file.write_text(
            """schema_version: v3
parameter_name: k_extra
parameter_units: per_day
parameter_definition: Extra parameter
cancer_type: PDAC
tags: []
derivation_id: deriv001
derivation_timestamp: '2025-01-01T00:00:00'
model_context:
  reactions: []
  rules: []
context_hash: 12345678
mathematical_role: Test parameter
parameter_range: positive_reals
study_overview: Test study
study_design: Test design
parameter_estimates:
  inputs: []
  derivation_code: 'median = 1.0'
  median: 1.0
  iqr: 0.1
  ci95: [0.9, 1.1]
  units: per_day
key_assumptions: []
derivation_explanation: Simple calculation
key_study_limitations: None
primary_data_sources: []
secondary_data_sources: []
biological_relevance:
  species_match:
    value: 1.0
    justification: Human
  system_match:
    value: 1.0
    justification: Match
  overall_confidence:
    value: 1.0
    justification: High
  indication_match:
    value: 1.0
    justification: Match
  regimen_match:
    value: 1.0
    justification: Match
  biomarker_population_match:
    value: 1.0
    justification: Match
  stage_burden_match:
    value: 1.0
    justification: Match
"""
        )

        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(setup_yaml_files),
            validation_results_dir=str(setup_validation_reports),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        # Execute
        requests = creator.create_batch_requests()

        # Verify - should only create requests for files with errors
        assert len(requests) == 2
        custom_ids = [req["custom_id"] for req in requests]
        assert "fix_k_extra_param_PDAC_12345678_deriv001" not in custom_ids

    def test_parameter_workflow_type(self, tmp_path):
        """Test that parameter workflow type is correctly determined."""
        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(tmp_path),
            validation_results_dir=str(tmp_path),
            output_file=str(output_file),
            model_class=ParameterMetadata,
        )

        assert creator.workflow_type == "parameter"

    def test_test_statistic_workflow_type(self, tmp_path):
        """Test that test_statistic workflow type is correctly determined."""
        output_file = tmp_path / "output.jsonl"
        creator = ValidationFixPromptBuilder(
            data_dir=str(tmp_path),
            validation_results_dir=str(tmp_path),
            output_file=str(output_file),
            model_class=TestStatistic,
        )

        assert creator.workflow_type == "test_statistic"

    def test_unknown_model_class(self, tmp_path):
        """Test error handling for unknown model class."""
        output_file = tmp_path / "output.jsonl"

        with pytest.raises(ValueError, match="Unknown model class"):
            ValidationFixPromptBuilder(
                data_dir=str(tmp_path),
                validation_results_dir=str(tmp_path),
                output_file=str(output_file),
                model_class=str,  # Invalid model class
            )
