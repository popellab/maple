"""
Unit tests for unpack_results module.

Tests the unpacking logic for both batch and immediate mode responses.
"""

import json
import pytest

from qsp_llm_workflows.process.unpack_results import process_results


class TestUnpackResults:
    """Test unpacking of batch and immediate mode results."""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary output directory."""
        output = tmp_path / "output"
        output.mkdir()
        return output

    @pytest.fixture
    def immediate_results_with_output_parsed(self, tmp_path):
        """Create immediate mode results file with output_parsed wrapper."""
        results_file = tmp_path / "immediate_results.jsonl"

        # Immediate mode response format with output_parsed wrapper
        result = {
            "id": "immediate_req_1",
            "custom_id": "fix_test_param_PDAC_abc123_deriv001",
            "response": {
                "status_code": 200,
                "body": {
                    "output_parsed": {
                        "mathematical_role": "Test role",
                        "parameter_range": "positive_reals",
                        "study_overview": "Test overview",
                        "study_design": "Test design",
                        "parameter_estimates": {
                            "inputs": [],
                            "derivation_code": "median = 1.0",
                            "median": 1.0,
                            "iqr": 0.1,
                            "ci95": [0.9, 1.1],
                            "units": "1/day",
                        },
                        "key_assumptions": [],
                        "derivation_explanation": "Test explanation",
                        "key_study_limitations": "Test limitations",
                        "primary_data_sources": [],
                        "secondary_data_sources": [],
                        "methodological_sources": [],
                        "biological_relevance": {
                            "species_match": {"value": 1.0, "justification": "Test"},
                            "system_match": {"value": 1.0, "justification": "Test"},
                            "overall_confidence": {"value": 1.0, "justification": "Test"},
                            "indication_match": {"value": 1.0, "justification": "Test"},
                            "regimen_match": {"value": 1.0, "justification": "Test"},
                            "biomarker_population_match": {
                                "value": 1.0,
                                "justification": "Test",
                            },
                            "stage_burden_match": {"value": 1.0, "justification": "Test"},
                        },
                    }
                },
            },
        }

        with open(results_file, "w") as f:
            f.write(json.dumps(result) + "\n")

        return results_file

    @pytest.fixture
    def immediate_results_without_wrapper(self, tmp_path):
        """Create immediate mode results file without output_parsed wrapper (legacy format)."""
        results_file = tmp_path / "immediate_results_legacy.jsonl"

        # Legacy immediate mode response format (direct content)
        result = {
            "id": "immediate_req_1",
            "custom_id": "fix_test_param_PDAC_abc123_deriv001",
            "response": {
                "status_code": 200,
                "body": {
                    "mathematical_role": "Test role",
                    "parameter_range": "positive_reals",
                    "study_overview": "Test overview",
                    "study_design": "Test design",
                    "parameter_estimates": {
                        "inputs": [],
                        "derivation_code": "median = 1.0",
                        "median": 1.0,
                        "iqr": 0.1,
                        "ci95": [0.9, 1.1],
                        "units": "1/day",
                    },
                    "key_assumptions": [],
                    "derivation_explanation": "Test explanation",
                    "key_study_limitations": "Test limitations",
                    "primary_data_sources": [],
                    "secondary_data_sources": [],
                    "methodological_sources": [],
                    "biological_relevance": {
                        "species_match": {"value": 1.0, "justification": "Test"},
                        "system_match": {"value": 1.0, "justification": "Test"},
                        "overall_confidence": {"value": 1.0, "justification": "Test"},
                        "indication_match": {"value": 1.0, "justification": "Test"},
                        "regimen_match": {"value": 1.0, "justification": "Test"},
                        "biomarker_population_match": {"value": 1.0, "justification": "Test"},
                        "stage_burden_match": {"value": 1.0, "justification": "Test"},
                    },
                },
            },
        }

        with open(results_file, "w") as f:
            f.write(json.dumps(result) + "\n")

        return results_file

    @pytest.fixture
    def original_yaml_file(self, output_dir):
        """Create original YAML file for validation fix testing."""
        yaml_file = output_dir / "test_param_PDAC_abc123_deriv001.yaml"

        content = """schema_version: v3
parameter_name: test_param
parameter_units: 1/day
parameter_definition: Test parameter
cancer_type: PDAC
tags: []
derivation_id: deriv001
derivation_timestamp: '2025-01-01T00:00:00'
model_context:
  reactions: []
  rules: []
context_hash: abc123
mathematical_role: Old role
parameter_range: positive_reals
study_overview: Old overview
study_design: Old design
parameter_estimates:
  inputs: []
  derivation_code: 'median = 0.5'
  median: 0.5
  iqr: 0.05
  ci95: [0.45, 0.55]
  units: 1/day
key_assumptions: []
derivation_explanation: Old explanation
key_study_limitations: Old limitations
primary_data_sources: []
secondary_data_sources: []
methodological_sources: []
biological_relevance:
  species_match:
    value: 0.8
    justification: Old
  system_match:
    value: 0.8
    justification: Old
  overall_confidence:
    value: 0.8
    justification: Old
  indication_match:
    value: 0.8
    justification: Old
  regimen_match:
    value: 0.8
    justification: Old
  biomarker_population_match:
    value: 0.8
    justification: Old
  stage_burden_match:
    value: 0.8
    justification: Old
"""
        yaml_file.write_text(content)
        return yaml_file

    def test_immediate_mode_extracts_output_parsed(
        self, immediate_results_with_output_parsed, output_dir, original_yaml_file
    ):
        """Test that immediate mode correctly extracts content from output_parsed wrapper."""
        # Process results
        process_results(immediate_results_with_output_parsed, output_dir, input_csv=None)

        # Check that file was created
        output_file = output_dir / "test_param_PDAC_abc123_deriv001.yaml"
        assert output_file.exists()

        # Read the output file
        content = output_file.read_text()

        # Verify that output_parsed is NOT in the file (bug fix)
        assert "output_parsed:" not in content

        # Verify that content fields are at top level
        assert "mathematical_role: Test role" in content
        assert "parameter_range: positive_reals" in content
        assert "study_overview: Test overview" in content

        # Verify headers were preserved
        assert "schema_version: v3" in content
        assert "parameter_name: test_param" in content
        assert "cancer_type: PDAC" in content

    def test_immediate_mode_without_wrapper_still_works(
        self, immediate_results_without_wrapper, output_dir, original_yaml_file
    ):
        """Test that immediate mode without output_parsed wrapper still works (legacy format)."""
        # Process results
        process_results(immediate_results_without_wrapper, output_dir, input_csv=None)

        # Check that file was created
        output_file = output_dir / "test_param_PDAC_abc123_deriv001.yaml"
        assert output_file.exists()

        # Read the output file
        content = output_file.read_text()

        # Verify that content fields are at top level
        assert "mathematical_role: Test role" in content
        assert "parameter_range: positive_reals" in content

    def test_validation_fix_preserves_headers(
        self, immediate_results_with_output_parsed, output_dir, original_yaml_file
    ):
        """Test that validation fixes preserve original headers."""
        # Process results
        process_results(immediate_results_with_output_parsed, output_dir, input_csv=None)

        # Read the output file
        output_file = output_dir / "test_param_PDAC_abc123_deriv001.yaml"
        content = output_file.read_text()

        # Verify that headers were preserved from original file
        assert "schema_version: v3" in content
        assert "parameter_name: test_param" in content
        assert "parameter_units: 1/day" in content
        assert "cancer_type: PDAC" in content
        assert "context_hash: abc123" in content

        # Verify that content was updated from fixed response
        assert "mathematical_role: Test role" in content
        assert "study_overview: Test overview" in content

        # Verify old content was replaced
        assert "Old role" not in content
        assert "Old overview" not in content
