"""
Unit tests for unpack_results module.

Tests the unpacking logic for both batch and immediate mode responses.
"""

import json
import pytest
import yaml

from qsp_llm_workflows.process.unpack_results import process_results, add_header_fields


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


class TestTestStatisticHeaders:
    """Test test statistic header handling including required_species."""

    def test_add_header_fields_includes_required_species(self):
        """Test that add_header_fields parses required_species to list."""
        json_data = {}
        metadata = {
            "test_statistic_id": "tumor_volume_day14",
            "cancer_type": "PDAC",
            "context_hash": "abc123",
            "model_context": "Test model",
            "scenario_context": "Test scenario",
            "required_species": "V_T.CD8, V_T.Treg, V_T.TumorVolume",
            "derived_species_description": "Tumor volume at day 14",
        }

        result = add_header_fields(json_data, metadata, "test_statistic")

        # Check required_species is parsed to list
        assert "required_species" in result
        assert isinstance(result["required_species"], list)
        assert len(result["required_species"]) == 3
        assert "V_T.CD8" in result["required_species"]
        assert "V_T.Treg" in result["required_species"]
        assert "V_T.TumorVolume" in result["required_species"]

        # Check derived_species_description
        assert result["derived_species_description"] == "Tumor volume at day 14"

    def test_add_header_fields_raises_error_for_empty_required_species(self):
        """Test that empty required_species raises ValueError."""
        json_data = {}
        metadata = {
            "test_statistic_id": "test_stat",
            "cancer_type": "PDAC",
            "context_hash": "abc123",
            "required_species": "",
            "derived_species_description": "Some description",
        }

        with pytest.raises(ValueError, match="required_species is required"):
            add_header_fields(json_data, metadata, "test_statistic")

    def test_add_header_fields_raises_error_for_empty_derived_species_description(self):
        """Test that empty derived_species_description raises ValueError."""
        json_data = {}
        metadata = {
            "test_statistic_id": "test_stat",
            "cancer_type": "PDAC",
            "context_hash": "abc123",
            "required_species": "V_T.CD8",
            "derived_species_description": "",
        }

        with pytest.raises(ValueError, match="derived_species_description is required"):
            add_header_fields(json_data, metadata, "test_statistic")

    def test_test_statistic_unpacking_with_required_species(self, tmp_path):
        """Test full unpacking of test statistic with required_species in header."""
        # Create input CSV with required_species
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            "test_statistic_id,cancer_type,context_hash,model_context,scenario_context,required_species,derived_species_description\n"
            'tumor_volume,PDAC,abc123,Model context,Scenario context,"V_T.CD8,V_T.Treg",Tumor volume ratio\n'
        )

        # Create results file
        results_file = tmp_path / "results.jsonl"
        result = {
            "custom_id": "test_stat_tumor_volume_0",
            "response": {
                "status_code": 200,
                "body": {
                    "output_parsed": {
                        "model_output": {"code": "return 1.0"},
                        "test_statistic_definition": "Test definition",
                        "study_overview": "Test overview",
                        "study_design": "Test design",
                        "test_statistic_estimates": {
                            "inputs": [],
                            "derivation_code": "median = 1.0",
                            "median": 1.0,
                            "iqr": 0.1,
                            "ci95": [0.9, 1.1],
                            "units": "ratio",
                            "key_assumptions": {"1": "Assumption 1"},
                        },
                        "derivation_explanation": "Test explanation",
                        "key_study_limitations": "Test limitations",
                        "primary_data_sources": [],
                        "secondary_data_sources": [],
                        "validation_weights": {
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

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Process results
        process_results(results_file, output_dir, input_csv)

        # Find the output file
        output_files = list(output_dir.glob("*.yaml"))
        assert len(output_files) == 1

        # Read and parse the output
        content = output_files[0].read_text()
        data = yaml.safe_load(content)

        # Verify required_species is in header as list
        assert "required_species" in data
        assert isinstance(data["required_species"], list)
        assert "V_T.CD8" in data["required_species"]
        assert "V_T.Treg" in data["required_species"]

        # Verify derived_species_description
        assert data["derived_species_description"] == "Tumor volume ratio"

        # Verify header field ordering (required_species should be near top)
        lines = content.split("\n")
        header_fields = [
            "schema_version",
            "test_statistic_id",
            "cancer_type",
            "context_hash",
        ]
        for field in header_fields:
            assert any(line.startswith(f"{field}:") for line in lines[:15])
