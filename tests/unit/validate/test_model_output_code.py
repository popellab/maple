"""
Tests for ModelOutputCodeValidator.
"""

import yaml

from qsp_llm_workflows.validate.check_model_output_code import ModelOutputCodeValidator


class TestModelOutputCodeValidator:
    """Test model output code validation for test statistics."""

    def test_valid_compute_test_statistic_function(self, tmp_path):
        """Test that valid compute_test_statistic function passes."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
import numpy as np

def compute_test_statistic(time, species_dict):
    tumor_vol = species_dict['V_T.TumorVolume']
    # Get value at day 14
    idx = np.argmin(np.abs(time - 14))
    return float(tumor_vol[idx])
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Valid" in message

    def test_missing_model_output_code(self, tmp_path):
        """Test that missing model_output.code fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "Missing model_output.code" in message

    def test_wrong_function_name(self, tmp_path):
        """Test that wrong function name fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
def wrong_function_name(time, species_dict):
    return 1.0
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "compute_test_statistic" in message

    def test_wrong_first_argument_name(self, tmp_path):
        """Test that wrong first argument name fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
def compute_test_statistic(t, species_dict):
    return 1.0
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "time" in message
        assert "'t'" in message

    def test_wrong_second_argument_name(self, tmp_path):
        """Test that wrong second argument name fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
def compute_test_statistic(time, data):
    return 1.0
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "species_dict" in message
        assert "'data'" in message

    def test_function_returns_array_fails(self, tmp_path):
        """Test that function returning array (not scalar) fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
import numpy as np

def compute_test_statistic(time, species_dict):
    return species_dict['V_T.TumorVolume']  # Returns array, not scalar
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "array" in message.lower() or "scalar" in message.lower()

    def test_function_returns_nan_fails(self, tmp_path):
        """Test that function returning NaN fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
import numpy as np

def compute_test_statistic(time, species_dict):
    return float('nan')
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "NaN" in message

    def test_skips_parameter_estimates(self, tmp_path):
        """Test that parameter estimate files are skipped."""
        yaml_file = tmp_path / "param.yaml"
        data = {
            "parameter_name": "k_growth",
            "parameter_estimates": {"derivation_code": "median = 1.0"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Skipped" in message

    def test_missing_required_species_fails(self, tmp_path):
        """Test that missing required_species fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": [],
            "model_output": {
                "code": """
def compute_test_statistic(time, species_dict):
    return 1.0
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "required_species" in message

    def test_code_with_syntax_error_fails(self, tmp_path):
        """Test that code with syntax error fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
def compute_test_statistic(time, species_dict)
    return 1.0  # Missing colon above
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "execution failed" in message.lower() or "error" in message.lower()

    def test_validate_returns_report(self, tmp_path):
        """Test that validate() returns a proper ValidationReport."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output": {
                "code": """
import numpy as np

def compute_test_statistic(time, species_dict):
    return 1.0
"""
            },
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        report = validator.validate()

        assert report.name == "Model Output Code Validation"
        assert len(report.passed) == 1
        assert len(report.failed) == 0
