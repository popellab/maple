"""
Tests for ModelOutputCodeValidator.
"""

import yaml

from maple.validate.check_model_output_code import ModelOutputCodeValidator


class TestModelOutputCodeValidator:
    """Test model output code validation for test statistics."""

    def test_valid_compute_test_statistic_function(self, tmp_path):
        """Test that valid compute_test_statistic function passes."""
        import json

        # Create species_units.json so mock data has correct units
        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.TumorVolume": "milliliter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
import numpy as np

def compute_test_statistic(time, species_dict, ureg):
    tumor_vol = species_dict['V_T.TumorVolume']
    # Get value at day 14
    idx = np.argmin(np.abs(time.magnitude - 14))
    return tumor_vol[idx]
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Valid" in message

    def test_missing_model_output_code(self, tmp_path):
        """Test that missing model_output_code fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "model_output_code" in message.lower()

    def test_missing_output_unit(self, tmp_path):
        """Test that missing output_unit fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    return 1.0 * ureg.milliliter
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "output_unit" in message.lower()

    def test_wrong_function_name(self, tmp_path):
        """Test that wrong function name fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
def wrong_function_name(time, species_dict, ureg):
    return 1.0 * ureg.milliliter
""",
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
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(t, species_dict, ureg):
    return 1.0 * ureg.milliliter
""",
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
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, data, ureg):
    return 1.0 * ureg.milliliter
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "species_dict" in message
        assert "'data'" in message

    def test_wrong_third_argument_name(self, tmp_path):
        """Test that wrong third argument name fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, units):
    return 1.0 * units.milliliter
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "ureg" in message
        assert "'units'" in message

    def test_function_returns_non_quantity_fails(self, tmp_path):
        """Test that function returning non-Quantity fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
import numpy as np

def compute_test_statistic(time, species_dict, ureg):
    return 1.0  # Returns float, not Quantity
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "pint.Quantity" in message or "Quantity" in message

    def test_function_returns_nan_fails(self, tmp_path):
        """Test that function returning NaN fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
import numpy as np

def compute_test_statistic(time, species_dict, ureg):
    return float('nan') * ureg.milliliter
""",
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
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    return 1.0 * ureg.milliliter
""",
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
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg)
    return 1.0 * ureg.milliliter  # Missing colon above
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "execution failed" in message.lower() or "error" in message.lower()

    def test_validate_returns_report(self, tmp_path):
        """Test that validate() returns a proper ValidationReport."""
        import json

        # Create species_units.json so mock data has correct units
        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.TumorVolume": "milliliter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
import numpy as np

def compute_test_statistic(time, species_dict, ureg):
    tumor_vol = species_dict['V_T.TumorVolume']
    return tumor_vol[0]
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        report = validator.validate()

        assert report.name == "Model Output Code Validation"
        assert len(report.passed) == 1
        assert len(report.failed) == 0

    def test_unit_mismatch_fails(self, tmp_path):
        """Test that returning wrong unit dimensionality fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",  # Expects volume
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    return 1.0 * ureg.second  # Returns time, not volume
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "mismatch" in message.lower() or "unit" in message.lower()

    def test_species_units_file_loads(self, tmp_path):
        """Test that species_units_file is loaded and used."""
        import json

        # Create species_units.json
        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.TumorVolume": "milliliter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_day14",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    tumor_vol = species_dict['V_T.TumorVolume']
    return tumor_vol[0]  # Already has milliliter units from species_units
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Valid" in message

    def test_dimensionless_output(self, tmp_path):
        """Test that dimensionless ratios validate correctly."""
        import json

        # Create species_units.json with cell counts
        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.CD8": "cell", "V_T.Treg": "cell"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "cd8_treg_ratio",
            "required_species": ["V_T.CD8", "V_T.Treg"],
            "output_unit": "dimensionless",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    cd8 = species_dict['V_T.CD8']
    treg = species_dict['V_T.Treg']
    # Ratio of cells cancels units -> dimensionless
    return (cd8[0] / treg[0]).to(ureg.dimensionless)
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Valid" in message

    def test_compartment_volume_access(self, tmp_path):
        """Test accessing compartment volumes like V_T directly."""
        import json

        # Create species_units.json with compartment volume
        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T": "milliliter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "tumor_volume_baseline",
            "required_species": ["V_T"],
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    tumor_vol = species_dict['V_T']
    return tumor_vol[0]
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Valid" in message

    def test_scalar_parameter_access(self, tmp_path):
        """Test accessing scalar parameters like initial_tumour_diameter."""
        import json

        # Create species_units.json with scalar parameter
        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.C1": "cell", "initial_tumour_diameter": "centimeter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "cell_density",
            "required_species": ["V_T.C1", "initial_tumour_diameter"],
            "output_unit": "cell / centimeter ** 3",
            "model_output_code": """
import numpy as np

def compute_test_statistic(time, species_dict, ureg):
    cells = species_dict['V_T.C1']
    diameter = species_dict['initial_tumour_diameter']
    # Calculate volume from diameter (sphere)
    radius = diameter / 2
    volume = (4/3) * np.pi * radius ** 3
    return (cells[0] / volume).to('cell / cm**3')
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is True
        assert "Valid" in message

    def test_invalid_output_unit_string(self, tmp_path):
        """Test that unparseable output_unit fails gracefully."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "test_stat",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "not_a_real_unit_xyz",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    return 1.0 * ureg.milliliter
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "parse" in message.lower() or "output_unit" in message.lower()

    def test_function_returns_inf_fails(self, tmp_path):
        """Test that function returning Inf fails."""
        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "test_stat",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    return float('inf') * ureg.milliliter
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(str(tmp_path))
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "Inf" in message

    def test_function_with_runtime_error(self, tmp_path):
        """Test that runtime errors are caught gracefully."""
        import json

        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.TumorVolume": "milliliter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "test_stat",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    # This will raise a KeyError
    return species_dict['NonExistentSpecies'][0]
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        assert is_valid is False
        assert "failed" in message.lower() or "error" in message.lower()

    def test_function_with_division_by_zero(self, tmp_path):
        """Test that division by zero is caught."""
        import json

        species_units_file = tmp_path / "species_units.json"
        with open(species_units_file, "w") as f:
            json.dump({"V_T.TumorVolume": "milliliter"}, f)

        yaml_file = tmp_path / "test_stat.yaml"
        data = {
            "test_statistic_id": "test_stat",
            "required_species": ["V_T.TumorVolume"],
            "output_unit": "milliliter",
            "model_output_code": """
def compute_test_statistic(time, species_dict, ureg):
    vol = species_dict['V_T.TumorVolume']
    return vol[0] / 0  # Division by zero
""",
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        validator = ModelOutputCodeValidator(
            str(tmp_path), species_units_file=str(species_units_file)
        )
        is_valid, message = validator.validate_file(str(yaml_file), data)

        # Division by zero with Pint creates Inf, which we catch
        assert is_valid is False
        assert "Inf" in message or "failed" in message.lower()
