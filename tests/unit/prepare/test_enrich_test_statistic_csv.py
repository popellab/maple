"""
Unit tests for test statistic CSV enrichment with Pint unit validation.

Tests the CodeUnitValidator class and enrichment functions.
"""

import pytest

import yaml

from qsp_llm_workflows.prepare.enrich_test_statistic_csv import (
    CodeUnitValidator,
    load_species_units,
    enrich_test_statistic_csv,
)


class TestCodeUnitValidator:
    """Test the CodeUnitValidator class."""

    @pytest.fixture
    def species_units(self):
        """Sample species units matching PDAC model."""
        return {
            "V_T.TGFb": "nanomolarity",
            "V_T.CCL2": "nanomolarity",
            "V_T.C1": "cell",
            "V_T.Treg": "cell",
            "V_T.CD8": "cell",
            "V_T.Th": "cell",
            "V_T.IFNg": "nanomolarity",
        }

    def test_valid_passthrough_nanomolarity(self, species_units):
        """Test valid code that passes through nanomolarity species."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_valid_passthrough_cell(self, species_units):
        """Test valid code that passes through cell species."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.C1'][0]"""

        validator = CodeUnitValidator(code, "cell", species_units)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_valid_dimensionless_ratio(self, species_units):
        """Test valid code that computes a dimensionless ratio."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    tregs = species_dict['V_T.Treg'][0]
    cd8 = species_dict['V_T.CD8'][0]
    return tregs / cd8"""

        validator = CodeUnitValidator(code, "dimensionless", species_units)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_valid_unit_conversion(self, species_units):
        """Test valid code with explicit unit conversion."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    import numpy as np
    cells = species_dict['V_T.C1']
    return cells[0] * (1e-6 * ureg.millimeter**3 / ureg.cell)"""

        validator = CodeUnitValidator(code, "millimeter ** 3", species_units)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_valid_growth_rate(self, species_units):
        """Test valid code computing growth rate (1/day)."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    import numpy as np
    t = time.magnitude
    cells = species_dict['V_T.C1'].magnitude
    mask = (t >= 0) & (t <= 60)
    slope, _ = np.polyfit(t[mask], np.log(cells[mask]), 1)
    return slope * (1 / ureg.day)"""

        validator = CodeUnitValidator(code, "1 / day", species_units)
        assert validator.validate() is True
        assert len(validator.errors) == 0

    def test_invalid_output_unit_mismatch(self, species_units):
        """Test that wrong output units are rejected."""
        # Code returns cell, but output_unit says nanomolarity
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.C1'][0]"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert len(validator.errors) == 1
        assert "unit mismatch" in validator.errors[0].lower()

    def test_invalid_missing_species(self, species_units):
        """Test that accessing non-existent species is rejected."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.NonExistent'][0]"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert len(validator.errors) == 1
        assert "species" in validator.errors[0].lower()

    def test_invalid_wrong_signature_missing_ureg(self, species_units):
        """Test that wrong function signature (missing ureg) is rejected."""
        code = """def compute_test_statistic(time, species_dict):
    return species_dict['V_T.TGFb'][0]"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert len(validator.errors) == 1
        assert "signature" in validator.errors[0].lower()

    def test_invalid_wrong_signature_wrong_params(self, species_units):
        """Test that wrong parameter names are rejected."""
        code = """def compute_test_statistic(t, data, units):
    return data['V_T.TGFb'][0]"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert "signature" in validator.errors[0].lower()

    def test_invalid_wrong_function_name(self, species_units):
        """Test that wrong function name is rejected."""
        code = """def calculate_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert "compute_test_statistic" in validator.errors[0]

    def test_invalid_syntax_error(self, species_units):
        """Test that syntax errors are caught."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0"""  # Missing bracket

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert "syntax" in validator.errors[0].lower()

    def test_invalid_no_species_access(self, species_units):
        """Test that code not accessing species_dict is rejected."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return 42.0 * ureg.nanomolar"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is False
        assert "species" in validator.errors[0].lower()

    def test_invalid_returns_non_quantity(self, species_units):
        """Test that returning a non-Pint quantity is rejected."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.C1'][0].magnitude"""  # Strips units

        validator = CodeUnitValidator(code, "cell", species_units)
        assert validator.validate() is False
        assert "pint" in validator.errors[0].lower() or "quantity" in validator.errors[0].lower()

    def test_invalid_unknown_output_unit(self, species_units):
        """Test that invalid Pint unit strings are rejected."""
        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"""

        validator = CodeUnitValidator(code, "invalid_unit_xyz", species_units)
        assert validator.validate() is False
        assert "invalid" in validator.errors[0].lower() or "unknown" in validator.errors[0].lower()

    def test_cleans_markdown_code_fences(self, species_units):
        """Test that markdown code fences are stripped."""
        code = """```python
def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]
```"""

        validator = CodeUnitValidator(code, "nanomolarity", species_units)
        assert validator.validate() is True

    def test_molarity_alias(self, species_units):
        """Test that molarity/molar aliases work."""
        species_units_with_molarity = {**species_units, "V_T.P0": "molarity"}

        code = """def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.P0'][0]"""

        validator = CodeUnitValidator(code, "molarity", species_units_with_molarity)
        assert validator.validate() is True


class TestLoadSpeciesUnits:
    """Test the load_species_units function."""

    def test_load_json_simple(self, tmp_path):
        """Test loading simple JSON format."""
        json_file = tmp_path / "species.json"
        json_file.write_text('{"V_T.TGFb": "nanomolarity", "V_T.C1": "cell"}')

        result = load_species_units(json_file)

        assert result["V_T.TGFb"] == "nanomolarity"
        assert result["V_T.C1"] == "cell"

    def test_load_json_nested(self, tmp_path):
        """Test loading nested JSON format with units key."""
        json_file = tmp_path / "species.json"
        json_file.write_text('{"V_T.TGFb": {"units": "nanomolarity"}, "V_T.C1": {"units": "cell"}}')

        result = load_species_units(json_file)

        assert result["V_T.TGFb"] == "nanomolarity"
        assert result["V_T.C1"] == "cell"

    def test_load_csv(self, tmp_path):
        """Test loading CSV format."""
        csv_file = tmp_path / "species.csv"
        csv_file.write_text(
            "Name,Units,Compartment,Notes\nTGFb,nanomolarity,V_T,TGF-beta\nC1,cell,V_T,Cancer cells"
        )

        result = load_species_units(csv_file)

        # Should have both simple and qualified names
        assert result["TGFb"] == "nanomolarity"
        assert result["V_T.TGFb"] == "nanomolarity"
        assert result["C1"] == "cell"
        assert result["V_T.C1"] == "cell"

    def test_load_csv_missing_units_defaults_dimensionless(self, tmp_path):
        """Test that missing units default to dimensionless."""
        csv_file = tmp_path / "species.csv"
        csv_file.write_text("Name,Units,Compartment\nFoo,,V_T")

        result = load_species_units(csv_file)

        assert result["Foo"] == "dimensionless"

    def test_unsupported_format_raises(self, tmp_path):
        """Test that unsupported file formats raise ValueError."""
        txt_file = tmp_path / "species.txt"
        txt_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported"):
            load_species_units(txt_file)


class TestEnrichTestStatisticCsv:
    """Test the full enrichment function."""

    @pytest.fixture
    def species_json(self, tmp_path):
        """Create a species units JSON file."""
        species_file = tmp_path / "species_units.json"
        species_file.write_text(
            """{
            "V_T.TGFb": "nanomolarity",
            "V_T.CCL2": "nanomolarity",
            "V_T.C1": "cell",
            "V_T.Treg": "cell",
            "V_T.CD8": "cell"
        }"""
        )
        return species_file

    @pytest.fixture
    def scenario_yaml(self, tmp_path):
        """Create a scenario YAML file."""
        scenario_file = tmp_path / "scenario.yaml"
        scenario_content = {"scenario_context": "Baseline PDAC progression without treatment."}
        with open(scenario_file, "w") as f:
            yaml.dump(scenario_content, f)
        return scenario_file

    def test_enrichment_succeeds_with_valid_csv(self, tmp_path, species_json, scenario_yaml):
        """Test successful enrichment with valid input."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            """test_statistic_id,output_unit,model_output_code
tgfb_baseline,nanomolarity,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"
"""
        )
        output_csv = tmp_path / "output.csv"

        enrich_test_statistic_csv(input_csv, scenario_yaml, species_json, output_csv)

        assert output_csv.exists()

        # Check output content
        import pandas as pd

        df = pd.read_csv(output_csv)
        assert len(df) == 1
        assert "scenario_context" in df.columns
        assert "context_hash" in df.columns
        assert df.iloc[0]["test_statistic_id"] == "tgfb_baseline"

    def test_enrichment_fails_with_invalid_units(self, tmp_path, species_json, scenario_yaml):
        """Test that enrichment fails with unit mismatch."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            """test_statistic_id,output_unit,model_output_code
bad_units,nanomolarity,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.C1'][0]"
"""
        )
        output_csv = tmp_path / "output.csv"

        with pytest.raises(ValueError, match="validation error"):
            enrich_test_statistic_csv(input_csv, scenario_yaml, species_json, output_csv)

    def test_enrichment_fails_with_missing_columns(self, tmp_path, species_json, scenario_yaml):
        """Test that enrichment fails with missing required columns."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            """test_statistic_id,output_unit
tgfb_baseline,nanomolarity
"""
        )  # Missing model_output_code
        output_csv = tmp_path / "output.csv"

        with pytest.raises(ValueError, match="Missing required columns"):
            enrich_test_statistic_csv(input_csv, scenario_yaml, species_json, output_csv)

    def test_enrichment_multiple_valid_rows(self, tmp_path, species_json, scenario_yaml):
        """Test enrichment with multiple valid rows."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            """test_statistic_id,output_unit,model_output_code
tgfb_baseline,nanomolarity,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"
treg_cd8_ratio,dimensionless,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.Treg'][0] / species_dict['V_T.CD8'][0]"
"""
        )
        output_csv = tmp_path / "output.csv"

        enrich_test_statistic_csv(input_csv, scenario_yaml, species_json, output_csv)

        import pandas as pd

        df = pd.read_csv(output_csv)
        assert len(df) == 2

    def test_context_hash_is_consistent(self, tmp_path, species_json, scenario_yaml):
        """Test that context_hash is the same for all rows (same scenario)."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text(
            """test_statistic_id,output_unit,model_output_code
stat1,nanomolarity,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"
stat2,nanomolarity,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.CCL2'][0]"
"""
        )
        output_csv = tmp_path / "output.csv"

        enrich_test_statistic_csv(input_csv, scenario_yaml, species_json, output_csv)

        import pandas as pd

        df = pd.read_csv(output_csv)
        assert df.iloc[0]["context_hash"] == df.iloc[1]["context_hash"]
