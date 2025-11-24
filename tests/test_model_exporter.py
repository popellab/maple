"""
Test that ModelDefinitionExporter has required methods and functionality.

These tests catch issues like missing export_to_json method or incorrect
initialization parameters.
"""

import pytest
import tempfile
from pathlib import Path

from qsp_llm_workflows.core.model_definition_exporter import ModelDefinitionExporter


def test_model_exporter_has_export_to_json_method():
    """Test that ModelDefinitionExporter has export_to_json method."""
    # Create a mock model file path
    mock_model_file = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
    mock_model_file.close()

    try:
        exporter = ModelDefinitionExporter(mock_model_file.name)

        # Check method exists
        assert hasattr(
            exporter, "export_to_json"
        ), "ModelDefinitionExporter is missing export_to_json method"

        # Check it's callable
        assert callable(exporter.export_to_json), "export_to_json is not callable"

    finally:
        Path(mock_model_file.name).unlink()


def test_model_exporter_has_export_definitions_method():
    """Test that ModelDefinitionExporter has export_definitions method."""
    mock_model_file = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
    mock_model_file.close()

    try:
        exporter = ModelDefinitionExporter(mock_model_file.name)

        assert hasattr(
            exporter, "export_definitions"
        ), "ModelDefinitionExporter is missing export_definitions method"
        assert callable(exporter.export_definitions), "export_definitions is not callable"

    finally:
        Path(mock_model_file.name).unlink()


def test_model_exporter_requires_matlab_model_file():
    """Test that ModelDefinitionExporter requires matlab_model_file argument."""
    with pytest.raises(TypeError):
        # Should fail if called without arguments
        ModelDefinitionExporter()


def test_model_exporter_validates_file_exists():
    """Test that ModelDefinitionExporter validates that the model file exists."""
    nonexistent_file = "/tmp/nonexistent_model_file_12345.m"

    with pytest.raises(ValueError, match="does not exist"):
        ModelDefinitionExporter(nonexistent_file)


def test_model_exporter_initialization():
    """Test that ModelDefinitionExporter initializes correctly."""
    mock_model_file = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
    mock_model_file.close()

    try:
        exporter = ModelDefinitionExporter(mock_model_file.name)

        # Check that attributes are set
        assert hasattr(exporter, "model_file")
        assert exporter.model_file.exists()
        assert exporter.model_file.suffix == ".m"
        assert exporter.model_type == "matlab_script"  # Default type

    finally:
        Path(mock_model_file.name).unlink()


def test_model_exporter_finds_matlab_export_script():
    """Test that ModelDefinitionExporter can find the MATLAB export script."""
    mock_model_file = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
    mock_model_file.close()

    try:
        # Check that the MATLAB script path is constructed correctly
        # The script should look for src/qsp_llm_workflows/matlab/export_model_definitions.m
        from qsp_llm_workflows.core.resource_utils import get_package_root

        package_root = get_package_root()
        expected_script = package_root / "matlab" / "export_model_definitions.m"

        assert expected_script.exists(), f"MATLAB export script not found at {expected_script}"

        # Verify exporter can be instantiated
        _ = ModelDefinitionExporter(mock_model_file.name)

    finally:
        Path(mock_model_file.name).unlink()


def test_model_exporter_supports_simbiology_project():
    """Test that ModelDefinitionExporter supports SimBiology project files."""
    mock_project_file = tempfile.NamedTemporaryFile(suffix=".sbproj", delete=False)
    mock_project_file.close()

    try:
        exporter = ModelDefinitionExporter(mock_project_file.name, model_type="simbiology_project")

        # Check that attributes are set correctly
        assert hasattr(exporter, "model_file")
        assert exporter.model_file.exists()
        assert exporter.model_file.suffix == ".sbproj"
        assert exporter.model_type == "simbiology_project"

    finally:
        Path(mock_project_file.name).unlink()


def test_model_exporter_invalid_model_type():
    """Test that ModelDefinitionExporter rejects invalid model types."""
    mock_model_file = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
    mock_model_file.close()

    try:
        with pytest.raises(ValueError, match="Invalid model_type"):
            ModelDefinitionExporter(mock_model_file.name, model_type="invalid_type")

    finally:
        Path(mock_model_file.name).unlink()


def test_matlab_script_handles_sbioloadproject_struct():
    """
    Regression test: Verify MATLAB script correctly handles sbioloadproject return value.

    sbioloadproject returns a struct with dynamic field names (e.g., 'mw_sbmod1'),
    not a Project object with a 'Models' property.

    This test verifies the MATLAB script uses:
        proj = sbioloadproject(file);
        modelFields = fieldnames(proj);
        model = proj.(modelFields{1});

    Rather than the incorrect:
        proj.Models  % This field doesn't exist!
    """
    from qsp_llm_workflows.core.resource_utils import get_package_root

    # Read the MATLAB export script
    package_root = get_package_root()
    matlab_script = package_root / "matlab" / "export_model_definitions.m"

    with open(matlab_script, "r") as f:
        script_content = f.read()

    # Verify correct struct access pattern is used
    assert (
        "fieldnames(proj)" in script_content
    ), "MATLAB script should use fieldnames(proj) to get model names from struct"
    assert (
        "proj.(modelFields{1})" in script_content or "proj.(fields{1})" in script_content
    ), "MATLAB script should use dynamic field access proj.(fieldNames{1}) to get model"

    # Verify incorrect pattern is NOT used
    assert "proj.Models" not in script_content, (
        "MATLAB script should NOT use proj.Models - sbioloadproject returns a struct, "
        "not a Project object with Models property"
    )
