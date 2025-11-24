"""
Test that required package resources exist and are accessible.

These tests catch issues like missing MATLAB scripts, templates, or prompts
that would cause runtime errors.
"""
import pytest
from pathlib import Path
from importlib import resources

from qsp_llm_workflows.core.resource_utils import get_package_root


def test_matlab_export_script_exists():
    """Test that MATLAB export script exists in package."""
    # Get package root
    package_root = get_package_root()
    matlab_dir = package_root / "matlab"
    export_script = matlab_dir / "export_model_definitions.m"

    assert matlab_dir.exists(), f"MATLAB directory not found: {matlab_dir}"
    assert export_script.exists(), f"MATLAB export script not found: {export_script}"

    # Verify it's a valid file with content
    assert export_script.stat().st_size > 0, "MATLAB export script is empty"


def test_matlab_helper_script_exists():
    """Test that MATLAB helper script exists in package."""
    package_root = get_package_root()
    helper_script = package_root / "matlab" / "parameterReactionTableExtended.m"

    assert helper_script.exists(), f"MATLAB helper script not found: {helper_script}"
    assert helper_script.stat().st_size > 0, "MATLAB helper script is empty"


def test_parameter_prompt_exists():
    """Test that parameter extraction prompt exists."""
    package_root = get_package_root()
    prompt_file = package_root / "prompts" / "qsp_parameter_extraction_prompt.md"

    assert prompt_file.exists(), f"Parameter prompt not found: {prompt_file}"
    assert prompt_file.stat().st_size > 0, "Parameter prompt is empty"


def test_test_statistic_prompt_exists():
    """Test that test statistic prompt exists."""
    package_root = get_package_root()
    prompt_file = package_root / "prompts" / "test_statistic_prompt.md"

    assert prompt_file.exists(), f"Test statistic prompt not found: {prompt_file}"
    assert prompt_file.stat().st_size > 0, "Test statistic prompt is empty"


def test_validation_fix_prompt_exists():
    """Test that validation fix prompt exists."""
    package_root = get_package_root()
    prompt_file = package_root / "prompts" / "validation_fix_prompt.md"

    assert prompt_file.exists(), f"Validation fix prompt not found: {prompt_file}"
    assert prompt_file.stat().st_size > 0, "Validation fix prompt is empty"


def test_prompt_assembly_config_exists():
    """Test that prompt assembly config exists."""
    package_root = get_package_root()
    config_file = package_root / "templates" / "configs" / "prompt_assembly.yaml"

    assert config_file.exists(), f"Prompt assembly config not found: {config_file}"
    assert config_file.stat().st_size > 0, "Prompt assembly config is empty"


def test_header_fields_config_exists():
    """Test that header fields config exists."""
    package_root = get_package_root()
    config_file = package_root / "templates" / "configs" / "header_fields.yaml"

    assert config_file.exists(), f"Header fields config not found: {config_file}"
    assert config_file.stat().st_size > 0, "Header fields config is empty"


def test_all_matlab_scripts_are_readable():
    """Test that all MATLAB scripts can be read."""
    package_root = get_package_root()
    matlab_dir = package_root / "matlab"

    # Find all .m files
    matlab_files = list(matlab_dir.glob("*.m"))

    assert len(matlab_files) > 0, "No MATLAB files found"

    for matlab_file in matlab_files:
        with open(matlab_file, 'r') as f:
            content = f.read()
            assert len(content) > 0, f"MATLAB file is empty: {matlab_file}"
            assert "function" in content, f"MATLAB file doesn't contain 'function': {matlab_file}"


def test_all_templates_are_valid_yaml():
    """Test that all template files are valid YAML."""
    import yaml

    package_root = get_package_root()
    template_dir = package_root / "templates"

    # Find all .yaml files
    yaml_files = list(template_dir.glob("**/*.yaml"))

    assert len(yaml_files) > 0, "No YAML templates found"

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in {yaml_file}: {e}")


def test_all_prompts_are_readable():
    """Test that all prompt files can be read."""
    package_root = get_package_root()
    prompts_dir = package_root / "prompts"

    # Find all .md files
    prompt_files = list(prompts_dir.glob("*.md"))

    assert len(prompt_files) > 0, "No prompt files found"

    for prompt_file in prompt_files:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0, f"Prompt file is empty: {prompt_file}"
