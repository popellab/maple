"""
Test that the package is installed correctly and core modules can be imported.
"""
import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import qsp_llm_workflows
    assert qsp_llm_workflows is not None


def test_core_modules_import():
    """Test that core modules can be imported."""
    from qsp_llm_workflows.core import batch_creator
    from qsp_llm_workflows.core import prompt_assembly
    from qsp_llm_workflows.core import header_utils
    from qsp_llm_workflows.core import resource_utils
    from qsp_llm_workflows.core import parameter_utils
    from qsp_llm_workflows.core import workflow_orchestrator

    assert batch_creator is not None
    assert prompt_assembly is not None
    assert header_utils is not None
    assert resource_utils is not None
    assert parameter_utils is not None
    assert workflow_orchestrator is not None


def test_cli_modules_import():
    """Test that CLI modules can be imported."""
    from qsp_llm_workflows.cli import extract
    from qsp_llm_workflows.cli import validate
    from qsp_llm_workflows.cli import fix
    from qsp_llm_workflows.cli import enrich
    from qsp_llm_workflows.cli import export_model
    from qsp_llm_workflows.cli import monitor

    assert extract is not None
    assert validate is not None
    assert fix is not None
    assert enrich is not None
    assert export_model is not None
    assert monitor is not None


def test_prepare_modules_import():
    """Test that prepare modules can be imported."""
    from qsp_llm_workflows.prepare import create_parameter_batch
    from qsp_llm_workflows.prepare import create_test_statistic_batch
    from qsp_llm_workflows.prepare import enrich_parameter_csv
    from qsp_llm_workflows.prepare import enrich_test_statistic_csv

    assert create_parameter_batch is not None
    assert create_test_statistic_batch is not None
    assert enrich_parameter_csv is not None
    assert enrich_test_statistic_csv is not None


def test_run_modules_import():
    """Test that run modules can be imported."""
    from qsp_llm_workflows.run import upload_batch
    from qsp_llm_workflows.run import upload_immediate
    from qsp_llm_workflows.run import batch_monitor

    assert upload_batch is not None
    assert upload_immediate is not None
    assert batch_monitor is not None


def test_process_modules_import():
    """Test that process modules can be imported."""
    from qsp_llm_workflows.process import unpack_results

    assert unpack_results is not None


def test_validate_modules_import():
    """Test that validate modules can be imported."""
    from qsp_llm_workflows.validate import run_all_validations
    from qsp_llm_workflows.validate import check_schema_compliance
    from qsp_llm_workflows.validate import test_code_execution
    from qsp_llm_workflows.validate import check_text_snippets
    from qsp_llm_workflows.validate import check_source_references
    from qsp_llm_workflows.validate import check_doi_validity

    assert run_all_validations is not None
    assert check_schema_compliance is not None
    assert test_code_execution is not None
    assert check_text_snippets is not None
    assert check_source_references is not None
    assert check_doi_validity is not None


def test_resource_access():
    """Test that package resources (templates, prompts) can be accessed."""
    from qsp_llm_workflows.core.resource_utils import (
        get_package_root,
        get_template_path,
        get_prompt_path,
        get_config_path
    )

    # Test that resource functions work
    package_root = get_package_root()
    assert package_root.exists()

    # Test template access
    template_path = get_template_path("parameter_metadata_template.yaml")
    assert template_path.exists()

    # Test prompt access
    prompt_path = get_prompt_path("qsp_parameter_extraction_prompt.md")
    assert prompt_path.exists()

    # Test config access
    config_path = get_config_path("prompt_assembly.yaml")
    assert config_path.exists()


def test_key_classes_instantiation():
    """Test that key classes can be instantiated without errors."""
    from qsp_llm_workflows.core.prompt_assembly import PromptAssembler
    from qsp_llm_workflows.core.header_utils import HeaderManager
    from qsp_llm_workflows.core.resource_utils import get_package_root

    base_dir = get_package_root()

    # Test PromptAssembler instantiation
    assembler = PromptAssembler(base_dir)
    assert assembler is not None
    assert assembler.base_dir == base_dir

    # Test HeaderManager instantiation
    header_manager = HeaderManager(base_dir)
    assert header_manager is not None
    assert header_manager.base_dir == base_dir
