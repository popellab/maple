#!/usr/bin/env python3
"""
Package resource access using importlib.resources.

Provides functions to read prompts, templates, and configs from the package.
Works correctly with all installation methods (editable, wheel, ZIP).
"""
from pathlib import Path
from importlib.resources import files, as_file


def read_prompt(prompt_name: str) -> str:
    """
    Read a prompt file from the prompts/ directory.

    Args:
        prompt_name: Name of prompt file (e.g., 'qsp_parameter_extraction_prompt.md')

    Returns:
        Prompt text content
    """
    prompts = files("qsp_llm_workflows").joinpath("prompts")
    prompt_file = prompts / prompt_name
    return prompt_file.read_text(encoding="utf-8")


def read_template(template_name: str) -> str:
    """
    Read a template file from the templates/ directory.

    Args:
        template_name: Name of template file (e.g., 'parameter_metadata_template.yaml')

    Returns:
        Template text content
    """
    templates = files("qsp_llm_workflows").joinpath("templates")
    template_file = templates / template_name
    return template_file.read_text(encoding="utf-8")


def read_config(config_name: str) -> str:
    """
    Read a config file from the templates/configs/ directory.

    Args:
        config_name: Name of config file (e.g., 'prompt_assembly.yaml')

    Returns:
        Config text content
    """
    configs = files("qsp_llm_workflows").joinpath("templates", "configs")
    config_file = configs / config_name
    return config_file.read_text(encoding="utf-8")


def read_shared_prompt(shared_name: str) -> str:
    """
    Read a shared prompt file from the prompts/shared/ directory.

    Args:
        shared_name: Name of shared prompt file (e.g., 'source_and_validation_rubrics.md')

    Returns:
        Shared prompt text content
    """
    shared = files("qsp_llm_workflows").joinpath("prompts", "shared")
    shared_file = shared / shared_name
    return shared_file.read_text(encoding="utf-8")


def get_package_root() -> Path:
    """
    Get the package root directory as a filesystem path.

    Note: This extracts resources to disk if needed (e.g., from ZIP).
    Use read_* functions above for text resources when possible.

    Returns:
        Path to the package root directory
    """
    package = files("qsp_llm_workflows")
    with as_file(package) as path:
        return Path(path)


def get_config_path(config_name: str) -> Path:
    """
    Get filesystem path to a config file.

    Args:
        config_name: Name of config file

    Returns:
        Path to the config file
    """
    return get_package_root() / "templates" / "configs" / config_name


def get_template_path(template_name: str) -> Path:
    """
    Get filesystem path to a template file.

    Args:
        template_name: Name of template file

    Returns:
        Path to the template file
    """
    return get_package_root() / "templates" / template_name
