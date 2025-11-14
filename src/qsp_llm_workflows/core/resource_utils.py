"""
Utilities for accessing package resources (templates, prompts, configs).

Uses importlib.resources for robust access across all installation types.
"""

from pathlib import Path
try:
    from importlib.resources import files
except ImportError:
    # Fallback for Python < 3.9
    from importlib_resources import files


def get_package_root() -> Path:
    """
    Get the root directory of the qsp_llm_workflows package.

    Returns:
        Path to the package root directory containing templates/, prompts/, etc.
    """
    return Path(str(files('qsp_llm_workflows')))


def get_template_path(template_name: str) -> Path:
    """
    Get path to a template file.

    Args:
        template_name: Name of template file (e.g., 'parameter_metadata_template_v3.yaml')

    Returns:
        Path to the template file
    """
    return get_package_root() / 'templates' / template_name


def get_config_path(config_name: str) -> Path:
    """
    Get path to a config file.

    Args:
        config_name: Name of config file (e.g., 'prompt_assembly.yaml')

    Returns:
        Path to the config file
    """
    return get_package_root() / 'templates' / 'configs' / config_name


def get_prompt_path(prompt_name: str) -> Path:
    """
    Get path to a prompt file.

    Args:
        prompt_name: Name of prompt file (e.g., 'parameter_prompt.md')

    Returns:
        Path to the prompt file
    """
    return get_package_root() / 'prompts' / prompt_name


def get_example_path(example_name: str) -> Path:
    """
    Get path to an example template file.

    Args:
        example_name: Name of example file (e.g., 'parameter_example.yaml')

    Returns:
        Path to the example file
    """
    return get_package_root() / 'templates' / 'examples' / example_name
