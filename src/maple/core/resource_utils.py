#!/usr/bin/env python3
"""
Package resource access using importlib.resources.

Provides functions to read prompts from the package.
Works correctly with all installation methods (editable, wheel, ZIP).
"""
from pathlib import Path
from importlib.resources import files, as_file


def read_prompt(prompt_name: str) -> str:
    """
    Read a prompt file from the prompts/ directory.

    Args:
        prompt_name: Name of prompt file (e.g., 'submodel_target_prompt.md')

    Returns:
        Prompt text content
    """
    prompts = files("maple").joinpath("prompts")
    prompt_file = prompts / prompt_name
    return prompt_file.read_text(encoding="utf-8")


def get_package_root() -> Path:
    """
    Get the package root directory as a filesystem path.

    Note: This extracts resources to disk if needed (e.g., from ZIP).
    Use read_prompt() for text resources when possible.

    Returns:
        Path to the package root directory
    """
    package = files("maple")
    with as_file(package) as path:
        return Path(path)
