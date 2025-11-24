"""
Simple prompt building functions.

Replaces complex YAML-based prompt assembly with straightforward string substitution.
"""

from qsp_llm_workflows.core.resource_utils import read_prompt, read_shared_prompt


def build_parameter_extraction_prompt(
    parameter_info: str,
    model_context: str,
    used_primary_studies: str = "",
) -> str:
    """
    Build parameter extraction prompt with substitutions.

    Args:
        parameter_info: Formatted parameter information (name, units, description)
        model_context: Mathematical role and biological context
        used_primary_studies: List of already-used studies (optional)

    Returns:
        Complete prompt with all placeholders replaced
    """
    # Load base prompt
    prompt = read_prompt("qsp_parameter_extraction_prompt.md")

    # Load shared rubrics
    rubrics = read_shared_prompt("source_and_validation_rubrics.md")

    # Substitute placeholders
    prompt = prompt.replace("{{PARAMETER_INFO}}", parameter_info)
    prompt = prompt.replace("{{MODEL_CONTEXT}}", model_context)
    prompt = prompt.replace("{{SOURCE_AND_VALIDATION_RUBRICS}}", rubrics)
    prompt = prompt.replace("{{USED_PRIMARY_STUDIES}}", used_primary_studies)

    return prompt


def build_test_statistic_prompt(
    model_context: str,
    scenario_context: str,
    required_species_with_units: str,
    derived_species_description: str,
    used_primary_studies: str = "",
) -> str:
    """
    Build test statistic prompt with substitutions.

    Args:
        model_context: Model structure and relevant variables
        scenario_context: Experimental scenario and dosing context
        required_species_with_units: Required species with units
        derived_species_description: Description of derived species/test statistic
        used_primary_studies: List of already-used studies (optional)

    Returns:
        Complete prompt with all placeholders replaced
    """
    # Load base prompt
    prompt = read_prompt("test_statistic_prompt.md")

    # Load shared rubrics
    rubrics = read_shared_prompt("source_and_validation_rubrics.md")

    # Substitute placeholders
    prompt = prompt.replace("{{MODEL_CONTEXT}}", model_context)
    prompt = prompt.replace("{{SCENARIO_CONTEXT}}", scenario_context)
    prompt = prompt.replace("{{REQUIRED_SPECIES_WITH_UNITS}}", required_species_with_units)
    prompt = prompt.replace("{{DERIVED_SPECIES_DESCRIPTION}}", derived_species_description)
    prompt = prompt.replace("{{SOURCE_AND_VALIDATION_RUBRICS}}", rubrics)
    prompt = prompt.replace("{{USED_PRIMARY_STUDIES}}", used_primary_studies)

    return prompt


def build_validation_fix_prompt(
    yaml_content: str,
    validation_errors: str,
) -> str:
    """
    Build validation fix prompt with substitutions.

    Args:
        yaml_content: The original YAML content that failed validation
        validation_errors: Detailed validation error messages

    Returns:
        Complete prompt with all placeholders replaced
    """
    # Load base prompt
    prompt = read_prompt("validation_fix_prompt.md")

    # Substitute placeholders
    prompt = prompt.replace("{{YAML_CONTENT}}", yaml_content)
    prompt = prompt.replace("{{VALIDATION_ERRORS}}", validation_errors)

    return prompt
