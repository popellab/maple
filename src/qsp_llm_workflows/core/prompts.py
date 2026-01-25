"""
Simple prompt building functions.

Replaces complex YAML-based prompt assembly with straightforward string substitution.
"""

from qsp_llm_workflows.core.resource_utils import read_prompt, read_shared_prompt


def build_parameter_extraction_prompt(
    parameter_info: str,
    model_context: str,
    cancer_type: str,
    used_primary_studies: str = "",
) -> str:
    """
    Build parameter extraction prompt with substitutions.

    Args:
        parameter_info: Formatted parameter information (name, units, description)
        model_context: Mathematical role and biological context
        cancer_type: Cancer type/indication (e.g., "PDAC", "melanoma")
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
    prompt = prompt.replace("{{CANCER_TYPE}}", cancer_type)
    prompt = prompt.replace("{{SOURCE_AND_VALIDATION_RUBRICS}}", rubrics)
    prompt = prompt.replace("{{USED_PRIMARY_STUDIES}}", used_primary_studies)

    return prompt


def build_test_statistic_prompt(
    model_context: str,
    scenario_context: str,
    required_species_with_units: str,
    derived_species_description: str,
    cancer_type: str,
    used_primary_studies: str = "",
) -> str:
    """
    Build test statistic prompt with substitutions.

    Args:
        model_context: Model structure and relevant variables
        scenario_context: Experimental scenario and dosing context
        required_species_with_units: Required species with units
        derived_species_description: Description of derived species/test statistic
        cancer_type: Cancer type/indication (e.g., "PDAC", "melanoma")
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
    prompt = prompt.replace("{{CANCER_TYPE}}", cancer_type)
    prompt = prompt.replace("{{SOURCE_AND_VALIDATION_RUBRICS}}", rubrics)
    prompt = prompt.replace("{{USED_PRIMARY_STUDIES}}", used_primary_studies)

    return prompt


def build_calibration_target_prompt(
    observable_description: str,
    cancer_type: str,
    model_species: str,
    model_indication: str,
    model_compartment: str,
    model_system: str,
    model_treatment_history: str,
    model_stage_burden: str,
    model_species_with_units: str,
    used_primary_studies: str = "",
    primary_source_title: str = "",
) -> str:
    """
    Build calibration target extraction prompt with substitutions.

    Args:
        observable_description: Description of observable to extract
        cancer_type: Cancer type/indication (e.g., "PDAC")
        model_species: Model species (e.g., "human")
        model_indication: Model indication (e.g., "PDAC")
        model_compartment: Model compartment (e.g., "tumor.primary")
        model_system: Model system (e.g., "clinical.resection")
        model_treatment_history: Model treatment history (e.g., "treatment_naive")
        model_stage_burden: Model stage/burden (e.g., "resectable")
        model_species_with_units: Formatted list of available model species with units
        used_primary_studies: Formatted list of already-used primary studies (optional)
        primary_source_title: Title of specific paper to extract from (optional, skips web search)

    Returns:
        Complete prompt with all placeholders replaced
    """
    # Load base prompt
    prompt = read_prompt("calibration_target_prompt.md")

    # Build source instruction based on whether a specific source is provided
    if primary_source_title and primary_source_title.strip():
        source_instruction = (
            f"**Extract from this specific paper:** {primary_source_title}\n\n"
            f"Do NOT search for other papers. Use ONLY this source as your primary data source.\n\n"
        )
    else:
        source_instruction = (
            f"**Find 1 peer-reviewed paper** reporting this observable in {cancer_type}.\n\n"
        )

    # Substitute placeholders
    prompt = prompt.replace("{{OBSERVABLE_DESCRIPTION}}", observable_description)
    prompt = prompt.replace("{{CANCER_TYPE}}", cancer_type)
    prompt = prompt.replace("{{MODEL_SPECIES}}", model_species)
    prompt = prompt.replace("{{MODEL_INDICATION}}", model_indication)
    prompt = prompt.replace("{{MODEL_COMPARTMENT}}", model_compartment)
    prompt = prompt.replace("{{MODEL_SYSTEM}}", model_system)
    prompt = prompt.replace("{{MODEL_TREATMENT_HISTORY}}", model_treatment_history)
    prompt = prompt.replace("{{MODEL_STAGE_BURDEN}}", model_stage_burden)
    prompt = prompt.replace("{{MODEL_SPECIES_WITH_UNITS}}", model_species_with_units)
    prompt = prompt.replace("{{USED_PRIMARY_STUDIES}}", used_primary_studies)
    prompt = prompt.replace("{{PRIMARY_SOURCE_TITLE}}", source_instruction)

    return prompt


def build_isolated_system_target_prompt(
    parameters: str,
    model_context: str,
    parameter_context: str = "",
    notes: str = "",
) -> str:
    """
    Build isolated system target extraction prompt.

    Args:
        parameters: Comma-separated parameter names to calibrate (e.g., "k_CD8_pro,k_CD8_death")
        model_context: High-level model description (from model_context.txt)
        parameter_context: Rich context for each parameter (reactions, species, etc.)
        notes: Optional notes/guidance for the extraction

    Returns:
        Complete prompt with placeholders replaced
    """
    prompt = read_prompt("isolated_system_target_prompt.md")

    prompt = prompt.replace("{{PARAMETERS}}", parameters)
    prompt = prompt.replace("{{MODEL_CONTEXT}}", model_context)
    prompt = prompt.replace(
        "{{PARAMETER_CONTEXT}}", parameter_context or "No parameter context available."
    )

    # Handle optional notes with mustache-style conditional
    if notes and notes.strip():
        prompt = prompt.replace("{{#NOTES}}", "")
        prompt = prompt.replace("{{/NOTES}}", "")
        prompt = prompt.replace("{{NOTES}}", notes)
    else:
        # Remove the entire notes line if empty
        import re

        prompt = re.sub(r"\{\{#NOTES\}\}.*?\{\{/NOTES\}\}\n?", "", prompt, flags=re.DOTALL)

    return prompt


def build_submodel_target_prompt(
    parameters: str,
    model_context: str,
    parameter_context: str = "",
    notes: str = "",
) -> str:
    """
    Build submodel target extraction prompt.

    Args:
        parameters: Comma-separated parameter names to calibrate (e.g., "k_CD8_pro,k_CD8_death")
        model_context: High-level model description (from model_context.txt)
        parameter_context: Rich context for each parameter (reactions, species, etc.)
        notes: Optional notes/guidance for the extraction

    Returns:
        Complete prompt with placeholders replaced
    """
    prompt = read_prompt("submodel_target_prompt.md")

    prompt = prompt.replace("{{PARAMETERS}}", parameters)
    prompt = prompt.replace("{{MODEL_CONTEXT}}", model_context)
    prompt = prompt.replace(
        "{{PARAMETER_CONTEXT}}", parameter_context or "No parameter context available."
    )

    # Handle optional notes with mustache-style conditional
    if notes and notes.strip():
        prompt = prompt.replace("{{#NOTES}}", "")
        prompt = prompt.replace("{{/NOTES}}", "")
        prompt = prompt.replace("{{NOTES}}", notes)
    else:
        # Remove the entire notes line if empty
        import re

        prompt = re.sub(r"\{\{#NOTES\}\}.*?\{\{/NOTES\}\}\n?", "", prompt, flags=re.DOTALL)

    return prompt
