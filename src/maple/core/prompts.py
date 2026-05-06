"""
Simple prompt building functions.

Replaces complex YAML-based prompt assembly with straightforward string substitution.
"""

from maple.core.resource_utils import read_prompt


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
    reference_db_entries: list[dict] | None = None,
    auxiliary_groups: list[dict] | None = None,
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
        reference_db_entries: List of reference value dicts with name, display_name, value, units, notes
        auxiliary_groups: List of auxiliary-parameter group dicts with keys
            ``name``, ``description``, ``base_prior`` (dict with ``distribution``,
            ``mu``, ``sigma``), ``member_deviation_sigma``. Listed in the prompt
            so the LLM knows which compartment / measurement bridges are
            available (and only those — never invented). Pass ``None`` or an
            empty list to render the prompt's auxiliary section as a no-op
            placeholder.

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

    # Inject reference database listing
    if reference_db_entries:
        lines = []
        for entry in reference_db_entries:
            name = entry["name"]
            display = entry.get("display_name", name)
            value = float(entry.get("value", 0))
            units = entry.get("units", "")
            value_str = f"{value:.4g}"
            lines.append(f"- `{name}`: {display} ({value_str} {units})")
        prompt = prompt.replace("{{REFERENCE_DATABASE}}", "\n".join(lines))
    else:
        prompt = prompt.replace("{{REFERENCE_DATABASE}}", "No reference database available.")

    # Inject auxiliary parameter group listing
    if auxiliary_groups:
        lines = []
        for group in auxiliary_groups:
            name = group["name"]
            description = (group.get("description") or "").strip().replace("\n", " ")
            base = group.get("base_prior") or {}
            distribution = base.get("distribution", "?")
            mu = base.get("mu")
            sigma = base.get("sigma")
            tau = group.get("member_deviation_sigma")
            mu_str = f"{float(mu):.4g}" if mu is not None else "?"
            sigma_str = f"{float(sigma):.4g}" if sigma is not None else "?"
            tau_str = f"{float(tau):.4g}" if tau is not None else "?"
            lines.append(
                f"- `{name}` ({distribution} base prior, mu={mu_str}, sigma={sigma_str}; "
                f"member_deviation_sigma={tau_str})\n  - {description}"
            )
        prompt = prompt.replace("{{AUXILIARY_GROUPS}}", "\n".join(lines))
    else:
        prompt = prompt.replace(
            "{{AUXILIARY_GROUPS}}",
            (
                "No auxiliary parameter groups are declared for this run. "
                "Do NOT use `observable.auxiliary_parameters` — every cal target "
                "must satisfy the strict matching requirements without bridging."
            ),
        )

    return prompt


def build_submodel_target_prompt(
    parameters: str,
    model_context: str,
    parameter_context: str = "",
    notes: str = "",
    used_primary_studies: str = "",
    reference_db_entries: list[dict] | None = None,
) -> str:
    """
    Build submodel target extraction prompt.

    Args:
        parameters: Comma-separated parameter names to calibrate (e.g., "k_CD8_pro,k_CD8_death")
        model_context: High-level model description (from model_context.txt)
        parameter_context: Rich context for each parameter (reactions, species, etc.)
        notes: Optional notes/guidance for the extraction
        used_primary_studies: Formatted list of already-used primary studies (optional)
        reference_db_entries: List of reference value dicts with name, display_name, value, units, notes

    Returns:
        Complete prompt with placeholders replaced
    """
    prompt = read_prompt("submodel_target_prompt.md")

    prompt = prompt.replace("{{PARAMETERS}}", parameters)
    prompt = prompt.replace("{{MODEL_CONTEXT}}", model_context)
    prompt = prompt.replace(
        "{{PARAMETER_CONTEXT}}", parameter_context or "No parameter context available."
    )
    prompt = prompt.replace("{{USED_PRIMARY_STUDIES}}", used_primary_studies)

    # Inject reference database listing
    if reference_db_entries:
        lines = []
        for entry in reference_db_entries:
            name = entry["name"]
            display = entry.get("display_name", name)
            value = float(entry.get("value", 0))
            units = entry.get("units", "")
            value_str = f"{value:.4g}"
            lines.append(f"- `{name}`: {display} ({value_str} {units})")
        prompt = prompt.replace("{{REFERENCE_DATABASE}}", "\n".join(lines))
    else:
        prompt = prompt.replace("{{REFERENCE_DATABASE}}", "No reference database available.")

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
