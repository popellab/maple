#!/usr/bin/env python3
"""
Parameter processing utilities for QSP workflows.

Functions for rendering parameter information blocks and collecting existing studies.
"""

from pathlib import Path


def render_parameter_to_search(
    name: str, units: str, definition: str, cancer_type: str = None
) -> str:
    """
    Render a parameter information block for prompt generation.

    Args:
        name: Parameter name
        units: Parameter units
        definition: Parameter definition/description
        cancer_type: Optional cancer type for this parameter

    Returns:
        Formatted parameter information block
    """
    base_info = f"**Parameter Name:** {name}\n**Units:** {units}\n**Definition:** {definition}"
    if cancer_type:
        base_info += f"\n**Target Cancer Type:** {cancer_type}"
    return base_info


def collect_existing_studies(
    cancer_type: str,
    parameter_name: str,
    parameter_storage_dir: Path,
) -> str:
    """
    Collect information about existing studies for a given parameter.

    Args:
        cancer_type: Cancer type for the parameter
        parameter_name: Name of the parameter
        parameter_storage_dir: Path to parameter storage directory (e.g., metadata-storage/parameter_estimates)

    Returns:
        Formatted string describing existing studies, or empty string if none exist
    """
    import yaml

    if not parameter_storage_dir.exists():
        return ""

    # Find all YAML files matching {parameter_name}_*.yaml pattern (flat structure)
    yaml_files = list(parameter_storage_dir.glob(f"{parameter_name}_*.yaml"))

    if not yaml_files:
        return ""

    # Collect source fields verbatim from all files matching cancer_type
    all_sources = []

    for yaml_file in sorted(yaml_files):
        try:
            # Parse filename to check cancer type
            # Format: {param_name}_{cancer_type}_deriv{num}.yaml
            filename_parts = yaml_file.stem.split("_")

            # Check if cancer_type appears in filename
            if cancer_type not in filename_parts:
                continue

            with open(yaml_file, "r", encoding="utf-8") as f:
                study_data = yaml.safe_load(f)

            if not study_data:
                continue

            # Extract raw source fields (handle multiple schema variants)
            if "data_sources" in study_data and study_data["data_sources"]:
                all_sources.append(("data_sources", study_data["data_sources"]))

            # Check v1 schema: sources (fallback)
            if "sources" in study_data and study_data["sources"]:
                all_sources.append(("sources", study_data["sources"]))

        except Exception as e:
            print(f"Warning: Could not process {yaml_file}: {e}")
            continue

    if not all_sources:
        return ""

    # Format the complete section with verbatim YAML
    header = "\n## Sources Already Used for This Parameter\n\n"
    header += "**IMPORTANT:** The following sources have already been used in previous extractions for this parameter. "
    header += "DO NOT re-use these sources. Instead, find NEW sources not listed below.\n\n"

    # Dump sources as YAML
    output = []
    for source_type, source_data in all_sources:
        output.append(f"### {source_type}")
        output.append("```yaml")
        import yaml as yaml_lib

        output.append(
            yaml_lib.dump(
                source_data, default_flow_style=False, sort_keys=False, allow_unicode=True
            ).strip()
        )
        output.append("```")
        output.append("")

    return header + "\n".join(output)


def collect_existing_studies_for_submodel_target(
    target_id: str,
    cancer_type: str,
    previous_extractions_dir: Path,
) -> str:
    """
    Collect information about existing studies for a given submodel target.

    Args:
        target_id: Target ID (e.g., "psc_activation")
        cancer_type: Cancer type for the target (e.g., "PDAC")
        previous_extractions_dir: Path to directory containing previous extraction YAML files

    Returns:
        Formatted string describing existing studies, or empty string if none exist
    """
    import yaml

    if not previous_extractions_dir or not previous_extractions_dir.exists():
        return ""

    # Find all YAML files matching {target_id}_{cancer_type}_deriv*.yaml pattern
    yaml_files = list(previous_extractions_dir.glob(f"{target_id}_{cancer_type}_deriv*.yaml"))

    if not yaml_files:
        return ""

    # Collect source fields from all matching files
    all_sources = []

    for yaml_file in sorted(yaml_files):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                study_data = yaml.safe_load(f)

            if not study_data:
                continue

            # Extract primary_data_source (SubmodelTarget schema)
            if "primary_data_source" in study_data and study_data["primary_data_source"]:
                source = study_data["primary_data_source"]
                # Include filename for reference
                all_sources.append((yaml_file.name, "primary_data_source", source))

            # Extract secondary_data_sources if present
            if "secondary_data_sources" in study_data and study_data["secondary_data_sources"]:
                for secondary in study_data["secondary_data_sources"]:
                    all_sources.append((yaml_file.name, "secondary_data_source", secondary))

        except Exception as e:
            print(f"Warning: Could not process {yaml_file}: {e}")
            continue

    if not all_sources:
        return ""

    # Format the complete section with verbatim YAML
    header = "\n## Sources Already Used for This Target\n\n"
    header += "**IMPORTANT:** The following sources have already been used in previous extractions for this target. "
    header += "DO NOT re-use these primary sources. Instead, find NEW sources not listed below.\n\n"

    # Dump sources as YAML, grouped by file
    output = []
    current_file = None
    import yaml as yaml_lib

    for filename, source_type, source_data in all_sources:
        if filename != current_file:
            if current_file is not None:
                output.append("")
            output.append(f"### From `{filename}`")
            current_file = filename

        output.append(f"**{source_type}:**")
        output.append("```yaml")
        output.append(
            yaml_lib.dump(
                source_data, default_flow_style=False, sort_keys=False, allow_unicode=True
            ).strip()
        )
        output.append("```")

    return header + "\n".join(output)
