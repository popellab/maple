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
    parameter_names: list[str] | None = None,
) -> str:
    """
    Collect information about existing studies for a given submodel target.

    For multi-parameter targets, also collects studies from single-parameter extractions
    for each individual parameter to prevent reuse.

    Args:
        target_id: Target ID (e.g., "psc_activation")
        cancer_type: Cancer type for the target (e.g., "PDAC")
        previous_extractions_dir: Path to directory containing previous extraction YAML files
        parameter_names: Optional list of parameter names for multi-parameter targets.
                        If provided, also searches for {param}_{cancer_type}_deriv*.yaml files.

    Returns:
        Formatted string describing existing studies, or empty string if none exist
    """
    import yaml

    if not previous_extractions_dir or not previous_extractions_dir.exists():
        return ""

    # Find all YAML files matching {target_id}_{cancer_type}_deriv*.yaml pattern
    yaml_files = list(previous_extractions_dir.glob(f"{target_id}_{cancer_type}_deriv*.yaml"))

    # Also search for individual parameter files if parameter_names provided
    if parameter_names:
        for param_name in parameter_names:
            param_files = list(
                previous_extractions_dir.glob(f"{param_name}_{cancer_type}_deriv*.yaml")
            )
            yaml_files.extend(param_files)

    # Deduplicate file list (in case of overlapping patterns)
    yaml_files = list(set(yaml_files))

    if not yaml_files:
        return ""

    # Collect primary sources from all matching files
    # Use dict keyed by DOI/URL to deduplicate sources used in multiple files
    sources_by_id: dict[str, tuple[str, dict]] = {}

    for yaml_file in sorted(yaml_files):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                study_data = yaml.safe_load(f)

            if not study_data:
                continue

            # Extract primary_data_source only (not secondary sources)
            if "primary_data_source" in study_data and study_data["primary_data_source"]:
                source = study_data["primary_data_source"]
                doi = source.get("doi", "")
                url = source.get("url", "")
                source_id = doi or url or yaml_file.name
                # Only add if not already seen (dedup by DOI/URL)
                if source_id not in sources_by_id:
                    sources_by_id[source_id] = (yaml_file.name, source)

        except Exception as e:
            print(f"Warning: Could not process {yaml_file}: {e}")
            continue

    if not sources_by_id:
        return ""

    # Format the complete section
    header = "\n## Sources Already Used for This Target\n\n"
    header += "**IMPORTANT:** The following primary sources have already been used in previous extractions "
    header += "for this target or its constituent parameters. "
    header += "DO NOT re-use these sources. Instead, find NEW sources not listed below.\n\n"

    # List only doi/url and title
    output = []
    for source_id, (filename, source_data) in sorted(sources_by_id.items()):
        doi = source_data.get("doi", "")
        url = source_data.get("url", "")
        title = source_data.get("title", "").strip()

        identifier = doi or url or "unknown"
        if title:
            output.append(f"- **{identifier}**: {title}")
        else:
            output.append(f"- {identifier}")

    return header + "\n".join(output)
