#!/usr/bin/env python3
"""
Parameter processing utilities for QSP workflows.

Functions for loading parameter data, building model context, and rendering
parameter information blocks for prompt generation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


# Target biological context block used in model context generation
TARGET_BIO_CONTEXT = (
    "## Target Biological Context\n\n"
    "- **Priority:** Human > (if absent) closest human in vitro/ex vivo > (if still absent) relevant animal in vivo.\n"
    "- Use therapy-induced/withdrawal/stress datasets only as bounds; do not center priors on those unless explicitly requested.\n"
)


def load_inputs(params_path: Path, reactions_path: Path, template_path: Path = None):
    """
    Load parameter and reaction data from CSV files.
    
    Args:
        params_path: Path to simbio_parameters.csv with Name, Units, Definition columns
        reactions_path: Path to model_context.csv with reaction information
        template_path: Optional path to template file (legacy parameter)
        
    Returns:
        Tuple of (params_df, reactions_df, template_text)
    """
    params_df = pd.read_csv(params_path)
    reactions_df = pd.read_csv(reactions_path)
    template_text = template_path.read_text() if template_path else ""
    
    # Validate required columns
    need_params = {"Name", "Units", "Definition"}
    need_rxns = {"Parameter", "Reaction", "ReactionRate", "Rule", "RuleType", "OtherParameters", "OtherSpeciesWithNotes"}
    miss_p = need_params - set(params_df.columns)
    miss_r = need_rxns - set(reactions_df.columns)

    if miss_p:
        raise ValueError(f"Parameters CSV missing columns: {sorted(miss_p)}")
    if miss_r:
        raise ValueError(f"Reactions CSV missing columns: {sorted(miss_r)}")

    return params_df, reactions_df, template_text


def index_param_info(params_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Create a parameter lookup index from the parameters DataFrame.
    
    Args:
        params_df: DataFrame with Name, Units, Definition, References columns
        
    Returns:
        Dictionary mapping parameter names to their metadata
    """
    return (
        params_df.assign(Name=params_df["Name"].astype(str))
                 .set_index("Name")[["Units", "Definition", "References"]]
                 .to_dict(orient="index")
    )


def parse_other_params_list(s: str) -> List[str]:
    """
    Parse parameter names from a string representation of a list.
    
    Args:
        s: String like "['k1','k2']" containing parameter names
        
    Returns:
        List of parameter names
    """
    if not isinstance(s, str) or not s.strip():
        return []
    # Extract names from a string like "['k1','k2']"
    return re.findall(r"'([^']+)'", s)


def render_parameter_to_search(name: str, units: str, definition: str, cancer_type: str = None) -> str:
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


def render_other_params_details(other_params: List[str], param_info: Dict[str, Dict[str, str]]) -> str:
    """
    Render details for other parameters referenced in reactions.
    
    Args:
        other_params: List of parameter names
        param_info: Parameter lookup dictionary from index_param_info()
        
    Returns:
        Formatted parameter details block
    """
    if not other_params:
        return "  - **Other parameters details:** —"
        
    lines = []
    for nm in sorted(set(other_params)):
        info = param_info.get(nm, {}) or {}
        units = (info.get("Units") or "").strip()
        definition = (info.get("Definition") or "").strip()
        
        if units and definition:
            lines.append(f"    - **{nm}** [{units}] — {definition}")
        elif definition:
            lines.append(f"    - **{nm}** — {definition}")
        elif units:
            lines.append(f"    - **{nm}** [{units}]")
        else:
            lines.append(f"    - **{nm}**")
            
    return "  - **Other parameters details:**\n" + "\n".join(lines)


def render_species_comp_details(species_json_str: str) -> str:
    """
    Render species/compartment details from JSON string.
    
    Args:
        species_json_str: JSON string containing species information
        
    Returns:
        Formatted species details block
    """
    items: List[Dict[str, Any]] = []
    if isinstance(species_json_str, str) and species_json_str.strip():
        try:
            parsed = json.loads(species_json_str)
            if isinstance(parsed, list):
                items = [x for x in parsed if isinstance(x, dict)]
        except Exception:
            items = []
            
    if not items:
        return "  - **Other species/compartment details:** —"

    # Deduplicate by (name, notes) while preserving order
    seen = set()
    lines = []
    for it in items:
        name = str(it.get("name", "")).strip()
        notes = str(it.get("notes", "")).strip()
        if not name:
            continue
        key = (name, notes)
        if key in seen:
            continue
        seen.add(key)
        if notes:
            lines.append(f"    - **{name}** — {notes}")
        else:
            lines.append(f"    - **{name}**")
            
    return "  - **Other species/compartment details:**\n" + "\n".join(lines) if lines else \
           "  - **Other species/compartment details:** —"


def build_model_context(param_name: str, rxns: pd.DataFrame, param_info: Dict[str, Dict[str, str]]) -> str:
    """
    Build model context information for a parameter.
    
    Args:
        param_name: Name of the parameter to build context for
        rxns: DataFrame with reaction and rule information for this parameter
        param_info: Parameter lookup dictionary from index_param_info()
        
    Returns:
        Formatted model context block
    """
    if rxns.empty:
        return (f"{param_name} is currently not referenced in any reactions or rules "
                f"according to the provided mapping table.")

    # Separate reactions and rules
    reactions = rxns[rxns['Reaction'].notna() & (rxns['Reaction'].astype(str).str.strip() != '')]
    rules = rxns[rxns['Rule'].notna() & (rxns['Rule'].astype(str).str.strip() != '')]

    bullets = []
    
    # Process reactions
    for _, row in reactions.iterrows():
        other_params_str = row.get("OtherParameters", "")
        other_params = parse_other_params_list(other_params_str)
        other_params_details = render_other_params_details(other_params, param_info)
        species_details = render_species_comp_details(row.get("OtherSpeciesWithNotes", ""))

        bullets.append(
            "- **Reaction:** `{}`\n"
            "  - **Rate:** `{}`\n"
            "  - **Other parameters in rate:** {}\n"
            "{}\n"
            "{}".format(
                row.get("Reaction", ""),
                row.get("ReactionRate", ""),
                other_params_str if (isinstance(other_params_str, str) and other_params_str.strip()) else "[]",
                other_params_details,
                species_details
            )
        )

    # Process rules
    for _, row in rules.iterrows():
        other_params_str = row.get("OtherParameters", "")
        other_params = parse_other_params_list(other_params_str)
        other_params_details = render_other_params_details(other_params, param_info)
        species_details = render_species_comp_details(row.get("OtherSpeciesWithNotes", ""))

        bullets.append(
            "- **Rule:** `{}`\n"
            "  - **Type:** {}\n"
            "  - **Other parameters in rule:** {}\n"
            "{}\n"
            "{}".format(
                row.get("Rule", ""),
                row.get("RuleType", ""),
                other_params_str if (isinstance(other_params_str, str) and other_params_str.strip()) else "[]",
                other_params_details,
                species_details
            )
        )

    if not bullets:
        return (f"{param_name} is currently not referenced in any reactions or rules "
                f"according to the provided mapping table.")

    # Return just the mathematical context without biological priority guidelines
    header = "Mathematical role and biological context for this parameter based on the model:\n"
    return header + "\n".join(bullets)


def collect_existing_studies(cancer_type: str, parameter_name: str,
                           parameter_storage_dir: Path = None) -> str:
    """
    Collect information about existing studies for a given parameter.

    Args:
        cancer_type: Cancer type for the parameter
        parameter_name: Name of the parameter
        parameter_storage_dir: Path to parameter storage directory (defaults to ../qsp-metadata-storage/parameter_estimates)

    Returns:
        Formatted string describing existing studies, or empty string if none exist
    """
    import yaml

    if parameter_storage_dir is None:
        # Default to sibling directory
        parameter_storage_dir = Path(__file__).parent.parent.parent / "qsp-metadata-storage" / "parameter_estimates"

    if not parameter_storage_dir.exists():
        return ""

    # Find all YAML files matching {parameter_name}_*.yaml pattern (flat structure)
    yaml_files = list(parameter_storage_dir.glob(f"{parameter_name}_*.yaml"))

    if not yaml_files:
        return ""

    # Collect source fields verbatim from all files
    all_sources = []

    for yaml_file in sorted(yaml_files):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                study_data = yaml.safe_load(f)

            if not study_data:
                continue

            # Extract raw source fields (handle multiple schema variants)
            if 'data_sources' in study_data and study_data['data_sources']:
                all_sources.append(('data_sources', study_data['data_sources']))

            if 'methodological_sources' in study_data and study_data['methodological_sources']:
                all_sources.append(('methodological_sources', study_data['methodological_sources']))

            # Check for quick estimate 'source' field (singular)
            if 'source' in study_data and study_data['source']:
                all_sources.append(('source', study_data['source']))

            # Check v1 schema: sources (fallback)
            if 'sources' in study_data and study_data['sources']:
                all_sources.append(('sources', study_data['sources']))

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
        output.append(yaml_lib.dump(source_data, default_flow_style=False, sort_keys=False, allow_unicode=True).strip())
        output.append("```")
        output.append("")

    return header + "\n".join(output)