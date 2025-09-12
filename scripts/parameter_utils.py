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
    need_rxns = {"Parameter", "Reaction", "ReactionRate", "OtherParameters", "OtherSpeciesWithNotes"}
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


def render_parameter_to_search(name: str, units: str, definition: str) -> str:
    """
    Render a parameter information block for prompt generation.
    
    Args:
        name: Parameter name
        units: Parameter units
        definition: Parameter definition/description
        
    Returns:
        Formatted parameter information block
    """
    explanation = (
        "**Field meaning:**\n"
        "- **Name**: The model parameter identifier to focus on.\n"
        "- **Units**: The unit system used for this parameter.\n"
        "- **Definition**: Short description of what this parameter represents.\n"
    )
    descriptor = f"**Parameter:** {name} [{units}] — {definition}".strip(" —")
    return explanation + "\n" + descriptor


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
        rxns: DataFrame with reaction information for this parameter
        param_info: Parameter lookup dictionary from index_param_info()
        
    Returns:
        Formatted model context block
    """
    if rxns.empty:
        body = (f"{param_name} is currently not referenced in any reactions "
                f"according to the provided mapping table.")
        return body + "\n\n" + TARGET_BIO_CONTEXT

    bullets = []
    for _, row in rxns.iterrows():
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

    # Add biological context block beneath the reaction info
    header = "Mathematical role and biological context for this parameter based on the model:\n"
    return header + "\n".join(bullets) + "\n\n" + TARGET_BIO_CONTEXT