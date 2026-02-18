#!/usr/bin/env python3
"""
Base class for creating prompts for different types of LLM processing tasks.

This module provides a common framework for generating OpenAI prompt assembly requests
with consistent patterns for request creation, file output, and error handling.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional


from qsp_llm_workflows.core.prompts import (
    build_parameter_extraction_prompt,
    build_test_statistic_prompt,
    build_calibration_target_prompt,
    build_isolated_system_target_prompt,
    build_submodel_target_prompt,
)
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic
from qsp_llm_workflows.core.calibration import CalibrationTarget, IsolatedSystemTarget
from qsp_llm_workflows.core.calibration.submodel_target import SubmodelTarget
from qsp_llm_workflows.core.model_structure import ModelStructure


class PromptBuilder(ABC):
    """
    Abstract base class for creating OpenAI prompt assembly requests.

    Provides common functionality for request creation, prompt assembly
    while allowing subclasses to implement specific logic for different prompt types.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize the prompt builder.

        Args:
            base_dir: Base directory of the project (used for relative path resolution)
        """
        self.base_dir = Path(base_dir)

    @abstractmethod
    def get_workflow_type(self) -> str:
        """
        Get the type identifier for this prompt builder.

        Used for default file naming and logging.

        Returns:
            String identifier for this workflow type (e.g., "parameter", "pooling_metadata")
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Process input data and generate prompts.

        This method should be implemented by subclasses to handle their specific
        input processing and prompt generation logic.

        Returns:
            List of prompt dictionaries with keys: custom_id, prompt, pydantic_model
        """
        pass


class ParameterPromptBuilder(PromptBuilder):
    """
    Creates prompts for parameter extraction from scientific literature.

    Processes CSV input with embedded parameter definitions and model context, generating
    prompts for comprehensive literature extraction. Requires columns: cancer_type,
    parameter_name, parameter_units, parameter_description, model_context.
    """

    def get_workflow_type(self) -> str:
        return "parameter"

    def format_model_context(self, model_context_json: str) -> str:
        """
        Parse and format the model context JSON into readable text for the LLM.

        Args:
            model_context_json: JSON string containing reactions_and_rules with model context

        Returns:
            Formatted markdown text describing the model context
        """

        try:
            context_data = json.loads(model_context_json)
        except json.JSONDecodeError as e:
            return f"Error parsing model context: {e}"

        output = []

        # Add derived from context if available
        if "derived_from_context" in context_data:
            derived = context_data["derived_from_context"]
            if derived:
                output.append("## Parameter Context")
                for item in derived:
                    output.append(
                        f"- **{item.get('name', 'Unknown')}**: {item.get('description', 'No description')}"
                    )
                output.append("")

        # Add reactions and rules
        if "reactions_and_rules" in context_data:
            reactions = context_data["reactions_and_rules"]
            if reactions:
                output.append("## Model Usage")
                output.append(
                    f"This parameter appears in {len(reactions)} reaction(s) and/or rule(s):"
                )
                output.append("")

                for i, rxn in enumerate(reactions, 1):
                    # Reaction or rule
                    if rxn.get("reaction"):
                        output.append(f"### {i}. Reaction: `{rxn['reaction']}`")
                        if rxn.get("reaction_rate"):
                            output.append(f"**Rate:** `{rxn['reaction_rate']}`")
                    elif rxn.get("rule"):
                        output.append(f"### {i}. Rule ({rxn.get('rule_type', 'unknown type')})")
                        output.append(f"**Expression:** `{rxn['rule']}`")

                    output.append("")

                    # Other parameters
                    other_params = rxn.get("other_parameters", [])
                    if other_params:
                        output.append("**Related Parameters:**")
                        for param in other_params:
                            name = param.get("name", "Unknown")
                            desc = param.get("description", "")
                            if desc:
                                output.append(f"- `{name}`: {desc}")
                            else:
                                output.append(f"- `{name}`")
                        output.append("")

                    # Other species
                    other_species = rxn.get("other_species", [])
                    if other_species:
                        output.append("**Related Species:**")
                        for species in other_species:
                            name = species.get("name", "Unknown")
                            desc = species.get("description", "")
                            if desc:
                                output.append(f"- `{name}`: {desc}")
                            else:
                                output.append(f"- `{name}`")
                        output.append("")

        return "\n".join(output) if output else "No model context available."

    def process(
        self,
        input_csv: Path,
        parameter_storage_dir: Path,
        reasoning_effort: str = "high",
    ) -> List[Dict[str, Any]]:
        """
        Process parameter extraction inputs and generate prompts.

        Args:
            input_csv: CSV file with columns: cancer_type, parameter_name, parameter_units,
                      parameter_description, model_context (JSON)
            parameter_storage_dir: Path to parameter storage directory for checking existing studies
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            List of prompt request dictionaries
        """
        import csv
        from .parameter_utils import render_parameter_to_search, collect_existing_studies

        # Process CSV and create requests
        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                cancer_type = row["cancer_type"]
                parameter_name = row["parameter_name"]
                units = row.get("parameter_units", "")
                definition = row.get("parameter_description", "")
                model_context_json = row.get("model_context", "{}")

                # Format the model context from JSON
                model_context_block = self.format_model_context(model_context_json)

                # Build parameter info block with cancer type
                parameter_block = render_parameter_to_search(
                    parameter_name, units, definition, cancer_type
                )

                # Collect existing studies to avoid re-extracting from same sources
                existing_studies = collect_existing_studies(
                    cancer_type, parameter_name, parameter_storage_dir
                )

                # Build the prompt using simple prompt builder
                prompt = build_parameter_extraction_prompt(
                    parameter_info=parameter_block,
                    model_context=model_context_block,
                    cancer_type=cancer_type,
                    used_primary_studies=existing_studies,
                )

                # Create prompt dict
                custom_id = f"{cancer_type}_{parameter_name}_{i}"
                # Create simple prompt dict

                request = {
                    "custom_id": custom_id,
                    "prompt": prompt,
                    "pydantic_model": ParameterMetadata,
                }

                requests.append(request)

        return requests


class TestStatisticPromptBuilder(PromptBuilder):
    """
    Creates prompts for test statistic generation from biological expectations.

    Processes CSV input with test statistic descriptions and biological expectations, generating
    prompts that create comprehensive test statistic definitions for QSP model validation.
    """

    def get_workflow_type(self) -> str:
        return "test_stat"

    def _get_default_species_units(self) -> Dict[str, str]:
        """
        Get default species units mapping for common species types.

        Returns:
            Dictionary mapping species names to their default units
        """
        return {
            "CD8": "cells",
            "CD4": "cells",
            "Treg": "cells",
            "Th": "cells",
            "APC": "cells",
            "mAPC": "cells",
            "DC": "cells",
            "Mac_M1": "cells",
            "Mac_M2": "cells",
            "MDSC": "cells",
            "C": "cells",
            "C_x": "cells",
            "C1": "cells",
            "GVAX_cells": "cells",
            "TumorVolume": "cm³",
            "K": "dimensionless",
            "IL2": "pg/mL",
            "IL10": "pg/mL",
            "IL12": "pg/mL",
            "IFNg": "pg/mL",
            "TNFa": "pg/mL",
            "TGFb": "pg/mL",
            "GMCSF": "pg/mL",
            "CCL2": "pg/mL",
            "NO": "μM",
            "ArgI": "units/mg protein",
            "ECM": "dimensionless",
            "CAF": "cells",
            "Fib": "cells",
            "c_vas": "pg/mL",
            "aPD1": "nanomolarity",
            "aPDL1": "nanomolarity",
            "aCTLA4": "nanomolarity",
            "T_eff": "cells",
            "CD8_exh": "cells",
            "Th_exh": "cells",
        }

    def _load_species_units_mapping(self) -> Dict[str, str]:
        """
        Get species units mapping for common species types.

        Since model_context.txt now contains all species with units,
        this method returns default unit mappings as a helper for formatting.

        Returns:
            Dictionary mapping species names/patterns to units
        """
        return self._get_default_species_units()

    def _parse_species_with_units(
        self, required_species: str, species_units_mapping: Dict[str, str]
    ) -> str:
        """
        Parse required_species string and format with units information.

        Args:
            required_species: Comma-separated string of species (e.g., "V_T.CD8,V_T.Treg")
            species_units_mapping: Dictionary mapping species names to units

        Returns:
            Formatted string with species and their units
        """
        if not required_species.strip():
            return ""

        species_list = [s.strip() for s in required_species.split(",") if s.strip()]
        formatted_species = []

        for species in species_list:
            # Remove any square brackets that might be in the input
            species = species.strip("[]").strip()
            # Extract the species name from compartment notation (e.g., V_T.CD8 -> CD8)
            if "." in species:
                compartment, species_name = species.split(".", 1)
            else:
                compartment, species_name = "", species

            # Look up units
            units = "dimensionless"  # default

            # Try exact match first
            if species_name in species_units_mapping:
                units = species_units_mapping[species_name]
            else:
                # Try partial matches for compound names
                for pattern, mapped_units in species_units_mapping.items():
                    if pattern.lower() in species_name.lower():
                        units = mapped_units
                        break

            # Format the output
            if compartment:
                formatted_species.append(
                    f"- `{species}`: {species_name} in {compartment} compartment (units: {units})"
                )
            else:
                formatted_species.append(f"- `{species}`: {species_name} (units: {units})")

        return "\n".join(formatted_species)

    def process(
        self,
        input_csv: Path,
        model_context_csv: Path = None,
        reasoning_effort: str = "high",
    ) -> List[Dict[str, Any]]:
        """
        Process test statistic inputs and generate prompts.

        Args:
            input_csv: CSV file with test_statistic_id and biological expectation columns
            model_context_csv: Optional CSV file with model structure information
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            List of prompt request dictionaries
        """
        import csv
        import pandas as pd

        # Load species units from simbio_parameters.csv
        species_units_mapping = self._load_species_units_mapping()

        # Load model context if provided
        model_context_info = {}
        if model_context_csv and Path(model_context_csv).exists():
            model_df = pd.read_csv(model_context_csv)
            # Create a lookup for model variables and their descriptions
            for _, row in model_df.iterrows():
                var_name = str(row.get("Variable", "")).strip()
                if var_name:
                    model_context_info[var_name] = {
                        "description": str(row.get("Description", "")).strip(),
                        "units": str(row.get("Units", "")).strip(),
                        "compartment": str(row.get("Compartment", "")).strip(),
                    }

        # Process CSV and create requests
        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                test_statistic_id = row.get("test_statistic_id", f"test_stat_{i}")
                model_context = row.get("model_context", "")
                scenario_context = row.get("scenario_context", "")
                required_species = row.get("required_species", "")
                derived_species_description = row.get("derived_species_description", "")

                if not model_context.strip():
                    print(f"Warning: Empty model context for {test_statistic_id}, skipping")
                    continue

                if not scenario_context.strip():
                    print(f"Warning: Empty scenario context for {test_statistic_id}, skipping")
                    continue

                if not required_species.strip():
                    print(f"Warning: Empty required species for {test_statistic_id}, skipping")
                    continue

                if not derived_species_description.strip():
                    print(
                        f"Warning: Empty derived species description for {test_statistic_id}, skipping"
                    )
                    continue

                # Use provided context directly, with optional model context CSV enhancement
                model_context_block = model_context
                if model_context_info:
                    model_context_block += "\n\n**Additional Model Variables:**\n\n"
                    for var_name, info in model_context_info.items():
                        model_context_block += f"- `{var_name}`: {info['description']}"
                        if info["units"]:
                            model_context_block += f" (units: {info['units']})"
                        if info["compartment"]:
                            model_context_block += f" [compartment: {info['compartment']}]"
                        model_context_block += "\n"

                # Use provided scenario context directly
                scenario_context_block = scenario_context

                # Parse required species with units information
                required_species_with_units = self._parse_species_with_units(
                    required_species, species_units_mapping
                )

                # Get cancer type from CSV row
                cancer_type = row.get("cancer_type", "unknown")

                # Build the prompt using simple prompt builder
                prompt = build_test_statistic_prompt(
                    model_context=model_context_block,
                    scenario_context=scenario_context_block,
                    required_species_with_units=required_species_with_units,
                    derived_species_description=derived_species_description,
                    cancer_type=cancer_type,
                    used_primary_studies="",  # No used studies tracking for test statistics yet
                )

                # Create prompt dict
                custom_id = f"test_stat_{test_statistic_id}_{i}"
                # Create simple prompt dict

                request = {
                    "custom_id": custom_id,
                    "prompt": prompt,
                    "pydantic_model": TestStatistic,
                }

                requests.append(request)

        return requests


class CalibrationTargetPromptBuilder(PromptBuilder):
    """
    Creates prompts for calibration target extraction from scientific literature.

    Processes CSV input with observable descriptions and model context, generating
    prompts to extract raw observables with experimental context for Bayesian inference.
    """

    def get_workflow_type(self) -> str:
        return "calibration_target"

    def process(
        self,
        input_csv: Path,
        species_units_file: Optional[Path] = None,
        reasoning_effort: str = "high",
        reference_values_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process calibration target inputs and generate prompts.

        Args:
            input_csv: CSV file with columns: calibration_target_id, cancer_type,
                      observable_description, model_species, model_indication,
                      model_compartment, model_system, model_treatment_history,
                      model_stage_burden, relevant_compartments, used_primary_studies (optional),
                      primary_source_title (optional)
            species_units_file: Optional JSON file mapping species -> units
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            reference_values_file: Optional YAML file with curated reference values

        Returns:
            List of prompt request dictionaries
        """
        import csv

        # Load all species units if provided
        all_species_units = {}
        if species_units_file and species_units_file.exists():
            with open(species_units_file, "r") as f:
                all_species_units = json.load(f)

        # Load reference values database
        reference_db_entries = None
        reference_db = None
        if reference_values_file and reference_values_file.exists():
            import yaml as _yaml

            with open(reference_values_file) as _f:
                _ref_data = _yaml.safe_load(_f)
            reference_db_entries = _ref_data.get("values", [])
            reference_db = {v["name"]: float(v["value"]) for v in reference_db_entries}

        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                calibration_target_id = row.get("calibration_target_id", f"target_{i}")
                cancer_type = row.get("cancer_type", "UNKNOWN")
                observable_description = row.get("observable_description", "")
                model_species = row.get("model_species", "")
                model_indication = row.get("model_indication", "")
                model_compartment = row.get("model_compartment", "")
                model_system = row.get("model_system", "")
                model_treatment_history = row.get("model_treatment_history", "")
                model_stage_burden = row.get("model_stage_burden", "")
                relevant_compartments = row.get("relevant_compartments", "")
                used_primary_studies = row.get("used_primary_studies", "")
                primary_source_title = row.get("primary_source_title", "")

                if not observable_description.strip():
                    print(
                        f"Warning: Empty observable description for {calibration_target_id}, skipping"
                    )
                    continue

                # Filter species by relevant compartments
                filtered_species_text = "Not provided"
                if relevant_compartments and all_species_units:
                    # Parse compartment list (comma-separated)
                    compartments = [c.strip() for c in relevant_compartments.split(",")]
                    # Filter species that start with any of the compartment prefixes
                    filtered_species = {
                        species: units
                        for species, units in all_species_units.items()
                        if any(species.startswith(f"{comp}.") for comp in compartments)
                    }
                    # Format as readable text
                    if filtered_species:
                        filtered_species_text = "\n".join(
                            f"- {species}: {units}"
                            for species, units in sorted(filtered_species.items())
                        )

                # Build the prompt
                prompt = build_calibration_target_prompt(
                    observable_description=observable_description,
                    cancer_type=cancer_type,
                    model_species=model_species or "Not specified",
                    model_indication=model_indication or "Not specified",
                    model_compartment=model_compartment or "Not specified",
                    model_system=model_system or "Not specified",
                    model_treatment_history=model_treatment_history or "Not specified",
                    model_stage_burden=model_stage_burden or "Not specified",
                    model_species_with_units=filtered_species_text,
                    used_primary_studies=used_primary_studies
                    or "None - this is the first extraction",
                    primary_source_title=primary_source_title,
                    reference_db_entries=reference_db_entries,
                )

                # Create prompt dict
                custom_id = f"cal_target_{calibration_target_id}_{i}"

                validation_context = {"species_units": all_species_units}
                if reference_db:
                    validation_context["reference_db"] = reference_db

                request = {
                    "custom_id": custom_id,
                    "prompt": prompt,
                    "pydantic_model": CalibrationTarget,
                    "validation_context": validation_context,
                }

                requests.append(request)

        return requests


class IsolatedSystemTargetPromptBuilder(PromptBuilder):
    """
    Creates prompts for isolated system target extraction from in vitro/preclinical literature.

    Processes CSV input with parameter names and generates prompts for
    finding experimental data to calibrate those parameters.

    Requires model_structure_file and model_context_file to be provided.
    """

    def get_workflow_type(self) -> str:
        return "isolated_system_target"

    def format_parameter_context(self, parameters: str, model_structure: ModelStructure) -> str:
        """
        Build rich context block for parameters from ModelStructure.

        Includes the broader reaction network - other reactions that involve
        the same species as the target parameter's reactions.

        Args:
            parameters: Comma-separated parameter names (e.g., "k_CD8_pro,k_CD8_death")
            model_structure: ModelStructure instance with parameters, reactions, species

        Returns:
            Formatted markdown text describing each parameter's role in the model
        """
        param_names = [p.strip() for p in parameters.split(",") if p.strip()]
        output = []

        for param_name in param_names:
            # Look up parameter in model structure
            param = model_structure.get_parameter(param_name)

            # Parameter header
            output.append(f"### Parameter: `{param_name}`")

            # Parameter info
            if param:
                if param.units:
                    output.append(f"- **Units:** {param.units}")
                if param.description:
                    output.append(f"- **Description:** {param.description}")
            else:
                output.append("- **Warning:** Parameter not found in model structure")

            output.append("")

            # Collect species from direct reactions for broader network lookup
            direct_species: set = set()

            # Get reactions using this parameter
            reactions = model_structure.get_reactions_for_parameter(param_name)
            if reactions:
                output.append(f"**Direct reactions ({len(reactions)}):**")
                output.append("")

                for i, rxn in enumerate(reactions, 1):
                    # Reaction header with stoichiometry
                    reactant_str = " + ".join(rxn.reactants) if rxn.reactants else "null"
                    product_str = " + ".join(rxn.products) if rxn.products else "null"
                    output.append(f"**{i}. Reaction:** `{reactant_str} -> {product_str}`")
                    if rxn.rate_law:
                        output.append(f"- **Rate:** `{rxn.rate_law}`")

                    # Related species (from reactants and products)
                    all_species = rxn.reactants + rxn.products
                    if all_species:
                        output.append("- **Species:**")
                        for species_name in all_species:
                            direct_species.add(species_name)
                            # Look up species description
                            species_obj = model_structure._species_by_name.get(species_name)
                            desc = species_obj.description if species_obj else ""
                            if desc:
                                output.append(f"  - `{species_name}`: {desc}")
                            else:
                                output.append(f"  - `{species_name}`")

                    # Other parameters in this reaction (excluding current)
                    other_params = [p for p in rxn.parameters if p != param_name]
                    if other_params:
                        output.append("- **Other parameters:**")
                        for other_param_name in other_params:
                            other_param = model_structure.get_parameter(other_param_name)
                            desc = other_param.description if other_param else ""
                            if desc:
                                output.append(f"  - `{other_param_name}`: {desc}")
                            else:
                                output.append(f"  - `{other_param_name}`")

                    output.append("")
            else:
                output.append("**Direct reactions:** No reactions found using this parameter.")
                output.append("")

            # Add broader reaction network - other reactions involving the same species
            if direct_species:
                # Filter out compartment volumes (V_T, V_C, etc.) - they appear in nearly every
                # reaction and add noise. Real biological species have format "Compartment.Species"
                biological_species = {s for s in direct_species if "." in s}

                # Find all other reactions involving these species
                # Group by reaction name to avoid repetition
                seen_reactions: set = set()
                broader_reactions = []

                for species_name in biological_species:
                    related_rxns = model_structure.get_reactions_for_species(species_name)
                    for rxn in related_rxns:
                        # Skip if already processed or uses current parameter
                        if rxn.name in seen_reactions:
                            continue
                        if param_name in rxn.parameters:
                            continue
                        # Skip if uses any of the target parameters (shown separately)
                        if any(p in rxn.parameters for p in param_names):
                            continue

                        seen_reactions.add(rxn.name)
                        broader_reactions.append(rxn)

                if broader_reactions:
                    output.append(
                        "**Broader reaction network** (other reactions involving the same species):"
                    )
                    output.append("")

                    for rxn in broader_reactions:
                        reactant_str = " + ".join(rxn.reactants) if rxn.reactants else "null"
                        product_str = " + ".join(rxn.products) if rxn.products else "null"
                        output.append(f"- `{reactant_str} -> {product_str}`")
                        if rxn.rate_law:
                            output.append(f"  - Rate: `{rxn.rate_law}`")

                        # List species with descriptions
                        rxn_species = [s for s in rxn.reactants + rxn.products if "." in s]
                        if rxn_species:
                            output.append("  - Species:")
                            for sp_name in sorted(set(rxn_species)):
                                species_obj = model_structure._species_by_name.get(sp_name)
                                desc = species_obj.description if species_obj else ""
                                if desc:
                                    output.append(f"    - `{sp_name}`: {desc}")
                                else:
                                    output.append(f"    - `{sp_name}`")

                        # List parameters with descriptions
                        if rxn.parameters:
                            output.append("  - Parameters:")
                            for p_name in sorted(rxn.parameters):
                                p_obj = model_structure.get_parameter(p_name)
                                desc = p_obj.description if p_obj else ""
                                if desc:
                                    output.append(f"    - `{p_name}`: {desc}")
                                else:
                                    output.append(f"    - `{p_name}`")
                    output.append("")

            output.append("---")
            output.append("")

        return "\n".join(output) if output else "No parameter context available."

    def process(
        self,
        input_csv: Path,
        model_structure_file: Path,
        model_context_file: Path,
        species_units_file: Optional[Path] = None,
        reasoning_effort: str = "high",
    ) -> List[Dict[str, Any]]:
        """
        Process isolated system target inputs and generate prompts.

        Args:
            input_csv: CSV file with columns: target_id, parameters, notes (optional)
            model_structure_file: JSON file with model structure (from qsp-export-model --export-structure)
            model_context_file: Text file with high-level model description
            species_units_file: Optional JSON file mapping species -> units
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            List of prompt request dictionaries
        """
        import csv

        # Load model structure (required) - single source of truth for parameters, reactions, species
        if not model_structure_file or not model_structure_file.exists():
            raise ValueError(
                f"model_structure_file is required for IsolatedSystemTarget workflow. "
                f"Got: {model_structure_file}. "
                f"Run qsp-export-model with --export-structure flag to generate it."
            )
        model_structure = ModelStructure.from_json(model_structure_file)

        # Load model context (required)
        if not model_context_file or not model_context_file.exists():
            raise ValueError(
                f"model_context_file is required for IsolatedSystemTarget workflow. "
                f"Got: {model_context_file}"
            )
        with open(model_context_file, "r", encoding="utf-8") as f:
            model_context = f.read().strip()

        # Load species units if provided
        all_species_units = {}
        if species_units_file and species_units_file.exists():
            with open(species_units_file, "r") as f:
                all_species_units = json.load(f)

        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                target_id = row.get("target_id", f"target_{i}")
                parameters = row.get("parameters", "")
                notes = row.get("notes", "")

                if not parameters.strip():
                    print(f"Warning: Empty parameters for {target_id}, skipping")
                    continue

                # Build rich parameter context from model structure
                parameter_context = self.format_parameter_context(parameters, model_structure)

                # Build the prompt
                prompt = build_isolated_system_target_prompt(
                    parameters=parameters,
                    model_context=model_context,
                    parameter_context=parameter_context,
                    notes=notes,
                )

                # Create prompt dict
                custom_id = f"isolated_target_{target_id}_{i}"

                request = {
                    "custom_id": custom_id,
                    "prompt": prompt,
                    "pydantic_model": IsolatedSystemTarget,
                    "validation_context": {
                        "species_units": all_species_units,
                        "model_structure": model_structure,
                    },
                }

                requests.append(request)

        return requests


class SubmodelTargetPromptBuilder(PromptBuilder):
    """
    Creates prompts for submodel target extraction from in vitro/preclinical literature.

    Processes CSV input with parameter names and generates prompts for
    finding experimental data to calibrate those parameters using the SubmodelTarget schema.

    This is the preferred schema for new calibration targets, replacing IsolatedSystemTarget.

    Requires model_structure_file and model_context_file to be provided.
    """

    def get_workflow_type(self) -> str:
        return "submodel_target"

    def format_parameter_context(self, parameters: str, model_structure: ModelStructure) -> str:
        """
        Build rich context block for parameters from ModelStructure.

        Includes the broader reaction network - other reactions that involve
        the same species as the target parameter's reactions.

        Args:
            parameters: Comma-separated parameter names (e.g., "k_CD8_pro,k_CD8_death")
            model_structure: ModelStructure instance with parameters, reactions, species

        Returns:
            Formatted markdown text describing each parameter's role in the model
        """
        param_names = [p.strip() for p in parameters.split(",") if p.strip()]
        output = []

        for param_name in param_names:
            # Look up parameter in model structure
            param = model_structure.get_parameter(param_name)

            # Parameter header
            output.append(f"### Parameter: `{param_name}`")

            # Parameter info
            if param:
                if param.units:
                    output.append(f"- **Units:** {param.units}")
                if param.description:
                    output.append(f"- **Description:** {param.description}")
            else:
                output.append("- **Warning:** Parameter not found in model structure")

            output.append("")

            # Collect species from direct reactions for broader network lookup
            direct_species: set = set()

            # Get reactions using this parameter
            reactions = model_structure.get_reactions_for_parameter(param_name)
            if reactions:
                output.append(f"**Direct reactions ({len(reactions)}):**")
                output.append("")

                for i, rxn in enumerate(reactions, 1):
                    # Reaction header with stoichiometry
                    reactant_str = " + ".join(rxn.reactants) if rxn.reactants else "null"
                    product_str = " + ".join(rxn.products) if rxn.products else "null"
                    output.append(f"**{i}. Reaction:** `{reactant_str} -> {product_str}`")
                    if rxn.rate_law:
                        output.append(f"- **Rate:** `{rxn.rate_law}`")

                    # Related species (from reactants and products)
                    all_species = rxn.reactants + rxn.products
                    if all_species:
                        output.append("- **Species:**")
                        for species_name in all_species:
                            direct_species.add(species_name)
                            # Look up species description
                            species_obj = model_structure._species_by_name.get(species_name)
                            desc = species_obj.description if species_obj else ""
                            if desc:
                                output.append(f"  - `{species_name}`: {desc}")
                            else:
                                output.append(f"  - `{species_name}`")

                    # Other parameters in this reaction (excluding current)
                    other_params = [p for p in rxn.parameters if p != param_name]
                    if other_params:
                        output.append("- **Other parameters:**")
                        for other_param_name in other_params:
                            other_param = model_structure.get_parameter(other_param_name)
                            desc = other_param.description if other_param else ""
                            if desc:
                                output.append(f"  - `{other_param_name}`: {desc}")
                            else:
                                output.append(f"  - `{other_param_name}`")

                    output.append("")
            else:
                output.append("**Direct reactions:** No reactions found using this parameter.")
                output.append("")

            # Add broader reaction network - other reactions involving the same species
            if direct_species:
                # Filter out compartment volumes (V_T, V_C, etc.) - they appear in nearly every
                # reaction and add noise. Real biological species have format "Compartment.Species"
                biological_species = {s for s in direct_species if "." in s}

                # Find all other reactions involving these species
                # Group by reaction name to avoid repetition
                seen_reactions: set = set()
                broader_reactions = []

                for species_name in biological_species:
                    related_rxns = model_structure.get_reactions_for_species(species_name)
                    for rxn in related_rxns:
                        # Skip if already processed or uses current parameter
                        if rxn.name in seen_reactions:
                            continue
                        if param_name in rxn.parameters:
                            continue
                        # Skip if uses any of the target parameters (shown separately)
                        if any(p in rxn.parameters for p in param_names):
                            continue

                        seen_reactions.add(rxn.name)
                        broader_reactions.append(rxn)

                if broader_reactions:
                    output.append(
                        "**Broader reaction network** (other reactions involving the same species):"
                    )
                    output.append("")

                    for rxn in broader_reactions:
                        reactant_str = " + ".join(rxn.reactants) if rxn.reactants else "null"
                        product_str = " + ".join(rxn.products) if rxn.products else "null"
                        output.append(f"- `{reactant_str} -> {product_str}`")
                        if rxn.rate_law:
                            output.append(f"  - Rate: `{rxn.rate_law}`")

                        # List species with descriptions
                        rxn_species = [s for s in rxn.reactants + rxn.products if "." in s]
                        if rxn_species:
                            output.append("  - Species:")
                            for sp_name in sorted(set(rxn_species)):
                                species_obj = model_structure._species_by_name.get(sp_name)
                                desc = species_obj.description if species_obj else ""
                                if desc:
                                    output.append(f"    - `{sp_name}`: {desc}")
                                else:
                                    output.append(f"    - `{sp_name}`")

                        # List parameters with descriptions
                        if rxn.parameters:
                            output.append("  - Parameters:")
                            for p_name in sorted(rxn.parameters):
                                p_obj = model_structure.get_parameter(p_name)
                                desc = p_obj.description if p_obj else ""
                                if desc:
                                    output.append(f"    - `{p_name}`: {desc}")
                                else:
                                    output.append(f"    - `{p_name}`")
                    output.append("")

            output.append("---")
            output.append("")

        return "\n".join(output) if output else "No parameter context available."

    def process(
        self,
        input_csv: Path,
        model_structure_file: Path,
        model_context_file: Path,
        species_units_file: Optional[Path] = None,
        reasoning_effort: str = "high",
        previous_extractions_dir: Optional[Path] = None,
        reference_values_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process submodel target inputs and generate prompts.

        Args:
            input_csv: CSV file with columns: target_id, parameters, notes (optional)
            model_structure_file: JSON file with model structure (from qsp-export-model --export-structure)
            model_context_file: Text file with high-level model description
            species_units_file: Optional JSON file mapping species -> units
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            previous_extractions_dir: Optional path to directory with previous extractions
                                      to avoid re-using the same primary sources
            reference_values_file: Optional path to reference_values.yaml with curated constants

        Returns:
            List of prompt request dictionaries
        """
        import csv

        from qsp_llm_workflows.core.parameter_utils import (
            collect_existing_studies_for_submodel_target,
        )

        # Load model structure (required) - single source of truth for parameters, reactions, species
        if not model_structure_file or not model_structure_file.exists():
            raise ValueError(
                f"model_structure_file is required for SubmodelTarget workflow. "
                f"Got: {model_structure_file}. "
                f"Run qsp-export-model with --export-structure flag to generate it."
            )
        model_structure = ModelStructure.from_json(model_structure_file)

        # Load model context (required)
        if not model_context_file or not model_context_file.exists():
            raise ValueError(
                f"model_context_file is required for SubmodelTarget workflow. "
                f"Got: {model_context_file}"
            )
        with open(model_context_file, "r", encoding="utf-8") as f:
            model_context = f.read().strip()

        # Load species units if provided
        all_species_units = {}
        if species_units_file and species_units_file.exists():
            with open(species_units_file, "r") as f:
                all_species_units = json.load(f)

        # Load reference values database
        reference_db = None
        reference_db_entries = None
        if reference_values_file and reference_values_file.exists():
            import yaml as _yaml

            with open(reference_values_file) as _f:
                _ref_data = _yaml.safe_load(_f)
            reference_db_entries = _ref_data.get("values", [])
            reference_db = {v["name"]: float(v["value"]) for v in reference_db_entries}

        requests = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                target_id = row.get("target_id", f"target_{i}")
                parameters = row.get("parameters", "")
                notes = row.get("notes", "")
                cancer_type = row.get("cancer_type", "")

                if not parameters.strip():
                    print(f"Warning: Empty parameters for {target_id}, skipping")
                    continue

                # Build rich parameter context from model structure
                parameter_context = self.format_parameter_context(parameters, model_structure)

                # Collect existing studies from previous extractions
                # For multi-parameter targets, also check individual parameter files
                used_primary_studies = ""
                if previous_extractions_dir and cancer_type:
                    # Parse parameter names from comma-separated string
                    param_names = [p.strip() for p in parameters.split(",") if p.strip()]
                    used_primary_studies = collect_existing_studies_for_submodel_target(
                        target_id=target_id,
                        cancer_type=cancer_type,
                        previous_extractions_dir=previous_extractions_dir,
                        parameter_names=param_names,
                    )

                # Build the prompt
                prompt = build_submodel_target_prompt(
                    parameters=parameters,
                    model_context=model_context,
                    parameter_context=parameter_context,
                    notes=notes,
                    used_primary_studies=used_primary_studies,
                    reference_db_entries=reference_db_entries,
                )

                # Create prompt dict
                custom_id = f"submodel_target_{target_id}_{i}"

                request = {
                    "custom_id": custom_id,
                    "prompt": prompt,
                    "pydantic_model": SubmodelTarget,
                    "validation_context": {
                        "species_units": all_species_units,
                        "model_structure": model_structure,
                        "reference_db": reference_db,
                    },
                }

                requests.append(request)

        return requests
