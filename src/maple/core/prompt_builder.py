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


from maple.core.prompts import (
    build_calibration_target_prompt,
    build_submodel_target_prompt,
)
from maple.core.calibration import CalibrationTarget
from maple.core.calibration.submodel_target import SubmodelTarget
from maple.core.model_structure import ModelStructure


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
            String identifier for this workflow type
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


def _collect_existing_studies(
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

    # Deduplicate file list
    yaml_files = list(set(yaml_files))

    if not yaml_files:
        return ""

    # Collect primary sources from all matching files
    sources_by_id: dict[str, tuple[str, dict]] = {}

    for yaml_file in sorted(yaml_files):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                study_data = yaml.safe_load(f)

            if not study_data:
                continue

            if "primary_data_source" in study_data and study_data["primary_data_source"]:
                source = study_data["primary_data_source"]
                doi = source.get("doi", "")
                url = source.get("url", "")
                source_id = doi or url or yaml_file.name
                if source_id not in sources_by_id:
                    sources_by_id[source_id] = (yaml_file.name, source)

        except Exception as e:
            print(f"Warning: Could not process {yaml_file}: {e}")
            continue

    if not sources_by_id:
        return ""

    header = "\n## Sources Already Used for This Target\n\n"
    header += (
        "**IMPORTANT:** The following primary sources have already been used in previous extractions "
        "for this target or its constituent parameters. "
        "DO NOT re-use these sources. Instead, find NEW sources not listed below.\n\n"
    )

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


class SubmodelTargetPromptBuilder(PromptBuilder):
    """
    Creates prompts for submodel target extraction from in vitro/preclinical literature.

    Processes CSV input with parameter names and generates prompts for
    finding experimental data to calibrate those parameters using the SubmodelTarget schema.

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
                used_primary_studies = ""
                if previous_extractions_dir and cancer_type:
                    param_names = [p.strip() for p in parameters.split(",") if p.strip()]
                    used_primary_studies = _collect_existing_studies(
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
