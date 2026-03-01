#!/usr/bin/env python3
"""
Export model definitions from SimBiology model.

This module extracts parameter and species definitions from a SimBiology model
and generates JSON definition files.
"""

import json
import pandas as pd
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any


class ModelDefinitionExporter:
    """Exports parameter and species definitions from SimBiology models."""

    def __init__(self, model_file: str, model_type: str = "matlab_script"):
        """
        Initialize the exporter.

        Args:
            model_file: Path to model file (MATLAB script .m or SimBiology project .sbproj)
            model_type: Type of model file - "matlab_script" or "simbiology_project"
        """
        self.model_file = Path(model_file).resolve()
        self.model_type = model_type
        self.temp_dir = None

        # Validate model type
        if model_type not in ["matlab_script", "simbiology_project"]:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be 'matlab_script' or 'simbiology_project'"
            )

        # Validate model file exists
        if not self.model_file.exists():
            raise ValueError(f"Model file does not exist: {model_file}")

    def export_definitions(self) -> Dict[str, Any]:
        """
        Export parameter definitions from the model.

        Note: Also exports species_units.json alongside the output file
        when called via export_to_json().

        Returns:
            Dict mapping parameter name -> {definition: dict}
        """
        print(f"Exporting definitions from {self.model_file}")

        # Create temporary directory for MATLAB output
        self.temp_dir = tempfile.mkdtemp(prefix="model_export_")

        try:
            # Run MATLAB to export model data
            self._run_matlab_export()

            # Load exported data
            simbio_params_df = self._load_simbio_parameters()
            simbio_species_df = self._load_simbio_species()
            simbio_compartments_df = self._load_simbio_compartments()
            model_context_df = self._load_model_context()

            # Cache data for species_units export
            self._cached_species_df = simbio_species_df
            self._cached_params_df = simbio_params_df
            self._cached_compartments_df = simbio_compartments_df

            # Generate parameter definitions
            param_definitions = self._generate_parameter_definitions(
                simbio_params_df, model_context_df, simbio_species_df
            )

            return param_definitions

        finally:
            # Clean up temporary directory
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)

    def _generate_species_units(
        self,
        simbio_species_df: pd.DataFrame,
        simbio_params_df: pd.DataFrame | None = None,
        simbio_compartments_df: pd.DataFrame | None = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate units and descriptions dictionary from SimBiology species, parameters, and compartments.

        Includes species (with compartments), parameters, and compartments since test statistic code
        (required_species) can reference all three types.

        Returns:
            Dict mapping name -> {units: str, description: str}
            Example: {
                'V_T.CD8': {'units': 'cell', 'description': 'CD8+ T cells in tumor'},
                'initial_tumour_diameter': {'units': 'centimeter', 'description': 'Initial tumor diameter'},
                'V_T': {'units': 'milliliter', 'description': 'Tumor compartment volume'}
            }
        """
        units_dict = {}

        # Add species units and descriptions (only qualified names with compartments)
        for _, row in simbio_species_df.iterrows():
            name = row["Name"]
            units = row["Units"] if pd.notna(row["Units"]) and row["Units"] else "dimensionless"
            description = row["Notes"] if pd.notna(row["Notes"]) else ""
            compartment = row.get("Compartment", "")

            # Only store qualified name (Compartment.Species) - skip simple names
            if pd.notna(compartment) and compartment:
                qualified_name = f"{compartment}.{name}"
                units_dict[qualified_name] = {"units": units, "description": str(description)}

        # Add parameter units (for scalar parameters used in test statistics)
        if simbio_params_df is not None:
            for _, row in simbio_params_df.iterrows():
                name = row["Name"]
                units = row["Units"] if pd.notna(row["Units"]) and row["Units"] else "dimensionless"
                description = row["Notes"] if pd.notna(row["Notes"]) else ""
                # Only add if not already present (species take priority)
                if name not in units_dict:
                    units_dict[name] = {"units": units, "description": str(description)}

        # Add compartment units (for compartment volumes like V_T, V_C, etc.)
        if simbio_compartments_df is not None and len(simbio_compartments_df) > 0:
            for _, row in simbio_compartments_df.iterrows():
                name = row["Name"]
                units = (
                    row["CapacityUnits"]
                    if pd.notna(row["CapacityUnits"]) and row["CapacityUnits"]
                    else "milliliter"
                )
                description = row["Notes"] if pd.notna(row.get("Notes", "")) else ""
                # Only add if not already present
                if name not in units_dict:
                    units_dict[name] = {"units": units, "description": str(description)}

        return units_dict

    def export_to_json(self, output_file: str):
        """
        Export parameter definitions and species units to JSON files.

        Creates two files:
        - <output_file>: Parameter definitions
        - <output_dir>/species_units.json: Species name -> unit mapping

        Args:
            output_file: Path to output JSON file for parameter definitions
        """
        # Export definitions (also caches species, params, and compartments data)
        definitions = self.export_definitions()

        # Generate units from cached data (includes species, parameters, and compartments)
        if hasattr(self, "_cached_species_df") and self._cached_species_df is not None:
            params_df = getattr(self, "_cached_params_df", None)
            compartments_df = getattr(self, "_cached_compartments_df", None)
            species_units = self._generate_species_units(
                self._cached_species_df, params_df, compartments_df
            )
        else:
            species_units = {}
            print("Warning: No species data available for species_units.json")

        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parameter definitions
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(definitions, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(definitions)} parameter definitions → {output_path}")

        # Write species units
        species_units_path = output_path.parent / "species_units.json"
        with open(species_units_path, "w", encoding="utf-8") as f:
            json.dump(species_units, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(species_units)} species units → {species_units_path}")

    def _run_matlab_export(self):
        """Run MATLAB to export model data."""
        print("Running MATLAB to export model data...")

        # Get the path to the MATLAB export script
        scripts_dir = Path(__file__).parent.parent / "matlab"
        export_script = scripts_dir / "export_model_definitions.m"

        if not export_script.exists():
            raise FileNotFoundError(f"MATLAB export script not found: {export_script}")

        # Add model directory and project root to MATLAB path
        model_dir = self.model_file.parent

        # Find project root (go up until we find a common project marker or reach 3 levels up)
        project_root = model_dir
        for _ in range(3):
            if project_root.parent == project_root:  # Reached filesystem root
                break
            project_root = project_root.parent
            # Stop if we find common project markers
            if any((project_root / marker).exists() for marker in [".git", "README.md", "LICENSE"]):
                break

        # Create MATLAB command with expanded path
        # Add: export script dir, model dir, project root, and recursively add project root subdirs
        matlab_cmd = (
            f"addpath('{scripts_dir}'); "
            f"addpath('{model_dir}'); "
            f"addpath(genpath('{project_root}')); "
            f"export_model_definitions('{self.model_file}', '{self.temp_dir}', '{self.model_type}')"
        )

        # Run MATLAB from project root
        result = subprocess.run(
            ["matlab", "-batch", matlab_cmd], capture_output=True, text=True, cwd=str(project_root)
        )

        if result.returncode != 0:
            raise RuntimeError(f"MATLAB export failed: {result.stderr}")

        print("MATLAB export completed successfully")

    def _load_simbio_parameters(self) -> pd.DataFrame:
        """Load simbio_parameters.csv from MATLAB export."""
        params_file = Path(self.temp_dir) / "simbio_parameters.csv"
        if not params_file.exists():
            raise FileNotFoundError(f"MATLAB did not generate {params_file}")

        return pd.read_csv(params_file)

    def _load_simbio_species(self) -> pd.DataFrame:
        """Load simbio_species.csv from MATLAB export."""
        species_file = Path(self.temp_dir) / "simbio_species.csv"
        if not species_file.exists():
            raise FileNotFoundError(f"MATLAB did not generate {species_file}")

        return pd.read_csv(species_file)

    def _load_simbio_compartments(self) -> pd.DataFrame:
        """Load simbio_compartments.csv from MATLAB export."""
        compartments_file = Path(self.temp_dir) / "simbio_compartments.csv"
        if not compartments_file.exists():
            # Compartments file is optional for backward compatibility
            return pd.DataFrame(columns=["Name", "CapacityUnits", "Notes"])

        return pd.read_csv(compartments_file)

    def _load_model_context(self) -> pd.DataFrame:
        """Load model_context.csv from MATLAB export."""
        context_file = Path(self.temp_dir) / "model_context.csv"
        if not context_file.exists():
            raise FileNotFoundError(f"MATLAB did not generate {context_file}")

        return pd.read_csv(context_file)

    def _generate_parameter_definitions(
        self,
        simbio_params_df: pd.DataFrame,
        model_context_df: pd.DataFrame,
        simbio_species_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generate parameter definition dictionaries.

        These are intermediary YAML files that will be processed to create
        the CSV list for LLM workflows.
        """
        param_definitions = {}

        # Create lookup dictionaries for descriptions
        # Parameters: name -> description (includes ALL parameters, even repeatedAssignment targets)
        param_descriptions = dict(zip(simbio_params_df["Name"], simbio_params_df["Notes"]))

        # Species: need to handle both "Species" and "Compartment.Species" lookups
        species_descriptions = {}
        for _, row in simbio_species_df.iterrows():
            species_name = row["Name"]
            compartment = row["Compartment"]
            description = row["Notes"]

            # Store under both names
            species_descriptions[species_name] = description
            if pd.notna(compartment):
                qualified_name = f"{compartment}.{species_name}"
                species_descriptions[qualified_name] = description

        # Also extract species and compartments from model_context with their descriptions
        # This captures compartments (like V_C, V_P, V_T) which aren't in simbio_species_df
        unique_species_and_compartments = self._extract_unique_species(model_context_df)
        for name, info in unique_species_and_compartments.items():
            # Only add if not already present (simbio_species takes priority)
            if name not in species_descriptions:
                species_descriptions[name] = info.get("description", "")

        # Check if IsRepeatedAssignment column exists
        has_repeated_assignment_flag = "IsRepeatedAssignment" in simbio_params_df.columns

        for _, param_row in simbio_params_df.iterrows():
            param_name = param_row["Name"]

            # Skip parameters that are updated via repeatedAssignment
            # These are kept in param_descriptions for lookups but not exported as main parameters
            if has_repeated_assignment_flag and param_row["IsRepeatedAssignment"]:
                continue

            # Get derivedFrom list - this is comma-separated list of parent parameter names
            derived_from_str = param_row.get("DerivedFrom", param_name)
            if pd.isna(derived_from_str):
                derived_from_list = [param_name]
            else:
                derived_from_list = [p.strip() for p in str(derived_from_str).split(",")]

            # Get model context entries for ALL parameters in derivedFrom list
            # PLUS the current parameter name (in case it was renamed after UserData was set)
            # This allows composite/renamed parameters to track their original contexts
            search_names = list(set(derived_from_list + [param_name]))
            param_context = model_context_df[
                model_context_df["Parameter"].isin(search_names)
            ].to_dict("records")

            # Create parameter definition (no value - that goes in parameter estimates)
            # Note: These YAMLs are intermediary files for generating CSV lists
            definition = {
                "parameter_definition": {
                    "name": param_name,
                    "units": param_row["Units"],
                    "description": str(param_row["Notes"]) if pd.notna(param_row["Notes"]) else "",
                    "derived_from": derived_from_list,  # Track parent parameters for matching
                    "model_context": self._process_model_context(
                        param_context, param_descriptions, species_descriptions, derived_from_list
                    ),
                }
            }

            param_definitions[param_name] = {
                "definition": definition,
            }

        return param_definitions

    def _process_model_context(
        self,
        context_entries: List[Dict],
        param_descriptions: Dict[str, str],
        species_descriptions: Dict[str, str],
        derived_from_list: List[str],
    ) -> Dict[str, Any]:
        """Process model context entries with full descriptions.

        Includes descriptions for:
        - other_parameters: All parameters appearing in the reaction/rule
        - other_species: All species appearing in the reaction/rule
        - derived_from: All parent parameters this parameter is derived from

        Looks up descriptions from:
        1. SimBiology model parameters (param_descriptions)
        2. SimBiology model species (species_descriptions)
        """
        processed = []

        for entry in context_entries:
            # Helper function to clean NaN values
            def clean_value(value, default=""):
                if pd.isna(value):
                    return default
                return str(value) if value is not None else default

            # Process other parameters with descriptions from model
            other_params = self._parse_json_field(entry.get("OtherParameters", "[]"))
            other_params_with_desc = []
            for pname in other_params:
                desc = param_descriptions.get(pname, "")
                other_params_with_desc.append(
                    {"name": pname, "description": str(desc) if pd.notna(desc) else ""}
                )

            # Process other species with descriptions from model
            species_list = self._parse_json_field(
                entry.get("OtherSpeciesWithNotes", "[]"), extract_names=True
            )
            other_species_with_desc = []
            for sname in species_list:
                desc = species_descriptions.get(sname, "")
                other_species_with_desc.append(
                    {"name": sname, "description": str(desc) if pd.notna(desc) else ""}
                )

            processed_entry = {
                "reaction": clean_value(entry.get("Reaction")),
                "reaction_rate": clean_value(entry.get("ReactionRate")),
                "rule": clean_value(entry.get("Rule")),
                "rule_type": clean_value(entry.get("RuleType")),
                "other_parameters": other_params_with_desc,
                "other_species": other_species_with_desc,
            }
            processed.append(processed_entry)

        # Add derived_from parameter descriptions from model
        derived_from_with_desc = []
        for pname in derived_from_list:
            desc = param_descriptions.get(pname, "")
            derived_from_with_desc.append(
                {"name": pname, "description": str(desc) if pd.notna(desc) else ""}
            )

        # Return context with derived_from info included
        return {"derived_from_context": derived_from_with_desc, "reactions_and_rules": processed}

    def _parse_json_field(self, json_str: str, extract_names: bool = False) -> List[str]:
        """
        Parse JSON string field, optionally extracting 'name' from dict items.

        Args:
            json_str: JSON string to parse
            extract_names: If True, extract 'name' field from dict items

        Returns:
            List of strings
        """
        try:
            # Handle both Python-style ['a', 'b'] and JSON-style ["a", "b"]
            json_str = str(json_str).replace("'", '"')
            data = json.loads(json_str)

            if extract_names and isinstance(data, list):
                return [item.get("name", "") for item in data if isinstance(item, dict)]
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    def _extract_unique_species(self, model_context_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Extract unique species from model context with their metadata."""
        unique_species = {}

        for _, row in model_context_df.iterrows():
            try:
                species_list = json.loads(str(row["OtherSpeciesWithNotes"]))
                for item in species_list:
                    if isinstance(item, dict) and "name" in item:
                        name = item["name"]
                        if name not in unique_species:
                            unique_species[name] = {
                                "description": item.get("notes", ""),
                                "compartment": item.get("compartment", ""),
                                "units": "dimensionless",  # Default
                            }
            except (json.JSONDecodeError, ValueError):
                continue

        return unique_species
