#!/usr/bin/env python3
"""
Queryable model structure for SimBiology models.

This module provides Pydantic models representing the structure of a SimBiology model
(species, reactions, compartments, parameters) with query methods for use as LLM tools.

Usage:
    from qsp_llm_workflows.core.model_structure import ModelStructure

    # Load from exported JSON
    model = ModelStructure.from_json("model_structure.json")

    # Query species in a compartment
    tumor_species = model.get_species_in_compartment("V_T")

    # Validate an entity exists
    result = model.validate_entity("k_prolif", entity_type="parameter")
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class ModelSpecies(BaseModel):
    """A species in the SimBiology model."""

    name: str = Field(description="Qualified name (Compartment.Species), e.g., 'V_T.CD8'")
    compartment: str = Field(description="Compartment containing this species")
    base_name: str = Field(description="Unqualified species name, e.g., 'CD8'")
    units: str = Field(default="dimensionless")
    description: str = Field(default="")

    @classmethod
    def from_qualified_name(
        cls, qualified_name: str, units: str = "dimensionless", description: str = ""
    ) -> ModelSpecies:
        """Create from qualified name like 'V_T.CD8'."""
        if "." in qualified_name:
            compartment, base_name = qualified_name.split(".", 1)
        else:
            compartment, base_name = "", qualified_name
        return cls(
            name=qualified_name,
            compartment=compartment,
            base_name=base_name,
            units=units,
            description=description,
        )


class ModelCompartment(BaseModel):
    """A compartment in the SimBiology model."""

    name: str
    volume: float | None = None
    volume_units: str = "milliliter"
    description: str = ""


class ModelParameter(BaseModel):
    """A parameter in the SimBiology model."""

    name: str
    value: float | None = None
    units: str = "dimensionless"
    description: str = ""


class ModelReaction(BaseModel):
    """A reaction in the SimBiology model."""

    name: str = Field(description="Reaction name or stoichiometry string")
    reactants: list[str] = Field(default_factory=list, description="Qualified species names")
    products: list[str] = Field(default_factory=list)
    rate_law: str = ""
    parameters: list[str] = Field(default_factory=list, description="Parameters in rate law")

    @computed_field
    @property
    def compartments(self) -> list[str]:
        """Compartments involved in this reaction."""
        comps = set()
        for species in self.reactants + self.products:
            if "." in species:
                comps.add(species.split(".")[0])
        return sorted(comps)

    @computed_field
    @property
    def is_multi_compartment(self) -> bool:
        """True if reaction spans multiple compartments."""
        return len(self.compartments) > 1


class EntityValidationResult(BaseModel):
    """Result of validating a model entity reference."""

    valid: bool
    entity_type: str
    entity_name: str
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ModelStructure(BaseModel):
    """Complete queryable model structure."""

    species: list[ModelSpecies] = Field(default_factory=list)
    compartments: list[ModelCompartment] = Field(default_factory=list)
    parameters: list[ModelParameter] = Field(default_factory=list)
    reactions: list[ModelReaction] = Field(default_factory=list)

    # Internal indices
    _species_by_name: dict[str, ModelSpecies] = {}
    _species_by_compartment: dict[str, list[ModelSpecies]] = {}
    _compartments_by_name: dict[str, ModelCompartment] = {}
    _parameters_by_name: dict[str, ModelParameter] = {}
    _reactions_by_name: dict[str, ModelReaction] = {}

    @model_validator(mode="after")
    def build_indices(self) -> ModelStructure:
        """Build lookup indices after model construction."""
        self._species_by_name = {s.name: s for s in self.species}
        self._species_by_compartment = defaultdict(list)
        for s in self.species:
            self._species_by_compartment[s.compartment].append(s)
        self._compartments_by_name = {c.name: c for c in self.compartments}
        self._parameters_by_name = {p.name: p for p in self.parameters}
        self._reactions_by_name = {r.name: r for r in self.reactions}
        return self

    # =========================================================================
    # Factory methods
    # =========================================================================

    @classmethod
    def from_json(cls, path: str | Path) -> ModelStructure:
        """Load model structure from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_species_units_json(cls, path: str | Path) -> ModelStructure:
        """Create partial ModelStructure from species_units.json (no reactions)."""
        with open(path) as f:
            data = json.load(f)

        species = []
        compartments_seen: set[str] = set()
        compartments = []
        parameters = []

        for name, info in data.items():
            units = info.get("units", "dimensionless")
            description = info.get("description", "")

            if "." in name:
                # Qualified species name
                species.append(
                    ModelSpecies.from_qualified_name(name, units=units, description=description)
                )
                comp_name = name.split(".")[0]
                if comp_name not in compartments_seen:
                    compartments_seen.add(comp_name)
                    compartments.append(ModelCompartment(name=comp_name))
            elif name.startswith("V_"):
                # Compartment
                compartments.append(
                    ModelCompartment(name=name, volume_units=units, description=description)
                )
                compartments_seen.add(name)
            else:
                # Parameter
                parameters.append(ModelParameter(name=name, units=units, description=description))

        return cls(species=species, compartments=compartments, parameters=parameters, reactions=[])

    def to_json(self, path: str | Path) -> None:
        """Save model structure to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    # =========================================================================
    # Basic queries
    # =========================================================================

    def get_species_in_compartment(self, compartment: str) -> list[ModelSpecies]:
        """Get all species in a compartment."""
        return list(self._species_by_compartment.get(compartment, []))

    def get_reactions_in_compartment(self, compartment: str) -> list[ModelReaction]:
        """Get reactions involving a compartment."""
        return [r for r in self.reactions if compartment in r.compartments]

    def get_reactions_for_species(self, species: str) -> list[ModelReaction]:
        """Get reactions involving a species."""
        return [r for r in self.reactions if species in r.reactants + r.products]

    def get_reactions_for_parameter(self, parameter: str) -> list[ModelReaction]:
        """Get reactions that use a parameter in their rate law."""
        return [r for r in self.reactions if parameter in r.parameters]

    def get_parameter(self, name: str) -> ModelParameter | None:
        """Get parameter by name, or None if not found."""
        return self._parameters_by_name.get(name)

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_entity(
        self,
        entity_name: str,
        entity_type: Literal["species", "compartment", "reaction", "parameter"] | None = None,
    ) -> EntityValidationResult:
        """Validate that an entity exists in the model."""
        checks = [
            ("species", self._species_by_name),
            ("compartment", self._compartments_by_name),
            ("reaction", self._reactions_by_name),
            ("parameter", self._parameters_by_name),
        ]

        if entity_type:
            checks = [(t, idx) for t, idx in checks if t == entity_type]

        for type_name, index in checks:
            if entity_name in index:
                return EntityValidationResult(
                    valid=True, entity_type=type_name, entity_name=entity_name
                )

        return EntityValidationResult(
            valid=False,
            entity_type=entity_type or "unknown",
            entity_name=entity_name,
            error=f"'{entity_name}' not found in model",
        )

    # =========================================================================
    # Convenience
    # =========================================================================

    @property
    def species_names(self) -> list[str]:
        return [s.name for s in self.species]

    @property
    def compartment_names(self) -> list[str]:
        return [c.name for c in self.compartments]

    def species_exists(self, name: str) -> bool:
        return name in self._species_by_name

    def compartment_exists(self, name: str) -> bool:
        return name in self._compartments_by_name

    def reaction_exists(self, name: str) -> bool:
        return name in self._reactions_by_name
