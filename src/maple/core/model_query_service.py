#!/usr/bin/env python3
"""
Model query service with Pydantic AI tools.

This module provides a service that wraps ModelStructure and exposes
query methods as Pydantic AI tools for LLM agents.

Usage:
    from pydantic_ai import Agent
    from maple.core.model_query_service import ModelQueryService

    # Create service from model structure
    service = ModelQueryService.from_json("model_structure.json")

    # Create agent with model query tools
    agent = Agent('openai:gpt-4o', tools=service.get_tools())
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from maple.core.model_structure import (
    ModelStructure,
)


# =============================================================================
# Tool input/output models
# =============================================================================


class SpeciesQueryInput(BaseModel):
    """Input for querying species."""

    compartment: str | None = Field(
        default=None, description="Filter to species in this compartment (e.g., 'V_T')"
    )


class SpeciesInfo(BaseModel):
    """Information about a species."""

    name: str = Field(description="Qualified species name (e.g., 'V_T.CD8')")
    compartment: str
    units: str
    description: str = ""


class CompartmentInfo(BaseModel):
    """Information about a compartment."""

    name: str
    volume: float | None = None
    volume_units: str = "milliliter"
    species_count: int = 0


class ReactionInfo(BaseModel):
    """Information about a reaction."""

    name: str
    reactants: list[str]
    products: list[str]
    compartments: list[str]
    is_multi_compartment: bool = Field(description="True if reaction spans multiple compartments")


# =============================================================================
# Model Query Service
# =============================================================================


class ModelQueryService:
    """
    Service providing queryable access to model structure.

    Wraps ModelStructure and provides methods suitable for use as Pydantic AI tools.
    """

    def __init__(self, model: ModelStructure):
        """
        Initialize service with a ModelStructure.

        Args:
            model: The ModelStructure to query
        """
        self.model = model

    @classmethod
    def from_json(cls, path: str | Path) -> ModelQueryService:
        """Create service from model_structure.json file."""
        model = ModelStructure.from_json(path)
        return cls(model)

    @classmethod
    def from_species_units_json(cls, path: str | Path) -> ModelQueryService:
        """Create service from species_units.json (partial - no reactions)."""
        model = ModelStructure.from_species_units_json(path)
        return cls(model)

    # =========================================================================
    # Tool methods (designed for Pydantic AI @agent.tool decorator)
    # =========================================================================

    def query_species(self, compartment: str | None = None) -> list[dict[str, Any]]:
        """
        Query model species, optionally filtered by compartment.

        Use this to find what species exist in the model and their units.

        Args:
            compartment: Filter to species in this compartment (e.g., 'V_T' for tumor)

        Returns:
            List of species with name, compartment, units, and description
        """
        if compartment:
            species_list = self.model.get_species_in_compartment(compartment)
        else:
            species_list = self.model.species

        return [
            {
                "name": s.name,
                "compartment": s.compartment,
                "units": s.units,
                "description": s.description,
            }
            for s in species_list
        ]

    def query_compartments(self) -> list[dict[str, Any]]:
        """
        Query all model compartments.

        Use this to understand the model's spatial structure.

        Returns:
            List of compartments with name, volume, units, and species count
        """
        return [
            {
                "name": c.name,
                "volume": c.volume,
                "volume_units": c.volume_units,
                "description": c.description,
                "species_count": len(self.model.get_species_in_compartment(c.name)),
            }
            for c in self.model.compartments
        ]

    def query_reactions(
        self, compartment: str | None = None, species: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Query model reactions, optionally filtered.

        Use this to understand how species interact and which reactions
        cross compartment boundaries.

        Args:
            compartment: Filter to reactions involving this compartment
            species: Filter to reactions involving this species

        Returns:
            List of reactions with name, reactants, products, and compartments
        """
        reactions = self.model.reactions

        if compartment:
            reactions = [r for r in reactions if compartment in r.compartments]

        if species:
            reactions = [r for r in reactions if species in r.reactants + r.products]

        return [
            {
                "name": r.name,
                "reactants": r.reactants,
                "products": r.products,
                "rate_law": r.rate_law,
                "compartments": r.compartments,
                "is_multi_compartment": r.is_multi_compartment,
            }
            for r in reactions
        ]

    def query_parameters(self, name_pattern: str | None = None) -> list[dict[str, Any]]:
        """
        Query model parameters, optionally filtered by name pattern.

        Use this to find what parameters exist in the model and their units.

        Args:
            name_pattern: Optional substring to filter parameter names

        Returns:
            List of parameters with name, value, units, and description
        """
        params = self.model.parameters

        if name_pattern:
            params = [p for p in params if name_pattern.lower() in p.name.lower()]

        return [
            {
                "name": p.name,
                "value": p.value,
                "units": p.units,
                "description": p.description,
            }
            for p in params
        ]

    def validate_entity(self, entity_name: str, entity_type: str) -> dict[str, Any]:
        """
        Validate that an entity exists in the model.

        Use this to check parameter or species names before using them in submodel code.

        Args:
            entity_name: Name of the species, compartment, reaction, or parameter
            entity_type: One of 'species', 'compartment', 'reaction', 'parameter'

        Returns:
            Validation result with 'valid' boolean and any errors/warnings
        """
        if entity_type not in ("species", "compartment", "reaction", "parameter"):
            return {
                "valid": False,
                "entity_type": entity_type,
                "entity_name": entity_name,
                "error": f"Invalid entity_type '{entity_type}'. Must be species, compartment, reaction, or parameter.",
            }

        result = self.model.validate_entity(entity_name, entity_type)  # type: ignore
        return result.model_dump()

    # =========================================================================
    # Pydantic AI tool registration
    # =========================================================================

    def get_tools(self) -> list[callable]:
        """
        Get list of tool methods for Pydantic AI agent registration.

        Usage:
            service = ModelQueryService.from_json("model_structure.json")
            agent = Agent('openai:gpt-4o', tools=service.get_tools())
        """
        return [
            self.query_parameters,
            self.query_species,
            self.query_reactions,
        ]

    def register_tools(self, agent) -> None:
        """
        Register all query tools with a Pydantic AI agent.

        Usage:
            service = ModelQueryService.from_json("model_structure.json")
            agent = Agent('openai:gpt-4o')
            service.register_tools(agent)

        Args:
            agent: Pydantic AI Agent instance
        """

        @agent.tool
        def query_species(ctx, compartment: str | None = None) -> list[dict[str, Any]]:
            """Query model species, optionally filtered by compartment."""
            return self.query_species(compartment)

        @agent.tool
        def query_compartments(ctx) -> list[dict[str, Any]]:
            """Query all model compartments."""
            return self.query_compartments()

        @agent.tool
        def query_reactions(
            ctx, compartment: str | None = None, species: str | None = None
        ) -> list[dict[str, Any]]:
            """Query model reactions, optionally filtered by compartment or species."""
            return self.query_reactions(compartment, species)

        @agent.tool
        def query_parameters(ctx, name_pattern: str | None = None) -> list[dict[str, Any]]:
            """Query model parameters, optionally filtered by name pattern."""
            return self.query_parameters(name_pattern)

        @agent.tool
        def validate_entity(ctx, entity_name: str, entity_type: str) -> dict[str, Any]:
            """Validate that an entity (species, parameter, etc.) exists in the model."""
            return self.validate_entity(entity_name, entity_type)
