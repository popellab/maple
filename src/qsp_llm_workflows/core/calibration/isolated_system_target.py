#!/usr/bin/env python3
"""
Isolated system target model for in vitro calibration data.

Extends CalibrationTarget with cuts that define reduced model boundary conditions.
Used for in vitro experiments where certain interactions are "clamped" or removed.
"""

import warnings
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from qsp_llm_workflows.core.calibration.calibration_target_models import CalibrationTarget
from qsp_llm_workflows.core.calibration.enums import System


# ============================================================================
# Cut Models
# ============================================================================


class SpeciesCut(BaseModel):
    """Cut applied to a species in the model (for isolated/in vitro systems)."""

    type: Literal["species"] = Field(default="species", description="Cut type")
    name: str = Field(description="Fully qualified species name (e.g., 'V_T.IL2', 'V_C.Drug')")
    condition: Literal["clamped", "excluded", "zero_flux", "prescribed"] = Field(
        description="Condition applied to the species"
    )

    # For clamped condition
    value: Optional[float] = Field(None, description="Fixed value (required if clamped)")
    unit: Optional[str] = Field(
        None, description="Pint-parseable unit (required if clamped or prescribed)"
    )

    # For prescribed condition
    time_course: Optional[str] = Field(
        None,
        description="Python function: def prescribed_value(time, ureg) -> Quantity",
    )

    @model_validator(mode="after")
    def validate_condition_fields(self) -> "SpeciesCut":
        """Ensure required fields are present for each condition type."""
        if self.condition == "clamped":
            if self.value is None:
                raise ValueError("'value' is required when condition is 'clamped'")
            if self.unit is None:
                raise ValueError("'unit' is required when condition is 'clamped'")
        elif self.condition == "prescribed":
            if self.time_course is None:
                raise ValueError("'time_course' is required when condition is 'prescribed'")
            if self.unit is None:
                raise ValueError("'unit' is required when condition is 'prescribed'")
        return self


class CompartmentCut(BaseModel):
    """Cut that excludes an entire compartment (for isolated/in vitro systems)."""

    type: Literal["compartment"] = Field(default="compartment", description="Cut type")
    name: str = Field(description="Compartment name (e.g., 'V_C', 'V_T', 'V_LN')")
    condition: Literal["excluded"] = Field(
        default="excluded", description="Compartment condition (always 'excluded')"
    )


class ReactionCut(BaseModel):
    """Cut that disables a reaction (for isolated/in vitro systems)."""

    type: Literal["reaction"] = Field(default="reaction", description="Cut type")
    name: str = Field(description="Reaction or rule name")
    condition: Literal["disabled"] = Field(
        default="disabled", description="Reaction condition (always 'disabled')"
    )


Cut = Annotated[
    Union[SpeciesCut, CompartmentCut, ReactionCut],
    Field(discriminator="type"),
]


# ============================================================================
# Isolated System Target (inherits from CalibrationTarget)
# ============================================================================


class IsolatedSystemTarget(CalibrationTarget):
    """
    A calibration target from an isolated/in vitro experimental system.

    Inherits all fields from CalibrationTarget and adds:
    - cuts: Define which parts of the model are active (reduced model)

    Used for in vitro experiments (cell lines, co-cultures, organoids) where
    certain interactions have been experimentally "clamped" or removed.

    The subnetwork is defined declaratively via cuts - matching how experimentalists
    think: "I'll control X, remove Y, and measure Z."
    """

    # The key addition - defines reduced model via boundary conditions
    cuts: List[Cut] = Field(
        description=(
            "List of cuts defining the subnetwork boundary conditions.\n"
            "- Species cuts: clamped (fixed value), excluded, zero_flux, prescribed (time course)\n"
            "- Compartment cuts: excluded (removes entire compartment)\n"
            "- Reaction cuts: disabled (rate = 0)"
        )
    )

    @field_validator("cuts")
    @classmethod
    def at_least_one_cut(cls, v: List[Cut]) -> List[Cut]:
        """Ensure at least one cut is provided."""
        if len(v) < 1:
            raise ValueError(
                "At least one cut is required to define the isolated system. "
                "Use CompartmentCut to exclude compartments not present in vitro."
            )
        return v

    @model_validator(mode="after")
    def validate_no_duplicate_cuts(self) -> "IsolatedSystemTarget":
        """Ensure no duplicate cuts (same type + name)."""
        seen = set()
        for cut in self.cuts:
            key = (cut.type, cut.name)
            if key in seen:
                raise ValueError(f"Duplicate cut: type='{cut.type}', name='{cut.name}'")
            seen.add(key)
        return self

    @model_validator(mode="after")
    def validate_in_vitro_system(self) -> "IsolatedSystemTarget":
        """Warn if system type doesn't match in vitro context."""
        system = self.experimental_context.system
        in_vitro_systems = {
            System.IN_VITRO_ORGANOID,
            System.IN_VITRO_PRIMARY_CELLS,
            System.IN_VITRO_CELL_LINE,
            System.EX_VIVO_FRESH,
            System.EX_VIVO_CULTURED,
        }

        if system not in in_vitro_systems:
            warnings.warn(
                f"IsolatedSystemTarget typically uses in_vitro or ex_vivo systems, "
                f"but system is '{system.value}'. Ensure cuts properly define the isolated context.",
                UserWarning,
            )

        return self


__all__ = [
    "SpeciesCut",
    "CompartmentCut",
    "ReactionCut",
    "Cut",
    "IsolatedSystemTarget",
]
