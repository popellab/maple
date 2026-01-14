#!/usr/bin/env python3
"""
Experimental context models for calibration targets.

Provides models for disease stage, treatment context, and the full
experimental context that describes where and how data was collected.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from qsp_llm_workflows.core.calibration.enums import (
    Indication,
    MouseSubspecifier,
    StageBurden,
    StageExtent,
    Species,
    System,
    TreatmentHistory,
    TreatmentStatus,
    enum_field_description,
)
from qsp_llm_workflows.core.calibration.shared_models import CellLine, CultureConditions


class Stage(BaseModel):
    """Disease stage with extent and burden."""

    extent: StageExtent = Field(description=enum_field_description(StageExtent, "Disease extent"))
    burden: StageBurden = Field(
        description=enum_field_description(
            StageBurden, "Disease burden (tumor size/volume category)"
        )
    )


class TreatmentContext(BaseModel):
    """Treatment context with history and current status."""

    history: List[TreatmentHistory] = Field(
        description=enum_field_description(
            TreatmentHistory, "Treatment history (select all that apply)"
        )
    )
    status: TreatmentStatus = Field(
        description=enum_field_description(
            TreatmentStatus,
            "Current treatment status. Use 'off_treatment' for untreated/baseline measurements",
        )
    )
    specifier: Optional[str] = Field(None, description="Optional drug name or class specifier")


class ExperimentalContext(BaseModel):
    """
    Experimental context for an observable.

    Uses typed enums for all context dimensions with hierarchical notation
    encoded in enum values (e.g., System.IN_VITRO_PRIMARY = "in_vitro.primary_cells").

    Supports both clinical/in vivo contexts (using indication, treatment, stage)
    and in vitro contexts (using cell_lines, culture_conditions).

    Note: `compartment` was removed as it's redundant with `system` - the System enum
    already encodes compartment information (e.g., in_vitro, clinical, mouse).
    """

    # Core fields (always required)
    species: Species = Field(description=enum_field_description(Species, "Species"))
    system: System = Field(description=enum_field_description(System, "Experimental system"))

    # Clinical/in vivo context (optional - used for integrated system targets)
    mouse_subspecifier: Optional[MouseSubspecifier] = Field(
        None,
        description=enum_field_description(
            MouseSubspecifier, "Optional mouse subspecifier (only if species is mouse)"
        ),
    )
    indication: Optional[Indication] = Field(
        None,
        description=enum_field_description(Indication, "Cancer indication (for clinical contexts)"),
    )
    treatment: Optional[TreatmentContext] = Field(
        None, description="Treatment context (for clinical contexts)"
    )
    stage: Optional[Stage] = Field(None, description="Disease stage (for clinical contexts)")

    # In vitro context (optional - used for isolated system targets)
    cell_lines: Optional[List[CellLine]] = Field(
        None,
        description="Cell lines used in experiment (for in vitro systems)",
    )
    culture_conditions: Optional[CultureConditions] = Field(
        None,
        description="Culture conditions (for in vitro systems)",
    )
    tissue_source: Optional[str] = Field(
        None,
        description="Tissue source for ex vivo systems (e.g., 'human PDAC resection specimen')",
    )
    assay_type: Optional[str] = Field(
        None,
        description="Assay type for cell-free systems (e.g., 'surface plasmon resonance')",
    )
