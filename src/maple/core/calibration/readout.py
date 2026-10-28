"""The measurement a target reports, separate from how the model computes it."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maple.core.calibration.enums import (
    REQUIRES_REFERENCE,
    AcquisitionMethod,
    AnalytePool,
    AssayModality,
    DetectionBasis,
    NormalizationMode,
    Preservation,
    QuantityKind,
    SamplingRegion,
    SamplingUnit,
    SpecimenCompartment,
    SpecimenMaterial,
)


class MeasurementContext(BaseModel):
    """Factual categorical description of a literature measurement process.

    Maple records the complete context without deciding which fields create a
    statistical correction. That activation, its references, and its prior
    weights belong to the downstream analysis specification.
    """

    model_config = ConfigDict(extra="forbid")

    acquisition_method: AcquisitionMethod
    specimen_compartment: SpecimenCompartment
    specimen_material: SpecimenMaterial
    preservation: Preservation
    sampling_unit: SamplingUnit
    sampling_region: SamplingRegion
    detection_basis: DetectionBasis
    analyte_pool: AnalytePool
    normalization_mode: NormalizationMode


class ReadoutReference(BaseModel):
    """What a relative readout is measured against."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["timepoint", "scenario"] = Field(
        description="Whether the reference is another time in this trajectory or another scenario."
    )
    timepoint: Optional[float] = Field(default=None, description="Reference time.")
    timepoint_unit: Optional[str] = Field(
        default=None, description="Pint-parseable unit for ``timepoint``."
    )
    scenario: Optional[str] = Field(default=None, description="Reference scenario name.")

    @model_validator(mode="after")
    def _matches_kind(self) -> "ReadoutReference":
        if self.kind == "timepoint":
            if self.timepoint is None or not self.timepoint_unit:
                raise ValueError("reference kind='timepoint' needs timepoint and timepoint_unit.")
            if self.scenario is not None:
                raise ValueError("reference kind='timepoint' must not set scenario.")
        else:
            if not self.scenario:
                raise ValueError("reference kind='scenario' needs scenario.")
            if self.timepoint is not None:
                raise ValueError("reference kind='scenario' must not set timepoint.")
        return self


class Readout(BaseModel):
    """What was measured, separately from how the model computes it.

    The fields are factual. A downstream analysis explicitly selects which, if
    any, define shared measurement corrections.
    """

    model_config = ConfigDict(extra="forbid")

    quantity_kind: QuantityKind = Field(description="What kind of quantity was reported.")
    assay_modality: AssayModality = Field(description="How it was measured.")
    measurement_context: Optional[MeasurementContext] = Field(
        default=None,
        description=(
            "Closed-category facts about specimen acquisition, processing, sampling, "
            "detection, and normalization. Required for literature targets; absent on "
            "mechanistic assertions that do not represent a measurement."
        ),
    )
    numerator_species: List[str] = Field(
        min_length=1, description="Model species composing the numerator."
    )
    denominator_species: List[str] = Field(
        default_factory=list,
        description="Model species composing the denominator. Empty for an absolute quantity.",
    )
    experimental_denominator: Optional[str] = Field(
        default=None,
        description="What the experiment divided by, in the paper's own words. Required for a "
        "density or a fraction; ``denominator_species`` names the model side.\n\n"
        "Examples:\n"
        "- 'mm^2 of tumor tissue (whole section including stroma)'\n"
        "- 'all cells in ROI (all nucleated cells)'\n"
        "- 'CD3+ T cells (pan-T-cell marker)'",
    )
    reference: Optional[ReadoutReference] = Field(
        default=None,
        description="What a relative quantity is measured against. Required when quantity_kind "
        "is 'foldchange', forbidden otherwise.",
    )

    @model_validator(mode="after")
    def _composition_is_usable(self) -> "Readout":
        if set(self.numerator_species) == set(self.denominator_species):
            raise ValueError(
                "numerator_species and denominator_species are identical, which is constant at 1."
            )
        return self

    @model_validator(mode="after")
    def _reference_matches_quantity_kind(self) -> "Readout":
        needs = self.quantity_kind in REQUIRES_REFERENCE
        if needs and self.reference is None:
            raise ValueError(
                f"quantity_kind='{self.quantity_kind.value}' is a relative quantity but no "
                "reference is declared. Say what it is measured against: a timepoint in this "
                "trajectory, or another scenario."
            )
        if not needs and self.reference is not None:
            raise ValueError(
                f"quantity_kind='{self.quantity_kind.value}' is an absolute quantity but a "
                "reference is declared. Only relative quantities take one."
            )
        return self
