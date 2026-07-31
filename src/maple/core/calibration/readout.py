"""Readouts: what was measured, independent of who it was measured in.

A readout is the measured quantity; a cohort is the patients. A calibration
target is their intersection. Readouts are registered so one quantity observed in
several cohorts is a single entry.

The vocabularies are project-declared rather than hardcoded enums. This file
records data only. Which kinds get design-matrix columns, which is the reference
level, and how modalities collapse are modelling choices and live with the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class VocabularyEntry(BaseModel):
    """One registered id in a controlled vocabulary."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The value readouts refer to.")
    description: str = Field(description="What this id covers.")
    requires_reference: bool = Field(
        default=False,
        description="Quantity kinds only: whether a readout of this kind must declare what it "
        "is measured relative to, as a fold change must.",
    )


class ReadoutReference(BaseModel):
    """What a relative readout is measured against."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="Either 'timepoint' or 'scenario'.")
    timepoint: Optional[float] = Field(default=None, description="Reference time.")
    timepoint_units: Optional[str] = Field(
        default=None, description="Pint-parseable units for ``timepoint``."
    )
    scenario: Optional[str] = Field(default=None, description="Reference scenario name.")

    @model_validator(mode="after")
    def _matches_kind(self) -> "ReadoutReference":
        if self.kind == "timepoint":
            if self.timepoint is None or not self.timepoint_units:
                raise ValueError(
                    "ReadoutReference kind='timepoint' needs timepoint and timepoint_units."
                )
            if self.scenario is not None:
                raise ValueError("ReadoutReference kind='timepoint' must not set scenario.")
        elif self.kind == "scenario":
            if not self.scenario:
                raise ValueError("ReadoutReference kind='scenario' needs scenario.")
            if self.timepoint is not None:
                raise ValueError("ReadoutReference kind='scenario' must not set timepoint.")
        else:
            raise ValueError(
                f"ReadoutReference kind must be 'timepoint' or 'scenario', got '{self.kind}'."
            )
        return self


class Readout(BaseModel):
    """One measured quantity.

    ``numerator_species`` and ``denominator_species`` declare the composition. The
    executable stays on the target's ``observable.code``, which ``registry_audit``
    checks against this declaration.
    """

    model_config = ConfigDict(extra="forbid")

    readout_id: str = Field(
        description="Stable identifier referenced by ``Observable.readout_id``."
    )
    description: str = Field(description="What this quantity is.")
    quantity_kind: str = Field(description="A registered ``quantity_kinds`` id.")
    assay_modality: str = Field(description="A registered ``assay_modalities`` id.")
    units: str = Field(description="Pint-parseable reported units.")
    numerator_species: List[str] = Field(description="Model species composing the numerator.")
    denominator_species: List[str] = Field(
        default_factory=list,
        description="Model species composing the denominator. Empty for an absolute quantity.",
    )
    reference: Optional[ReadoutReference] = Field(
        default=None,
        description="What the quantity is measured relative to. Required when its "
        "``quantity_kind`` declares ``requires_reference``.",
    )
    notes: Optional[str] = Field(default=None, description="Anything the fields above miss.")

    @model_validator(mode="after")
    def _species_present(self) -> "Readout":
        if not self.numerator_species:
            raise ValueError(f"Readout '{self.readout_id}' declares no numerator_species.")
        overlap = sorted(set(self.numerator_species) & set(self.denominator_species))
        if overlap and set(self.numerator_species) == set(self.denominator_species):
            raise ValueError(
                f"Readout '{self.readout_id}' has identical numerator and denominator species, "
                "which is constant at 1."
            )
        return self


class ReadoutRegistry(BaseModel):
    """A project's readouts and the vocabularies they draw on."""

    model_config = ConfigDict(extra="forbid")

    quantity_kinds: List[VocabularyEntry] = Field(default_factory=list)
    assay_modalities: List[VocabularyEntry] = Field(default_factory=list)
    readouts: List[Readout] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique_and_resolve(self) -> "ReadoutRegistry":
        for name, entries in (
            ("quantity_kinds", self.quantity_kinds),
            ("assay_modalities", self.assay_modalities),
        ):
            ids = [e.id for e in entries]
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                raise ValueError(f"Duplicate id(s) in {name}: {dupes}")

        rids = [r.readout_id for r in self.readouts]
        dupes = sorted({i for i in rids if rids.count(i) > 1})
        if dupes:
            raise ValueError(f"Duplicate readout_id(s): {dupes}")

        kinds = {e.id: e for e in self.quantity_kinds}
        modalities = {e.id for e in self.assay_modalities}
        for r in self.readouts:
            if r.quantity_kind not in kinds:
                raise ValueError(
                    f"Readout '{r.readout_id}' uses quantity_kind '{r.quantity_kind}', which is "
                    f"not registered. Registered: {sorted(kinds)}. There is no 'other' bucket; "
                    "add an entry to quantity_kinds if this is a genuinely new kind."
                )
            if r.assay_modality not in modalities:
                raise ValueError(
                    f"Readout '{r.readout_id}' uses assay_modality '{r.assay_modality}', which "
                    f"is not registered. Registered: {sorted(modalities)}. There is no 'other' "
                    "bucket; add an entry to assay_modalities if this is a genuinely new assay."
                )
            if kinds[r.quantity_kind].requires_reference and r.reference is None:
                raise ValueError(
                    f"Readout '{r.readout_id}' is quantity_kind '{r.quantity_kind}', which "
                    "requires_reference, but declares no reference."
                )
        return self

    def get(self, readout_id: str) -> Optional[Readout]:
        return self.as_dict().get(readout_id)

    def as_dict(self) -> Dict[str, Readout]:
        return {r.readout_id: r for r in self.readouts}


def load_readouts(path: Path | str) -> ReadoutRegistry:
    """Load a readout registry YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Readout registry not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return ReadoutRegistry.model_validate(raw)
