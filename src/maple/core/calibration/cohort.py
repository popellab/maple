"""Cohorts: one study's patients, measured once.

A calibration target names one row of the observation vector. The cohort is the
unit of independence: population inference forms one covariance block per cohort,
whose off-diagonal comes from resampling whole patients. Cross-target checks live
in ``registry_audit``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EligibilityInterval(BaseModel):
    """An inclusion criterion the study applied, as an interval on a measurement.

    Stated in the reported units of the named target, since that is what the study
    screened on.
    """

    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(
        description="Calibration target whose observable the criterion is applied to. "
        "Eligibility is stated in that measurement's reported units."
    )
    lo: Optional[float] = Field(
        default=None, description="Inclusive lower bound in ``units``. Omit if one-sided."
    )
    hi: Optional[float] = Field(
        default=None, description="Inclusive upper bound in ``units``. Omit if one-sided."
    )
    units: str = Field(description="Pint-parseable units the bounds are stated in.")
    rationale: str = Field(description="Where the source states the criterion.")

    @model_validator(mode="after")
    def _bounds_are_usable(self) -> "EligibilityInterval":
        if self.lo is None and self.hi is None:
            raise ValueError(
                f"EligibilityInterval on '{self.target_id}' has neither lo nor hi; it selects "
                "everyone. Drop it instead."
            )
        if self.lo is not None and self.hi is not None and self.lo >= self.hi:
            raise ValueError(
                f"EligibilityInterval on '{self.target_id}' has lo={self.lo} >= hi={self.hi}, "
                "which selects nobody."
            )
        return self


class Cohort(BaseModel):
    """One study's patients. Every literature-derived target names exactly one."""

    model_config = ConfigDict(extra="forbid")

    cohort_id: str = Field(
        description="Stable identifier referenced by ``CalibrationTarget.cohort_id``. Name it "
        "for the patients, not for a measurement or scenario."
    )
    description: str = Field(
        description="Who these patients are and what measurement occasion the statistics come from."
    )
    scenarios: List[str] = Field(
        description="QSP scenario(s) this cohort's targets are evaluated under. More than one "
        "when the cohort reports a contrast between scenarios in the same patients, such as a "
        "paired pre/post fold change."
    )
    n_c: int = Field(
        ge=1,
        description="Patients the reported statistics are computed over. This is the resampling "
        "n, not the number enrolled or screened, and never a sum across studies.",
    )
    n_c_is_floor: bool = Field(
        default=False,
        description="True when ``n_c`` is a lower bound rather than an exact count.",
    )
    source_tag: str = Field(
        description="The single source reporting this cohort. A target drawing on several "
        "sources is a pooled estimate, not a cohort, and must be split."
    )
    eligibility: List[EligibilityInterval] = Field(
        default_factory=list,
        description="Inclusion criteria the study applied. Empty when it reports its full sample.",
    )
    shares_patients_with: List[str] = Field(
        default_factory=list,
        description="Other ``cohort_id``s drawing on overlapping patients. Blocks are assumed "
        "independent, so an entry here records a violation: prefer merging the cohorts via "
        "``scenarios``.",
    )
    notes: Optional[str] = Field(default=None, description="Anything the fields above miss.")

    @model_validator(mode="after")
    def _well_formed(self) -> "Cohort":
        if not self.scenarios:
            raise ValueError(f"Cohort '{self.cohort_id}' declares no scenarios.")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError(f"Cohort '{self.cohort_id}' repeats a scenario.")
        if self.cohort_id in self.shares_patients_with:
            raise ValueError(f"Cohort '{self.cohort_id}' lists itself in shares_patients_with.")
        if len(set(self.shares_patients_with)) != len(self.shares_patients_with):
            raise ValueError(f"Cohort '{self.cohort_id}' repeats an id in shares_patients_with.")
        return self


class CohortRegistry(BaseModel):
    """The declared cohorts of a project."""

    model_config = ConfigDict(extra="forbid")

    cohorts: List[Cohort] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique_and_overlaps_resolve(self) -> "CohortRegistry":
        counts: Dict[str, int] = {}
        for c in self.cohorts:
            counts[c.cohort_id] = counts.get(c.cohort_id, 0) + 1
        dupes = sorted(k for k, v in counts.items() if v > 1)
        if dupes:
            raise ValueError(f"Duplicate cohort_id(s) in registry: {dupes}")

        by_id = {c.cohort_id: c for c in self.cohorts}
        for c in self.cohorts:
            missing = [o for o in c.shares_patients_with if o not in by_id]
            if missing:
                raise ValueError(
                    f"Cohort '{c.cohort_id}' declares shares_patients_with {missing}, which are "
                    "not in the registry."
                )
            # Overlap is symmetric; a one-sided declaration hides it from
            # whichever cohort is inspected first.
            for other in c.shares_patients_with:
                if c.cohort_id not in by_id[other].shares_patients_with:
                    raise ValueError(
                        f"Cohort '{c.cohort_id}' declares overlap with '{other}', but '{other}' "
                        "does not declare it back."
                    )
        return self

    def get(self, cohort_id: str) -> Optional[Cohort]:
        return self.as_dict().get(cohort_id)

    def as_dict(self) -> Dict[str, Cohort]:
        return {c.cohort_id: c for c in self.cohorts}


def load_cohorts(path: Path | str) -> CohortRegistry:
    """Load a cohort registry YAML: a mapping with a ``cohorts:`` key, or a bare list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cohort registry not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if isinstance(raw, list):
        raw = {"cohorts": raw}
    return CohortRegistry.model_validate(raw)
