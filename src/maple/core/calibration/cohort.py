"""Cohorts: one study's patients, measured once.

A calibration target names one row of the observation vector. A cohort owns an
n, an eligibility rule and a level, but it is not the unit of independence: the
block is, and population inference forms one covariance block per block, whose
off-diagonal comes from resampling whole patients. Cohorts that share people
declare a ``PatientBlock`` saying how many; everything else is its own block.
Cross-target checks live in ``registry_audit``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

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
    notes: Optional[str] = Field(default=None, description="Anything the fields above miss.")

    @model_validator(mode="after")
    def _well_formed(self) -> "Cohort":
        if not self.scenarios:
            raise ValueError(f"Cohort '{self.cohort_id}' declares no scenarios.")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError(f"Cohort '{self.cohort_id}' repeats a scenario.")
        return self


class Stratum(BaseModel):
    """Patients labelled by every cohort they belong to, and how many there are.

    One line of a partition: a patient falls in exactly one stratum, and cohort
    membership follows from which strata name it.
    """

    model_config = ConfigDict(extra="forbid")

    cohorts: List[str] = Field(
        min_length=1,
        description="Every ``cohort_id`` these patients belong to. A single entry means "
        "patients only that cohort measured.",
    )
    n: int = Field(ge=1, description="How many patients. Omit the stratum rather than write 0.")

    @model_validator(mode="after")
    def _cohorts_distinct(self) -> "Stratum":
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError(f"Stratum {sorted(self.cohorts)} repeats a cohort_id.")
        return self

    @property
    def key(self) -> FrozenSet[str]:
        return frozenset(self.cohorts)


class PatientBlock(BaseModel):
    """Cohorts that share people. Declared only where they do.

    Resampling draws a block's patients once and evaluates each cohort's rows on
    its own members, which is what carries the across-cohort correlation into the
    covariance. Cohorts sharing nobody need no entry.

    ``strata`` counts the sharing out. It is the Venn regions of the member
    cohorts, which is a complete description of the overlap: identities never
    reach the resample, so the counts are all of it. Counting patients also keeps
    the description realisable, since a partition of people always describes some
    people, and settles three-way memberships that pairwise overlaps leave open.

    Omit ``strata`` when the overlap is known but uncounted, which is the usual
    state for two papers reporting one trial. The block still says the cohorts are
    not independent; it just cannot say by how much, so a consumer building a
    covariance has to fall back and report that it did.
    """

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(description="Stable identifier. Name it for the study or trial.")
    description: str = Field(description="Who these patients are and how the cohorts divide them.")
    cohorts: List[str] = Field(
        min_length=2,
        description="The ``cohort_id``s this block spans. A block records sharing between "
        "cohorts, so one cohort is not a block.",
    )
    strata: Optional[List[Stratum]] = Field(
        default=None,
        description="The partition of the block's patients. Omit when the overlap is known "
        "but nobody has counted it.",
    )
    notes: Optional[str] = Field(
        default=None, description="Where the source states the split, or why it does not."
    )

    @model_validator(mode="after")
    def _well_formed(self) -> "PatientBlock":
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError(f"Block '{self.block_id}' repeats a cohort_id.")
        if self.strata is None:
            return self
        if not self.strata:
            raise ValueError(
                f"Block '{self.block_id}' has an empty strata list. Omit the key to declare an "
                "uncounted overlap; an empty partition describes no patients."
            )
        keys = [s.key for s in self.strata]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Block '{self.block_id}' repeats a stratum. Two lines naming the same cohorts "
                "describe one group of patients; sum them."
            )
        declared = set(self.cohorts)
        named = set().union(*keys)
        if named - declared:
            raise ValueError(
                f"Block '{self.block_id}' has strata naming {sorted(named - declared)}, which "
                "are not in its cohorts."
            )
        if declared - named:
            raise ValueError(
                f"Block '{self.block_id}' spans {sorted(declared - named)} but no stratum places "
                "them. Every member needs patients, or it does not belong to the block."
            )
        return self

    @property
    def is_quantified(self) -> bool:
        """Whether the overlap is counted, and so whether a joint resample is defined."""
        return self.strata is not None

    @property
    def members(self) -> FrozenSet[str]:
        return frozenset(self.cohorts)

    @property
    def n_patients(self) -> Optional[int]:
        """Patients in the block, which is the number a joint resample draws."""
        return None if self.strata is None else sum(s.n for s in self.strata)

    def size_of(self, cohort_id: str) -> Optional[int]:
        """Block patients belonging to ``cohort_id``. Equals that cohort's ``n_c``."""
        if self.strata is None:
            return None
        return sum(s.n for s in self.strata if cohort_id in s.key)

    def overlap(self, a: str, b: str) -> Optional[int]:
        """Patients in both cohorts. Zero is a fact, not a missing value."""
        if self.strata is None:
            return None
        return sum(s.n for s in self.strata if a in s.key and b in s.key)


class CohortRegistry(BaseModel):
    """The declared cohorts of a project, and the blocks over them.

    Counted blocks partition: a cohort belongs to at most one, since its patients
    can be divided up only once. Uncounted blocks overlay, so a cohort may sit in
    any number of them. That is what lets a study whose internal sharing is
    reported sit inside a wider overlap nobody has counted.
    """

    model_config = ConfigDict(extra="forbid")

    cohorts: List[Cohort] = Field(default_factory=list)
    blocks: List[PatientBlock] = Field(
        default_factory=list,
        description="Groups of cohorts that share patients. Cohorts sharing nobody are their "
        "own block and need no entry.",
    )

    @model_validator(mode="after")
    def _ids_unique_and_blocks_resolve(self) -> "CohortRegistry":
        counts: Dict[str, int] = {}
        for c in self.cohorts:
            counts[c.cohort_id] = counts.get(c.cohort_id, 0) + 1
        dupes = sorted(k for k, v in counts.items() if v > 1)
        if dupes:
            raise ValueError(f"Duplicate cohort_id(s) in registry: {dupes}")

        block_ids = [b.block_id for b in self.blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError(
                f"Duplicate block_id(s) in registry: "
                f"{sorted({b for b in block_ids if block_ids.count(b) > 1})}"
            )

        by_id = {c.cohort_id: c for c in self.cohorts}
        claimed: Dict[str, str] = {}
        for b in self.blocks:
            missing = sorted(set(b.cohorts) - set(by_id))
            if missing:
                raise ValueError(
                    f"Block '{b.block_id}' names {missing}, which are not in the registry."
                )
            if not b.is_quantified:
                continue
            for cid in b.cohorts:
                # One partition per cohort: two would each claim all its patients.
                if cid in claimed:
                    raise ValueError(
                        f"Cohort '{cid}' is counted by blocks '{claimed[cid]}' and "
                        f"'{b.block_id}'. Its patients can be divided up once; leave one "
                        "block's strata off to declare an uncounted overlap instead."
                    )
                claimed[cid] = b.block_id
                size = b.size_of(cid)
                if size != by_id[cid].n_c:
                    raise ValueError(
                        f"Block '{b.block_id}' places {size} patients in cohort '{cid}', which "
                        f"declares n_c={by_id[cid].n_c}. The strata are that cohort's patients, "
                        "so they have to add up to it."
                    )
        return self

    def get(self, cohort_id: str) -> Optional[Cohort]:
        return self.as_dict().get(cohort_id)

    def as_dict(self) -> Dict[str, Cohort]:
        return {c.cohort_id: c for c in self.cohorts}

    def counted_block_for(self, cohort_id: str) -> Optional[PatientBlock]:
        """The block whose strata divide this cohort's patients, if one does."""
        for b in self.blocks:
            if b.is_quantified and cohort_id in b.members:
                return b
        return None

    def uncounted_blocks_for(self, cohort_id: str) -> List[PatientBlock]:
        """Declared overlaps involving this cohort that nobody has counted."""
        return [b for b in self.blocks if not b.is_quantified and cohort_id in b.members]

    @property
    def uncounted_blocks(self) -> List[PatientBlock]:
        """Every declared overlap without strata. A consumer building a covariance
        cannot honour these and has to report that it drew their cohorts apart."""
        return [b for b in self.blocks if not b.is_quantified]


def load_cohorts(path: Path | str) -> CohortRegistry:
    """Load a cohort registry YAML: a mapping with ``cohorts:`` and optional ``blocks:``,
    or a bare list of cohorts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cohort registry not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if isinstance(raw, list):
        raw = {"cohorts": raw}
    return CohortRegistry.model_validate(raw)
