"""Cross-target checks against the cohort and readout registries.

Pydantic validators see one target at a time. These need the whole loaded set:
whether a cohort_id resolves, whether two targets claiming one readout compute
the same thing, whether a registry entry is used at all.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from maple.core.calibration.cohort import CohortRegistry
from maple.core.calibration.denominator_audit import numerator_and_denominator
from maple.core.calibration.readout import ReadoutRegistry


@dataclass(frozen=True)
class RegistryProblem:
    """One cross-target defect."""

    kind: str
    target_ids: Tuple[str, ...]
    detail: str


def _observable(target: Dict[str, Any]) -> Dict[str, Any]:
    return target.get("observable") or {}


def _estimates(target: Dict[str, Any]) -> Dict[str, Any]:
    return target.get("empirical_data") or {}


def _source_refs(target: Dict[str, Any]) -> List[str]:
    return sorted(
        {i.get("source_ref") for i in _estimates(target).get("inputs") or [] if i.get("source_ref")}
    )


def _is_literature(target: Dict[str, Any]) -> bool:
    return (target.get("epistemic_basis") or "literature") == "literature"


def find_registry_problems(
    targets: Dict[str, Dict[str, Any]],
    cohorts: CohortRegistry,
    readouts: ReadoutRegistry,
) -> List[RegistryProblem]:
    """Every resolvable defect in ``{target_id: parsed_yaml}`` against the registries."""
    problems: List[RegistryProblem] = []
    by_cohort = cohorts.as_dict()
    by_readout = readouts.as_dict()

    for tid, data in sorted(targets.items()):
        obs = _observable(data)
        rid = obs.get("readout_id")
        cid = data.get("cohort_id")

        if rid and rid not in by_readout:
            problems.append(
                RegistryProblem(
                    "unknown_readout",
                    (tid,),
                    f"readout_id '{rid}' is not in the readout registry. Add an entry, or point "
                    "at the existing readout this target measures.",
                )
            )
        if cid and cid not in by_cohort:
            problems.append(
                RegistryProblem(
                    "unknown_cohort",
                    (tid,),
                    f"cohort_id '{cid}' is not in the cohort registry.",
                )
            )

        if not _is_literature(data):
            continue

        refs = _source_refs(data)
        if len(refs) > 1 and cid:
            problems.append(
                RegistryProblem(
                    "pooled_target",
                    (tid,),
                    f"inputs cite {len(refs)} sources ({refs}) but the target names cohort "
                    f"'{cid}'. A pooled estimate is not a cohort: its patients were never "
                    "measured together, so there is no resampling distribution over them. "
                    "Split into one target per source.",
                )
            )

        cohort = by_cohort.get(cid) if cid else None
        if cohort is not None:
            n_eval = _estimates(data).get("n_evaluable")
            if n_eval is not None and n_eval > cohort.n_c:
                problems.append(
                    RegistryProblem(
                        "n_evaluable_exceeds_cohort",
                        (tid,),
                        f"n_evaluable={n_eval} exceeds cohort '{cid}' n_c={cohort.n_c}.",
                    )
                )
            src = (data.get("primary_data_source") or {}).get("source_tag")
            if src and src != cohort.source_tag:
                problems.append(
                    RegistryProblem(
                        "source_disagrees_with_cohort",
                        (tid,),
                        f"primary_data_source '{src}' differs from cohort '{cid}' source_tag "
                        f"'{cohort.source_tag}'.",
                    )
                )

    problems.extend(_duplicate_rows(targets))
    problems.extend(_readout_composition_disagrees(targets))
    return problems


def _duplicate_rows(targets: Dict[str, Dict[str, Any]]) -> List[RegistryProblem]:
    """Two targets giving one cohort the same readout are one row reported twice."""
    seen: Dict[Tuple[str, str], List[str]] = {}
    for tid, data in targets.items():
        cid, rid = data.get("cohort_id"), _observable(data).get("readout_id")
        if cid and rid:
            seen.setdefault((cid, rid), []).append(tid)
    return [
        RegistryProblem(
            "duplicate_row",
            tuple(sorted(members)),
            f"cohort '{cid}' has {len(members)} targets for readout '{rid}'. One cohort reports "
            "a readout once; a second target is either a duplicate or belongs to another cohort.",
        )
        for (cid, rid), members in sorted(seen.items())
        if len(members) > 1
    ]


def _readout_composition_disagrees(
    targets: Dict[str, Dict[str, Any]],
) -> List[RegistryProblem]:
    """Targets sharing a readout must compute the same species expression."""
    by_readout: Dict[str, List[Tuple[str, tuple, tuple]]] = {}
    for tid, data in targets.items():
        obs = _observable(data)
        rid = obs.get("readout_id")
        if not rid:
            continue
        num, den = numerator_and_denominator(obs.get("code") or "")
        by_readout.setdefault(rid, []).append((tid, tuple(sorted(num)), tuple(sorted(den))))

    problems = []
    for rid, members in sorted(by_readout.items()):
        distinct = {(n, d) for _, n, d in members}
        if len(distinct) > 1:
            detail = "; ".join(f"{tid}: {n} / {d}" for tid, n, d in sorted(members))
            problems.append(
                RegistryProblem(
                    "readout_composition_disagrees",
                    tuple(sorted(t for t, _, _ in members)),
                    f"targets sharing readout '{rid}' compute different species expressions "
                    f"({detail}). One readout is one quantity: either the code is wrong or "
                    "these are different readouts.",
                )
            )
    return problems


def check_registries(
    targets: Dict[str, Dict[str, Any]],
    cohorts: CohortRegistry,
    readouts: ReadoutRegistry,
) -> None:
    """Raise on any registry defect; warn on unused registry entries."""
    problems = find_registry_problems(targets, cohorts, readouts)
    warn_unused_registry_entries(targets, cohorts, readouts)
    if not problems:
        return
    lines = [f"{len(problems)} registry problem(s):"]
    for p in problems:
        lines.append(f"\n[{p.kind}] {', '.join(p.target_ids)}\n  {p.detail}")
    raise ValueError("\n".join(lines))


def warn_unused_registry_entries(
    targets: Dict[str, Dict[str, Any]],
    cohorts: CohortRegistry,
    readouts: ReadoutRegistry,
) -> List[str]:
    """Warn on registry entries no target refers to. Returns the unused ids."""
    used_c = {d.get("cohort_id") for d in targets.values() if d.get("cohort_id")}
    used_r = {_observable(d).get("readout_id") for d in targets.values()}
    unused = sorted(c.cohort_id for c in cohorts.cohorts if c.cohort_id not in used_c)
    unused += sorted(r.readout_id for r in readouts.readouts if r.readout_id not in used_r)
    if unused:
        warnings.warn(
            f"Registry entries no target uses: {unused}. Stale entries drift out of step with "
            "the corpus; remove them or add the target.",
            UserWarning,
        )
    return unused


def resolve_n(target: Dict[str, Any], cohorts: CohortRegistry) -> Optional[int]:
    """Patients behind this target's statistics: ``n_evaluable``, else the cohort's ``n_c``."""
    n_eval = _estimates(target).get("n_evaluable")
    if n_eval is not None:
        return int(n_eval)
    cohort = cohorts.get(target.get("cohort_id") or "")
    return cohort.n_c if cohort else None
