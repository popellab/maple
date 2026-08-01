"""Cross-target checks against the cohort registry.

Pydantic validators see one target at a time. These need the whole loaded set:
whether a cohort_id resolves, whether a target pools several sources, whether two
targets give one cohort the same quantity twice.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from maple.core.calibration.cohort import CohortRegistry
from maple.core.calibration.denominator_audit import numerator_and_denominator


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
    targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry
) -> List[RegistryProblem]:
    """Every resolvable defect in ``{target_id: parsed_yaml}`` against the registry."""
    problems: List[RegistryProblem] = []
    by_cohort = cohorts.as_dict()

    for tid, data in sorted(targets.items()):
        cid = data.get("cohort_id")

        if cid and cid not in by_cohort:
            problems.append(
                RegistryProblem(
                    "unknown_cohort", (tid,), f"cohort_id '{cid}' is not in the cohort registry."
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
        if cohort is None:
            continue

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
    return problems


def _duplicate_rows(targets: Dict[str, Dict[str, Any]]) -> List[RegistryProblem]:
    """Two targets computing one model quantity for one cohort are one row reported twice.

    Keyed on the species expression parsed from ``observable.code``, so it cannot
    go stale against the code that actually runs.
    """
    seen: Dict[Tuple[str, tuple, tuple, Any], List[str]] = {}
    for tid, data in targets.items():
        cid = data.get("cohort_id")
        if not cid:
            continue
        obs = _observable(data)
        num, den = numerator_and_denominator(obs.get("code") or "")
        if not num and not den:
            continue
        key = (cid, tuple(sorted(num)), tuple(sorted(den)), obs.get("readout_time"))
        seen.setdefault(key, []).append(tid)
    return [
        RegistryProblem(
            "duplicate_row",
            tuple(sorted(members)),
            f"cohort '{key[0]}' has {len(members)} targets computing {key[1]} / {key[2]} at "
            f"t={key[3]}. One cohort reports a quantity once; a second target is either a "
            "duplicate or belongs to another cohort.",
        )
        for key, members in sorted(seen.items())
        if len(members) > 1
    ]


def check_registries(targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry) -> None:
    """Raise on any registry defect; warn on cohorts no target uses."""
    problems = find_registry_problems(targets, cohorts)
    warn_unused_cohorts(targets, cohorts)
    if not problems:
        return
    lines = [f"{len(problems)} registry problem(s):"]
    for p in problems:
        lines.append(f"\n[{p.kind}] {', '.join(p.target_ids)}\n  {p.detail}")
    raise ValueError("\n".join(lines))


def warn_unused_cohorts(targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry) -> List[str]:
    """Warn on cohorts no target refers to. Returns the unused ids."""
    used = {d.get("cohort_id") for d in targets.values() if d.get("cohort_id")}
    unused = sorted(c.cohort_id for c in cohorts.cohorts if c.cohort_id not in used)
    if unused:
        warnings.warn(
            f"Cohorts no target uses: {unused}. Stale entries drift out of step with the "
            "corpus; remove them or add the target.",
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
