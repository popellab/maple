"""Cross-target checks against the cohort registry.

Pydantic validators see one target at a time. These need the whole loaded set:
whether a cohort_id resolves, whether a target pools several sources, whether two
targets give one cohort the same quantity twice, and whether a cohort's rows are
deterministic functions of each other.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from maple.core.calibration.cohort import CohortRegistry


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


def _arms(target: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per-role inputs of a cross-scenario target; empty for a scalar target."""
    return _observable(target).get("inputs") or []


def _readout(target: Dict[str, Any]) -> Dict[str, Any]:
    return _observable(target).get("readout") or {}


def _composition(readout: Dict[str, Any]) -> Optional[Tuple[tuple, tuple]]:
    """Declared (numerator, denominator) species, or None when nothing is declared."""
    num = tuple(sorted(readout.get("numerator_species") or []))
    den = tuple(sorted(readout.get("denominator_species") or []))
    return (num, den) if num else None


def _cohort_ids(target: Dict[str, Any]) -> List[str]:
    """Cohorts this target draws on: one for a scalar target, one per arm for a contrast."""
    cid = target.get("cohort_id")
    if cid:
        return [cid]
    return [a["cohort_id"] for a in _arms(target) if a.get("cohort_id")]


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

    problems.extend(_cross_scenario_arms(targets, cohorts))
    problems.extend(_duplicate_rows(targets))
    problems.extend(_singular_blocks(targets))
    return problems


def _cross_scenario_arms(
    targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry
) -> List[RegistryProblem]:
    """Each arm of a contrast against the cohort it names."""
    by_cohort = cohorts.as_dict()
    problems: List[RegistryProblem] = []

    for tid, data in sorted(targets.items()):
        arms = _arms(data)
        if not arms:
            continue
        for arm in arms:
            cid = arm.get("cohort_id")
            if not cid:
                continue
            cohort = by_cohort.get(cid)
            if cohort is None:
                problems.append(
                    RegistryProblem(
                        "unknown_cohort",
                        (tid,),
                        f"role '{arm.get('role')}' names cohort '{cid}', which is not in the "
                        "cohort registry.",
                    )
                )
                continue
            if arm.get("scenario") and arm["scenario"] not in cohort.scenarios:
                problems.append(
                    RegistryProblem(
                        "scenario_not_in_cohort",
                        (tid,),
                        f"role '{arm.get('role')}' runs scenario '{arm['scenario']}' but "
                        f"cohort '{cid}' declares {cohort.scenarios}. The arm's patients were "
                        "not measured under that condition.",
                    )
                )
            n_eval = arm.get("n_evaluable")
            if n_eval is not None and n_eval > cohort.n_c:
                problems.append(
                    RegistryProblem(
                        "n_evaluable_exceeds_cohort",
                        (tid,),
                        f"role '{arm.get('role')}' has n_evaluable={n_eval}, above cohort "
                        f"'{cid}' n_c={cohort.n_c}.",
                    )
                )

        named = [c for c in _cohort_ids(data) if c in by_cohort]
        overlapping = sorted(
            {
                tuple(sorted((a, b)))
                for a in named
                for b in by_cohort[a].shares_patients_with
                if b in named
            }
        )
        if overlapping:
            problems.append(
                RegistryProblem(
                    "paired_contrast_as_cross_scenario",
                    (tid,),
                    f"arms name cohorts that declare shared patients: {overlapping}. The "
                    "contrast is then within-patient, and a paired contrast belongs in a "
                    "single CalibrationTarget whose observable declares a reference. Arms of "
                    "a cross-scenario target must be disjoint sets of people, since its "
                    "variance comes from resampling each arm independently.",
                )
            )

    problems.extend(_redundant_arms(targets))
    return problems


def _arm_composition(arm: Dict[str, Any]) -> Optional[Tuple[tuple, tuple]]:
    """An arm's declared composition, in the same shape as a scalar target's."""
    return _composition(arm.get("readout") or {})


def _redundant_arms(targets: Dict[str, Dict[str, Any]]) -> List[RegistryProblem]:
    """An arm duplicating a standalone target on its cohort double-counts that number.

    A cross-scenario term earns a likelihood only when its per-arm constituents are
    deliberately kept out of the fit. Once a constituent is also a target, the fit
    conditions on it and on the contrast, which is the same belief twice.
    """
    scalar: Dict[Tuple[str, tuple], List[str]] = {}
    for tid, data in targets.items():
        cid = data.get("cohort_id")
        if not cid or _arms(data):
            continue
        key = _composition(_readout(data))
        if key:
            scalar.setdefault((cid, key), []).append(tid)

    problems = []
    for tid, data in sorted(targets.items()):
        for arm in _arms(data):
            cid = arm.get("cohort_id")
            key = _arm_composition(arm)
            if not cid or not key:
                continue
            for other in sorted(scalar.get((cid, key), [])):
                problems.append(
                    RegistryProblem(
                        "redundant_cross_scenario_arm",
                        tuple(sorted((tid, other))),
                        f"role '{arm.get('role')}' computes what target '{other}' already "
                        f"reports for cohort '{cid}'. Conditioning on the constituent and on "
                        "the contrast counts one measurement twice; drop one.",
                    )
                )
    return problems


def _duplicate_rows(targets: Dict[str, Dict[str, Any]]) -> List[RegistryProblem]:
    """Two targets computing one model quantity for one cohort are one row reported twice.

    Keyed on the readout's declared composition, so it covers absolute quantities
    as well as ratios.
    """
    seen: Dict[Tuple[str, tuple, tuple, Any], List[str]] = {}
    for tid, data in targets.items():
        cid = data.get("cohort_id")
        comp = _composition(_readout(data))
        if not cid or comp is None:
            continue
        key = (cid, comp[0], comp[1], _observable(data).get("readout_time"))
        seen.setdefault(key, []).append(tid)
    return [
        RegistryProblem(
            "duplicate_row",
            tuple(sorted(members)),
            f"cohort '{key[0]}' has {len(members)} targets computing {list(key[1])} / "
            f"{list(key[2])} at t={key[3]}. One cohort reports a quantity once; a second "
            "target is either a duplicate or belongs to another cohort.",
        )
        for key, members in sorted(seen.items())
        if len(members) > 1
    ]


#: Relative agreement at which two reported fractions count as complementary.
_SUM_TOL = 1e-3


def _reported_center(target: Dict[str, Any]) -> Optional[float]:
    """The location the source printed: a reported median, else a mean, else ``median``."""
    od = _estimates(target).get("observed_distribution") or {}
    stats = od.get("statistics") or []
    for want, p in (("quantile", 0.5), ("mean", None), ("geometric_mean", None)):
        for s in stats:
            if s.get("stat") == want and s.get("p") == p:
                return float(s["value"])
    med = _estimates(target).get("median")
    return float(med[0]) if med else None


def _singular_blocks(targets: Dict[str, Dict[str, Any]]) -> List[RegistryProblem]:
    """Rows of one cohort that are deterministic functions of each other.

    Such a block is singular, and the ridge that keeps it invertible turns into a
    huge direction in its inverse rather than an error. Two cheap detectors: a set
    of fractions whose numerators partition their shared denominator, and a pair
    whose printed centers sum to one.
    """
    fractions: Dict[str, List[Tuple[str, frozenset, frozenset, Optional[float]]]] = {}
    for tid, data in targets.items():
        cid = data.get("cohort_id")
        readout = _readout(data)
        comp = _composition(readout)
        if not cid or comp is None or readout.get("quantity_kind") != "fraction":
            continue
        fractions.setdefault(cid, []).append(
            (tid, frozenset(comp[0]), frozenset(comp[1]), _reported_center(data))
        )

    problems: List[RegistryProblem] = []
    for cid, rows in sorted(fractions.items()):
        problems.extend(_partitioning_rows(cid, rows))
        problems.extend(_centers_summing_to_one(cid, rows))
    return problems


def _partitioning_rows(
    cid: str, rows: List[Tuple[str, frozenset, frozenset, Any]]
) -> List[RegistryProblem]:
    """Fractions over one denominator whose numerators exactly partition it sum to 1."""
    by_den: Dict[frozenset, List[Tuple[str, frozenset]]] = {}
    for tid, num, den, _ in rows:
        by_den.setdefault(den, []).append((tid, num))

    out: List[RegistryProblem] = []
    for den, members in sorted(by_den.items(), key=lambda kv: sorted(t for t, _ in kv[1])):
        if len(members) < 2:
            continue
        union: set = set()
        disjoint = True
        for _, num in members:
            if union & num:
                disjoint = False
                break
            union |= num
        if not disjoint or union != set(den):
            continue
        out.append(
            RegistryProblem(
                "singular_block",
                tuple(sorted(tid for tid, _ in members)),
                f"cohort '{cid}' has {len(members)} fractions over {sorted(den)} whose "
                f"numerators partition it, so they sum to 1 by construction and the block "
                "is singular. Drop one, or widen the denominator so they do not exhaust it.",
            )
        )
    return out


def _centers_summing_to_one(
    cid: str, rows: List[Tuple[str, frozenset, frozenset, Any]]
) -> List[RegistryProblem]:
    """Two printed fractions summing to one are one number reported twice."""
    out: List[RegistryProblem] = []
    centers = sorted((tid, c) for tid, _, _, c in rows if c is not None)
    for i, (tid_a, a) in enumerate(centers):
        for tid_b, b in centers[i + 1 :]:
            total = 1.0 if max(a, b) <= 1.0 else 100.0
            if abs(a + b - total) <= _SUM_TOL * total:
                out.append(
                    RegistryProblem(
                        "singular_block",
                        tuple(sorted((tid_a, tid_b))),
                        f"cohort '{cid}' reports {a} and {b}, which sum to {total}. As data "
                        "these are one number reported twice and the block is singular, even "
                        "where the model observables are not exactly complementary.",
                    )
                )
    return out


def check_registries(targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry) -> None:
    """Raise on any registry defect; warn on unused cohorts and on merged blocks."""
    problems = find_registry_problems(targets, cohorts)
    warn_unused_cohorts(targets, cohorts)
    warn_merged_blocks(targets, cohorts)
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


def covariance_blocks(
    targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry
) -> List[FrozenSet[str]]:
    """Cohorts that must share one covariance block, as connected components.

    Blocks are assumed independent, so two cohorts must sit in one whenever a row
    depends on both. Two ways that happens, and they need opposite draws.

    A target drawing on both joins disjoint sets of people: resample each cohort
    independently, which puts the zeros between arms and the covariance around the
    derived row. Cohorts that share patients are resampled once from the block's
    patient set, each cohort's rows evaluated on its own members; drawing those
    independently would restore the independence the block exists to deny.

    The return keeps the partition and not which relation joined a pair, so a
    caller building V has to recover that from ``shares_patients_with``.
    """
    parent = {c.cohort_id: c.cohort_id for c in cohorts.cohorts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for c in cohorts.cohorts:
        for other in c.shares_patients_with:
            if other in parent:
                union(c.cohort_id, other)

    for data in targets.values():
        named = [c for c in _cohort_ids(data) if c in parent]
        for other in named[1:]:
            union(named[0], other)

    blocks: Dict[str, set] = {}
    for cid in parent:
        blocks.setdefault(find(cid), set()).add(cid)
    return sorted((frozenset(b) for b in blocks.values()), key=lambda b: sorted(b))


def warn_merged_blocks(
    targets: Dict[str, Dict[str, Any]], cohorts: CohortRegistry
) -> List[FrozenSet[str]]:
    """Warn on blocks spanning several cohorts. Returns them."""
    merged = [b for b in covariance_blocks(targets, cohorts) if len(b) > 1]
    for block in merged:
        warnings.warn(
            f"Cohorts {sorted(block)} form one covariance block: a row depends on more than "
            "one of them, so they are not independent. Inference must draw them together.",
            UserWarning,
        )
    return merged


def resolve_n(target: Dict[str, Any], cohorts: CohortRegistry) -> Optional[int]:
    """Patients behind this target's statistics: ``n_evaluable``, else the cohort's ``n_c``."""
    n_eval = _estimates(target).get("n_evaluable")
    if n_eval is not None:
        return int(n_eval)
    cohort = cohorts.get(target.get("cohort_id") or "")
    return cohort.n_c if cohort else None
