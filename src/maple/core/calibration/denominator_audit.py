"""Cross-target denominator checks.

Every other calibration validator is single-target: pydantic hands a model
validator one object, so it can only ask questions about that object. Some
defects are not properties of a target at all — they are properties of a *pair*.
This module holds the checks that need the whole loaded set, and is called from
:func:`maple.core.calibration.test_stats_loader.load_calibration_targets`.

The motivating case (pdac, July 2026). Two targets computed the identical model
expression, ``Treg / (Treg + Th + Th_exh)``, and asserted different values for
it — 0.50 from a panel that counted Tregs against *polarised* CD4 only, 0.34
from one that counted against *all* CD4. Both declared the denominator audit
correctly. Both wrote the contradiction out in English, in adjacent fields:

    treg_fraction_cd4        "...(Treg + Th1 + Th2 + Th17), excluding Th0 bystanders"
    treg_fraction_hiraoka    "CD4+ tumor-infiltrating T lymphocytes..."

One says *excluding*, the other says *includes*. Nothing compared them, so both
entered the likelihood at full weight and became the top two misfit drivers —
57% of joint influence off 15% of the observables. The extraction agent did its
job; the pipeline had no place to notice.

Note that the CIs of that pair *overlap*, so a value-agreement check would not
have caught it. The signal is the denominator prose, not the numbers.
"""

from __future__ import annotations

import ast
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# --------------------------------------------------------------------------- #
# Reading the observable's numerator and denominator out of its code           #
# --------------------------------------------------------------------------- #
def _species_reached(node: ast.AST, env: Dict[str, Set[str]]) -> Set[str]:
    """Species-dict keys reachable from ``node``, resolving local variables."""
    found: Set[str] = set()
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "species_dict"
            and isinstance(sub.slice, ast.Constant)
        ):
            found.add(sub.slice.value)
        elif isinstance(sub, ast.Name) and sub.id in env:
            found |= env[sub.id]
    return found


def numerator_and_denominator(code: str) -> Tuple[Set[str], Set[str]]:
    """Species on each side of the division(s) in an ``observable.code`` body.

    Returns ``(numerator, denominator)``. Both are empty when the code does not
    divide, or fails to parse (syntax is reported by the schema validators).

    Real observables bind intermediates first — ``total_t = treg + cd8 + th``,
    then ``treg / total_t`` — so a single forward pass records what each local
    name reaches before the divisions are read. Names are resolved
    transitively, which covers the chained case.

    An observable that divides more than once (a ratio of two fractions) unions
    both sides; that is coarse but conservative, and no live target does it.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()

    env: Dict[str, Set[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            env[node.targets[0].id] = _species_reached(node.value, env)

    numerator: Set[str] = set()
    denominator: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator |= _species_reached(node.left, env)
            denominator |= _species_reached(node.right, env)
    return numerator, denominator


# --------------------------------------------------------------------------- #
# A — cross-target mapping collisions                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MappingCollision:
    """Two or more targets reducing to one model quantity."""

    numerator: Tuple[str, ...]
    denominator: Tuple[str, ...]
    readout_time: Optional[float]
    members: Tuple[str, ...]
    experimental_denominators: Tuple[str, ...]
    justified_by: Tuple[str, ...] = field(default=())

    @property
    def is_justified(self) -> bool:
        return bool(self.justified_by)


def _mapping_key(target: Dict[str, Any]) -> Optional[tuple]:
    observable = target.get("observable") or {}
    readout = observable.get("readout") or {}
    denominator = readout.get("denominator_species") or []
    if not denominator:
        return None
    return (
        tuple(sorted(readout.get("numerator_species") or [])),
        tuple(sorted(denominator)),
        observable.get("readout_time"),
    )


def find_mapping_collisions(targets: Dict[str, Dict[str, Any]]) -> List[MappingCollision]:
    """Group ``{target_id: parsed_yaml}`` by model quantity; return the groups of >1.

    The key is (numerator species, denominator species, readout time) as declared
    on the readout; ``check_code_matches_readout`` separately holds the code to that
    declaration. Including the numerator matters: nine density targets divide by
    ``V_T`` and are not in conflict, because they count different cells.

    Callers pass one scenario at a time. Across scenarios the same expression is
    expected (a baseline and a day-21 arm), and is not a collision.
    """
    groups: Dict[tuple, List[str]] = {}
    for target_id, data in targets.items():
        key = _mapping_key(data)
        if key is None:
            continue
        groups.setdefault(key, []).append(target_id)

    collisions = []
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members = sorted(members)
        observables = [(targets[m].get("observable") or {}) for m in members]
        collisions.append(
            MappingCollision(
                numerator=key[0],
                denominator=key[1],
                readout_time=key[2],
                members=tuple(members),
                experimental_denominators=tuple(
                    (o.get("readout") or {}).get("experimental_denominator") or ""
                    for o in observables
                ),
                justified_by=tuple(
                    m
                    for m, o in zip(members, observables)
                    if (o.get("duplicate_mapping_justification") or "").strip()
                ),
            )
        )
    return collisions


def check_mapping_collisions(targets: Dict[str, Dict[str, Any]]) -> None:
    """Raise when two targets compute one model quantity without saying why.

    No similarity threshold. Prose is too weak a discriminator to tune on: in
    the pdac corpus two targets phrasing the SAME denominator differently
    ("mm^2 of tumor tissue section" vs "mm^2 of intratumoral tumor section area,
    pooled across the three cohorts") score 0.38 on token overlap, while the
    genuinely contradictory Treg pair scores 0.06. Any cut between those is
    fitted to one example. So a collision is always surfaced, and the author
    declares which kind it is via ``duplicate_mapping_justification``.

    That is the right default anyway: two targets reducing to one quantity is a
    claim worth stating explicitly, whether or not it is sound.
    """
    unjustified = [c for c in find_mapping_collisions(targets) if not c.is_justified]
    if not unjustified:
        return

    blocks = []
    for c in unjustified:
        lines = [
            f"  {' + '.join(c.numerator) or '<none>'} / {' + '.join(c.denominator)}"
            + (f"  at t={c.readout_time}" if c.readout_time is not None else ""),
        ]
        for member, exp in zip(c.members, c.experimental_denominators):
            lines.append(f"    - {member}")
            lines.append(f"        experimental_denominator: {exp or '<not declared>'}")
        blocks.append("\n".join(lines))

    raise ValueError(
        "Calibration targets compute the same model quantity without declaring why:\n\n"
        + "\n\n".join(blocks)
        + "\n\nEach group reduces to ONE model expression, so its members make one "
        "claim several times over and each enters the likelihood at full weight. "
        "Compare the experimental_denominator lines above:\n\n"
        "  - If they describe DIFFERENT experimental quantities, one of the model "
        "mappings is wrong. Re-derive it against a denominator that matches what "
        "the paper actually measured. This is the real defect the check exists "
        "for — the pdac Treg pair mapped 'Tregs among polarised CD4' and 'Tregs "
        "among all CD4' onto the same expression, and the disagreement showed up "
        "as the top two misfit drivers instead of as a extraction bug.\n"
        "  - If they describe the SAME quantity in different cohorts, set "
        "observable.duplicate_mapping_justification on at least one of them, "
        "naming the cohorts. Then check the values are mutually compatible; a "
        "disagreement beyond the CIs is a cross-study conflict to adjudicate, "
        "not something to average away."
    )


# --------------------------------------------------------------------------- #
# C — a declared denominator bias should be heard                              #
# --------------------------------------------------------------------------- #
#: A declaration opening with "None" / "N/A" asserts there is no bias — the field
#: docstring offers exactly that ("None (denominator fully captured by model
#: species)"), so it is an answer, not an omission.
_NO_BIAS = re.compile(r"^\s*(none|n/?a)\b", re.I)

#: A usable declaration says which way the bias runs, or how big it is. Prose is
#: matched loosely on purpose: this decides whether to *mention* a target, never
#: whether to reject one.
_DIRECTION_OR_MAGNITUDE = re.compile(
    r"""(
        \d+\s*[-–—]\s*\d+\s*(x|%|-?fold)     # a range: "50-70%", "2-3x"
      | \d+(\.\d+)?\s*(x|%|-?fold)\b         # a multiplier: "3x", "10-fold"
      | \b(higher|lower)\b
      | \b(over|under)-?(predict|estimat)    # "overpredict", "under-estimate"
      | \binflat                             # "inflates the denominator"
      | \bbias(es|ed)?\s+(up|down)
    )""",
    re.I | re.X,
)


@dataclass(frozen=True)
class DeclaredDenominatorBias:
    target_id: str
    experimental_denominator: str
    unmodeled_components: str
    states_direction_or_magnitude: bool


def collect_declared_biases(
    targets: Dict[str, Dict[str, Any]],
) -> List[DeclaredDenominatorBias]:
    """Targets whose own ``unmodeled_denominator_components`` admits a bias.

    Declarations that open with "None" are skipped — they assert the denominator
    is fully captured, which is an answer rather than an omission.
    """
    out = []
    for target_id in sorted(targets):
        observable = targets[target_id].get("observable") or {}
        unmodeled = (observable.get("unmodeled_denominator_components") or "").strip()
        if not unmodeled or _NO_BIAS.match(unmodeled):
            continue
        out.append(
            DeclaredDenominatorBias(
                target_id=target_id,
                experimental_denominator=(
                    (observable.get("readout") or {}).get("experimental_denominator") or ""
                ).strip(),
                unmodeled_components=unmodeled,
                states_direction_or_magnitude=bool(_DIRECTION_OR_MAGNITUDE.search(unmodeled)),
            )
        )
    return out


def warn_declared_biases(targets: Dict[str, Dict[str, Any]]) -> List[DeclaredDenominatorBias]:
    """Surface declared denominator biases once per load; return them for diagnostics.

    ``unmodeled_denominator_components`` asks the author to document "expected
    direction and magnitude of systematic bias", and authors fill it in. It then
    had no consequence anywhere: the pdac Hiraoka target declared that its
    experimental denominator includes bystander CD4+ cells the model does not
    represent, and still entered the likelihood at full weight, unweighted and
    unflagged. A field only a reader can act on is not a control.

    **One** warning, not one per target. In the pdac corpus 32 of 52 targets
    declare a bias; a per-target warning would be 32 lines on every load and
    would be tuned out within a week.

    Deliberately not an error, on two counts. Whether a declared bias should
    downweight a target or gate it out of the joint is an inference-side policy
    decision, not a schema one — hence the returned list, so the caller can act.
    And measured against the live corpus, only 2 of 32 declarations state a
    direction or magnitude, so requiring one would block 30 targets. That is a
    real quality gap and a fair thing to enforce later, but it is a corpus
    migration, not a validator flip.
    """
    biases = collect_declared_biases(targets)
    if not biases:
        return biases

    vague = [b for b in biases if not b.states_direction_or_magnitude]
    message = (
        f"{len(biases)} calibration target(s) declare an unmodeled denominator "
        "component. Each enters the likelihood at full weight regardless — decide "
        "whether they should be downweighted or reconciled first."
    )
    if vague:
        listing = "\n".join(f"    - {b.target_id}" for b in vague)
        message += (
            f"\n\n{len(vague)} of them name the missing components but never state the "
            "expected DIRECTION or MAGNITUDE of the bias, which is what "
            "unmodeled_denominator_components asks for. Without it the declaration "
            "cannot be acted on, only read:\n"
            f"{listing}"
        )
    warnings.warn(message, UserWarning)
    return biases


@dataclass(frozen=True)
class CodeReadoutMismatch:
    """A target whose code divides by something other than its declared readout."""

    target_id: str
    declared: Tuple[str, ...]
    in_code: Tuple[str, ...]


def find_code_readout_mismatches(
    targets: Dict[str, Dict[str, Any]],
) -> List[CodeReadoutMismatch]:
    """Targets whose ``observable.code`` disagrees with ``readout.denominator_species``.

    The declaration carries the row's identity, so the code has to compute what it
    says. Only checked where the code divides: an observable that reaches its
    denominator some other way (an area built from a constant) is not comparable
    this way.

    A sum the model defines as an aggregate is named by that aggregate on both
    sides, so this is a plain comparison. Listing the members instead is what
    goes stale when a pool joins the rule, and is what this then reports.
    """
    out = []
    for tid, data in sorted(targets.items()):
        observable = data.get("observable") or {}
        declared = set((observable.get("readout") or {}).get("denominator_species") or [])
        _, in_code = numerator_and_denominator(observable.get("code") or "")
        if not declared or not in_code or declared == in_code:
            continue
        out.append(CodeReadoutMismatch(tid, tuple(sorted(declared)), tuple(sorted(in_code))))
    return out


def warn_code_readout_mismatches(
    targets: Dict[str, Dict[str, Any]],
) -> List[CodeReadoutMismatch]:
    """Warn on each mismatch. Returns them."""
    found = find_code_readout_mismatches(targets)
    for m in found:
        warnings.warn(
            f"{m.target_id}: readout.denominator_species={list(m.declared)} but the code "
            f"divides by {list(m.in_code)}. Name the model aggregate that defines the sum, "
            "on both sides.",
            UserWarning,
        )
    return found


__all__ = [
    "numerator_and_denominator",
    "CodeReadoutMismatch",
    "find_code_readout_mismatches",
    "warn_code_readout_mismatches",
    "MappingCollision",
    "find_mapping_collisions",
    "check_mapping_collisions",
    "DeclaredDenominatorBias",
    "collect_declared_biases",
    "warn_declared_biases",
]
