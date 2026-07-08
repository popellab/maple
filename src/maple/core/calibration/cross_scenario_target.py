#!/usr/bin/env python3
"""
Pydantic models for Cross-Scenario Calibration Targets.

A CrossScenarioCalibrationTarget composes outputs from multiple per-scenario
simulations into a single derived observable evaluated under one parameter
draw theta. Use cases:

- Counterfactual fold-changes ("iCAF fraction with TGFb blockade vs baseline")
- Treatment-vs-untreated comparisons ("tumor diameter d90 GVAX vs untreated")
- Cross-arm invariants ("within-TLA CD8 number is unchanged by urelumab")

The model is intentionally leaner than CalibrationTarget. It does NOT carry
its own species_dict, denominator audit, or population aggregation block.

Design — why each input owns its observable
--------------------------------------------
A cross-scenario target is only worth a likelihood term when its constituent
per-arm quantities are NOT already columns in the joint-NPE training x. If they
were, the NPE already conditions on them and on every relationship between them
(conditioning on two marginals already pins their ratio) — a cross-scenario
likelihood term would be redundant, or worse double-count a belief already in
the fit. The legitimate case is the opposite: a per-arm quantity we deliberately
keep OUT of x (e.g. an absolute within-TLA CD8 count whose unmodeled denominator
only cancels in a ratio), composed across arms so only the invariant constrains
the fit.

So each input defines its OWN per-arm observable (``observable_code`` +
``required_species``) rather than pointing at a per-scenario calibration target.
The target is self-contained: it owns both the per-arm extraction and the
cross-arm reduction, in one YAML, with no reference to any other target and
without entering the NPE x as a standalone constraint. The per-arm scalar is
computed at derive time on the worker (while the trajectory is still in hand,
before ``discard_trajectories`` drops it); the cross-arm reduction runs
downstream once every scenario's matrix is gathered and aligned on
``sample_index``.

See pdac-build/notes/calibration/cross_scenario_derive_time_inputs.md for the
full design rationale (this single self-contained input kind replaced the older
``test_statistic`` / ``timeseries`` split).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maple.core.calibration.calibration_target_models import (
    CalibrationTargetEstimates,
)
from maple.core.calibration.shared_models import SecondarySource, Source


class CrossScenarioInput(BaseModel):
    """
    One per-arm input to a cross-scenario observable.

    Each input is keyed by ``role`` (a stable name the reduction code
    references, e.g. 'numerator', 'denominator', 'treated', 'untreated')
    and defines a self-contained per-scenario observable: a Python function
    that maps one scenario's simulated trajectory to a single scalar.

    The observable is evaluated at derive time on the HPC worker, in the same
    row-group pass that derives the regular per-scenario test stats, using the
    same ``build_test_stat_registry`` / ``compute_test_statistics_batch``
    machinery. So ``observable_code`` must match the per-scenario test-stat
    contract exactly:

        def compute_test_statistic(time, species_dict) -> float

    where ``time`` is a numpy array of time points (days) and ``species_dict``
    maps each entry of ``required_species`` to a raw float or numpy array in its
    canonical model_structure.json units (no Pint). The function does its own
    time selection (e.g. interpolate to day 21) and returns one scalar.

    The returned raw scalar is handed straight to the reduction in
    ``CrossScenarioObservable.code`` (the runtime path is pintless — the
    reduction does any unit arithmetic numerically inline, exactly like a
    per-scenario test stat). The physical meaning of the scalar is documented
    by the model species it derives from; the only declared unit is the
    target-level ``CrossScenarioObservable.units`` on the reduction output.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(
        description=(
            "Stable name the reduction code uses to look this input up "
            "(e.g. 'numerator', 'denominator', 'treated', 'untreated'). Must "
            "be unique across the inputs list of a single "
            "CrossScenarioCalibrationTarget."
        )
    )
    scenario: str = Field(
        description=(
            "Scenario name. Must match one of the scenario keys passed to "
            "the SBI runner (e.g. 'gvax_nivo_neoadjuvant_zheng2022', "
            "'gvax_nivo_urelumab_neoadjuvant_heumann2023'). Cross-scenario "
            "derive will fail loudly at runtime if the scenario isn't present "
            "in scenario_meta."
        )
    )
    observable_code: str = Field(
        description=(
            "Python source defining "
            "compute_test_statistic(time, species_dict) -> float — the same "
            "contract as a per-scenario test stat. Evaluated at derive time on "
            "the worker against this scenario's trajectory. Must do its own "
            "time selection and return a single scalar.\n\n"
            "Example (within-TLA CD8 number at day 21):\n"
            "def compute_test_statistic(time, species_dict):\n"
            "    import numpy as np\n"
            "    t = np.asarray(time, dtype=float)\n"
            "    n = (np.asarray(species_dict['V_T.CD8_TLA'], dtype=float)\n"
            "         + np.asarray(species_dict['V_T.CD8_TLA_act'], dtype=float))\n"
            "    return float(np.interp(21.0, t, n))"
        )
    )
    required_species: List[str] = Field(
        min_length=1,
        description=(
            "Model species / compartments / parameters the observable_code "
            "accesses (e.g. ['V_T.CD8_TLA', 'V_T.CD8_TLA_act']). Same format "
            "and resolution strategies as a per-scenario test stat's "
            "required_species (series / param / template / missing)."
        ),
    )


class CrossScenarioObservable(BaseModel):
    """
    Composer specification for a cross-scenario observable.

    The composer is a pure reduction over the per-arm scalars produced by each
    input. Its signature is ``compute(inputs) -> float`` (the runtime path is
    pintless), where ``inputs`` is a role-keyed dict and each ``inputs[role]``
    is a raw float in the per-arm observable's canonical units. The composer
    returns one raw float; any unit arithmetic is done numerically inline.

    Because every input is a scalar, the reduction never touches raw species or
    timeseries — that extraction lives entirely in each input's
    ``observable_code``.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        description=(
            "Python function source defining "
            "compute(inputs) -> float. Pure reduction over the role-keyed "
            "per-arm scalars; does not touch raw species. Pintless — do any "
            "unit arithmetic numerically inline.\n\n"
            "Example (cross-arm invariance ratio):\n"
            "def compute(inputs):\n"
            "    return inputs['urelumab'] / inputs['nivo']"
        )
    )
    units: str = Field(
        description=(
            "Label for the composer output's units. Must match "
            "empirical_data.units (validated). Used for documentation and the "
            "units-consistency check only — NOT parsed by the pintless runtime. "
            "Typical values: 'dimensionless' (fold ratios, invariants), 'day' "
            "(event times), 'cell/mm**2' (density differences)."
        )
    )
    inputs: List[CrossScenarioInput] = Field(
        min_length=2,
        description=(
            "List of role-keyed per-arm inputs the reduction consumes. Must "
            "have at least 2 inputs (otherwise it isn't a cross-scenario "
            "observable). Roles must be unique."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_roles(self) -> "CrossScenarioObservable":
        roles = [inp.role for inp in self.inputs]
        if len(roles) != len(set(roles)):
            seen = set()
            dupes = []
            for r in roles:
                if r in seen:
                    dupes.append(r)
                seen.add(r)
            raise ValueError(
                f"CrossScenarioObservable roles must be unique; "
                f"duplicates: {sorted(set(dupes))}"
            )
        return self


class CrossScenarioCalibrationTarget(BaseModel):
    """
    A calibration target derived from multiple per-scenario simulations.

    Composes outputs across N>=2 scenarios on the same parameter draw
    theta, evaluating a per-arm observable per scenario and a Python
    reduction over the resulting scalars. Used for counterfactual
    fold-changes, treatment-vs-untreated comparisons, and cross-arm
    invariants that don't fit a single-scenario CalibrationTarget — and
    specifically for relationships whose per-arm constituents are kept out
    of the joint-NPE training x on purpose.

    The empirical_data / epistemic_basis / provenance fields mirror
    CalibrationTarget. Most cross-scenario targets are
    epistemic_basis='mechanistic' with deliberately wide CI95 (no
    primary publication directly reports a counterfactual fold-change),
    but literature targets are supported when a paper reports a
    measurement that maps cleanly onto a cross-scenario composition.
    """

    model_config = ConfigDict(extra="forbid")

    cross_scenario_target_id: str = Field(
        description=(
            "Unique target identifier. Convention: "
            "'<observable>_<scen_a>_vs_<scen_b>_<role_descriptor>' "
            "(e.g. 'cd8_intla_number_invariance_nivo_vs_urelumab')."
        )
    )

    observable: CrossScenarioObservable = Field(
        description="Composer specification with role-keyed per-arm inputs."
    )

    empirical_data: CalibrationTargetEstimates = Field(
        description=(
            "Same shape as CalibrationTarget.empirical_data. For most "
            "cross-scenario targets this is a single-element distribution "
            "summarizing the literature claim or mechanistic prior."
        )
    )

    # --- Narrative ---
    study_interpretation: str = Field(
        description=(
            "Why this cross-scenario observable matters for calibration "
            "and what direction/magnitude the literature or mechanistic "
            "reasoning supports. Cross-scenario equivalent of "
            "CalibrationTarget.study_interpretation."
        )
    )
    key_assumptions: List[str] = Field(
        min_length=1,
        description=(
            "List of key assumptions. For mechanistic targets, MUST "
            "include the rationale for the chosen distribution width and "
            "the biological reasoning behind the directional claim."
        ),
    )
    key_study_limitations: List[str] = Field(
        default_factory=list,
        description="Optional list of limitations.",
    )

    # --- Epistemic basis ---
    epistemic_basis: Literal["literature", "mechanistic"] = Field(
        default="mechanistic",
        description=(
            "Default 'mechanistic' for cross-scenario targets — most "
            "counterfactual claims are qualitative literature inequalities "
            "translated to vague distributions. Use 'literature' only when "
            "a paper reports a quantitative cross-scenario measurement "
            "(e.g. ratio of post-treatment to pretreatment tumor diameter)."
        ),
    )

    # --- Sources ---
    primary_data_source: Optional[Source] = Field(
        default=None,
        description=(
            "Required when epistemic_basis='literature'. May be null for " "mechanistic targets."
        ),
    )
    secondary_data_sources: List[SecondarySource] = Field(
        default_factory=list,
        description="Secondary sources (constants, reference values).",
    )

    # --- Footer / metadata ---
    cancer_type: Optional[str] = Field(default=None, description="Cancer type (e.g. 'PDAC').")
    tags: List[str] = Field(default_factory=list, description="Metadata tags.")
    derivation_id: Optional[str] = Field(default=None)
    derivation_timestamp: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_epistemic_basis_consistency(self) -> "CrossScenarioCalibrationTarget":
        if self.epistemic_basis == "literature" and self.primary_data_source is None:
            raise ValueError(
                "epistemic_basis='literature' requires primary_data_source. "
                "If this target encodes a soft prior from biological "
                "reasoning rather than a published measurement, set "
                "epistemic_basis='mechanistic' and document the rationale "
                "in key_assumptions."
            )
        return self

    @model_validator(mode="after")
    def validate_units_match_empirical(self) -> "CrossScenarioCalibrationTarget":
        if self.observable.units != self.empirical_data.units:
            raise ValueError(
                f"observable.units ('{self.observable.units}') must match "
                f"empirical_data.units ('{self.empirical_data.units}')."
            )
        return self
