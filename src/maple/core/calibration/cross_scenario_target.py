#!/usr/bin/env python3
"""
Pydantic models for Cross-Scenario Calibration Targets.

A CrossScenarioCalibrationTarget composes outputs from multiple per-scenario
simulations into a single derived observable evaluated under one parameter
draw theta. Use cases:

- Counterfactual fold-changes ("iCAF fraction with TGFb blockade vs baseline")
- Treatment-vs-untreated comparisons ("tumor diameter d90 GVAX vs untreated")
- Time-to-event ratios across scenarios

The model is intentionally leaner than CalibrationTarget. It does NOT carry
its own species_dict, denominator audit, or population aggregation block —
those concerns belong to the per-scenario observables that feed this target
as scalar inputs (input_kind='test_statistic') or to the composer code
itself (input_kind='timeseries').

See pdac-build/notes/calibration/cross_scenario_observable_spike.md for the
full design rationale (Option D3+).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maple.core.calibration.calibration_target_models import (
    CalibrationTargetEstimates,
)
from maple.core.calibration.shared_models import SecondarySource, Source


class CrossScenarioInput(BaseModel):
    """
    One input to a cross-scenario observable.

    Each input is keyed by ``role`` (a stable name the observable code
    references, e.g. 'untreated', 'treated', 'numerator') and pulls data
    from one scenario in one of two modes:

    - ``input_kind='test_statistic'``: pull a precomputed scalar from that
      scenario's per-scenario test_stats matrix. Requires
      ``test_statistic_id`` to identify the target. Cheapest path; use
      when a per-scenario CalibrationTarget or PredictionTarget already
      computes the value you need.

    - ``input_kind='timeseries'``: pull the full simulation timeseries
      from that scenario's parquet. Requires ``required_species`` listing
      the model species to expose to the composer. Use when no natural
      per-scenario summary exists (e.g. time-to-event compositions).
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(
        description=(
            "Stable name the composer code uses to look this input up "
            "(e.g. 'untreated', 'treated', 'numerator'). Must be unique "
            "across the inputs list of a single CrossScenarioCalibrationTarget."
        )
    )
    scenario: str = Field(
        description=(
            "Scenario name. Must match one of the scenario keys passed to "
            "the SBI runner (e.g. 'clinical_progression_longterm', "
            "'gvax_neoadjuvant_longterm', 'tgfb_blockade_baseline'). "
            "Cross-scenario derive will fail loudly at runtime if the "
            "scenario isn't present in scenario_meta."
        )
    )
    input_kind: Literal["test_statistic", "timeseries"] = Field(
        description=(
            "How to resolve this input. 'test_statistic' pulls a scalar "
            "from the per-scenario x_raw matrix; 'timeseries' pulls the "
            "full sim_df and exposes a species_dict to the composer."
        )
    )
    test_statistic_id: Optional[str] = Field(
        default=None,
        description=(
            "Required when input_kind='test_statistic'. The "
            "test_statistic_id of a per-scenario CalibrationTarget or "
            "PredictionTarget defined for the scenario referenced above."
        ),
    )
    required_species: Optional[List[str]] = Field(
        default=None,
        description=(
            "Required when input_kind='timeseries'. List of full-model "
            "species the composer accesses (e.g. ['V_T.C1', 'V_T']). "
            "Same format as Observable.species. Resolved via the same "
            "species_plan strategies as compute_test_statistics_batch "
            "(series, param, template, missing)."
        ),
    )

    @model_validator(mode="after")
    def validate_kind_payload(self) -> "CrossScenarioInput":
        if self.input_kind == "test_statistic":
            if not self.test_statistic_id:
                raise ValueError(
                    f"CrossScenarioInput role='{self.role}' has "
                    "input_kind='test_statistic' but no test_statistic_id. "
                    "Provide the per-scenario test_statistic_id this input "
                    "should pull from."
                )
            if self.required_species is not None:
                raise ValueError(
                    f"CrossScenarioInput role='{self.role}' has "
                    "input_kind='test_statistic' but required_species is set. "
                    "required_species is only valid for input_kind='timeseries'."
                )
        elif self.input_kind == "timeseries":
            if not self.required_species:
                raise ValueError(
                    f"CrossScenarioInput role='{self.role}' has "
                    "input_kind='timeseries' but required_species is empty. "
                    "List the model species the composer accesses."
                )
            if self.test_statistic_id is not None:
                raise ValueError(
                    f"CrossScenarioInput role='{self.role}' has "
                    "input_kind='timeseries' but test_statistic_id is set. "
                    "test_statistic_id is only valid for input_kind='test_statistic'."
                )
        return self


class CrossScenarioObservable(BaseModel):
    """
    Composer specification for a cross-scenario observable.

    The composer is a Python function with signature
    ``compute(inputs, ureg) -> Pint Quantity``, where ``inputs`` is a
    role-keyed dict whose value shape depends on each input's ``input_kind``:

    - ``input_kind='test_statistic'``: ``inputs[role]`` is a scalar
      Pint Quantity carrying the per-scenario units declared in that
      target's YAML.
    - ``input_kind='timeseries'``: ``inputs[role]`` is a dict mapping
      ``'time'`` and each entry of ``required_species`` to a Pint
      Quantity (scalar for compartment volumes, array for species
      timeseries), matching the species_dict shape exposed to per-scenario
      observables.

    The composer must return a scalar Pint Quantity with units matching
    ``units`` below.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        description=(
            "Python function source defining "
            "compute(inputs, ureg) -> Pint Quantity. See class docstring "
            "for the inputs dict shape per input_kind.\n\n"
            "Example (scalar fold ratio):\n"
            "def compute(inputs, ureg):\n"
            "    return inputs['treated'] / inputs['untreated']\n\n"
            "Example (timeseries time-to-50% threshold):\n"
            "def compute(inputs, ureg):\n"
            "    import numpy as np\n"
            "    def t_half(scen):\n"
            "        c = scen['V_T.C1'].to('cell').magnitude\n"
            "        t = scen['time'].to('day').magnitude\n"
            "        target = 0.5 * c[0]\n"
            "        below = np.where(c <= target)[0]\n"
            "        return float(t[below[0]]) if len(below) else float('nan')\n"
            "    return (t_half(inputs['treated']) /\n"
            "            t_half(inputs['untreated'])) * ureg.dimensionless"
        )
    )
    units: str = Field(
        description=(
            "Pint-parseable units of the composer output. Must match "
            "empirical_data.units. Typical values: 'dimensionless' (fold "
            "ratios), 'day' (event times), 'cell/mm**2' (density "
            "differences)."
        )
    )
    inputs: List[CrossScenarioInput] = Field(
        min_length=2,
        description=(
            "List of role-keyed inputs the composer consumes. Must have "
            "at least 2 inputs (otherwise it isn't a cross-scenario "
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
    theta, evaluating a Python composer over scalar test-statistic inputs
    and/or full-timeseries inputs. Used for counterfactual fold-changes,
    treatment-vs-untreated comparisons, and other cross-scenario
    observables that don't fit a single-scenario CalibrationTarget.

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
            "(e.g. 'tumor_diameter_d90_gvax_vs_untreated_fold')."
        )
    )

    observable: CrossScenarioObservable = Field(
        description="Composer specification with role-keyed inputs."
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
