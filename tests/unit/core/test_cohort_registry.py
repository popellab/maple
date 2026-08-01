"""Cohort registry, its cross-target checks, and the Observable measurement attributes."""

import warnings

import pytest
import yaml
from pydantic import ValidationError

from maple.core.calibration.cohort import Cohort, CohortRegistry, load_cohorts
from maple.core.calibration.enums import AssayModality, QuantityKind
from maple.core.calibration.observable import Observable
from maple.core.calibration.registry_audit import (
    check_registries,
    find_registry_problems,
    resolve_n,
    warn_unused_cohorts,
)

_CODE = (
    "def compute_observable(time, species_dict, constants):\n"
    "    return species_dict['V_T.CD8'] / species_dict['V_T.nucleated']\n"
)


def _cohort(**over):
    base = dict(
        cohort_id="li2022_arm_a",
        description="Arm A patients, paired pre/post biopsy.",
        scenarios=["baseline_no_treatment"],
        n_c=9,
        source_tag="Li2022_CancerCell_PDAC_AntiPD1",
    )
    base.update(over)
    return Cohort(**base)


def _registry(**over):
    return CohortRegistry(cohorts=[_cohort(**over)])


def _observable(**over):
    base = dict(
        quantity_kind=QuantityKind.FRACTION,
        assay_modality=AssayModality.MIHC,
        code=_CODE,
        units="dimensionless",
        species=["V_T.CD8", "V_T.nucleated"],
        support="unit_interval",
        readout_time=0.0,
        readout_time_unit="day",
    )
    base.update(over)
    return Observable(**base)


def _target(**over):
    base = dict(
        cohort_id="li2022_arm_a",
        epistemic_basis="literature",
        observable={"code": _CODE, "readout_time": 0.0},
        empirical_data={"inputs": [{"source_ref": "Li2022_CancerCell_PDAC_AntiPD1"}]},
        primary_data_source={"source_tag": "Li2022_CancerCell_PDAC_AntiPD1"},
    )
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# Cohort                                                                       #
# --------------------------------------------------------------------------- #
class TestCohort:
    def test_minimal_cohort_validates(self):
        assert _cohort().n_c == 9

    def test_no_scenarios_rejected(self):
        with pytest.raises(ValueError, match="declares no scenarios"):
            _cohort(scenarios=[])

    def test_repeated_scenario_rejected(self):
        with pytest.raises(ValueError, match="repeats a scenario"):
            _cohort(scenarios=["a", "a"])

    def test_self_overlap_rejected(self):
        with pytest.raises(ValueError, match="lists itself"):
            _cohort(shares_patients_with=["li2022_arm_a"])

    def test_zero_n_rejected(self):
        with pytest.raises(ValueError):
            _cohort(n_c=0)

    def test_source_tag_is_singular(self):
        """A cohort is one study's patients, so pooling is unrepresentable."""
        with pytest.raises(ValueError):
            _cohort(source_tag=["a", "b"])

    def test_eligibility_needs_a_bound(self):
        with pytest.raises(ValueError, match="neither lo nor hi"):
            _cohort(
                eligibility=[
                    {"target_id": "cd8_fraction", "units": "dimensionless", "rationale": "x"}
                ]
            )

    def test_eligibility_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="selects nobody"):
            _cohort(
                eligibility=[
                    {
                        "target_id": "cd8_fraction",
                        "lo": 0.5,
                        "hi": 0.1,
                        "units": "dimensionless",
                        "rationale": "x",
                    }
                ]
            )


class TestCohortRegistry:
    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate cohort_id"):
            CohortRegistry(cohorts=[_cohort(), _cohort()])

    def test_overlap_must_resolve(self):
        with pytest.raises(ValueError, match="not in the registry"):
            CohortRegistry(cohorts=[_cohort(shares_patients_with=["ghost"])])

    def test_overlap_must_be_symmetric(self):
        a = _cohort(cohort_id="a", shares_patients_with=["b"])
        b = _cohort(cohort_id="b")
        with pytest.raises(ValueError, match="does not declare it back"):
            CohortRegistry(cohorts=[a, b])

    def test_symmetric_overlap_accepted(self):
        a = _cohort(cohort_id="a", shares_patients_with=["b"])
        b = _cohort(cohort_id="b", shares_patients_with=["a"])
        assert len(CohortRegistry(cohorts=[a, b]).cohorts) == 2

    def test_load_from_yaml(self, tmp_path):
        p = tmp_path / "cohorts.yaml"
        p.write_text(yaml.safe_dump({"cohorts": [_cohort().model_dump()]}))
        assert load_cohorts(p).get("li2022_arm_a").n_c == 9

    def test_load_bare_list(self, tmp_path):
        p = tmp_path / "cohorts.yaml"
        p.write_text(yaml.safe_dump([_cohort().model_dump()]))
        assert len(load_cohorts(p).cohorts) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cohorts(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------- #
# Observable measurement attributes                                            #
# --------------------------------------------------------------------------- #
class TestObservableMeasurementAttributes:
    """The attributes inference reads to build the measurement-discrepancy design.

    They live on the observable rather than in a registry: two observables
    agreeing on kind and modality already share a design row, so a shared
    identifier would add nothing.
    """

    def test_minimal_observable_validates(self):
        assert _observable().quantity_kind == QuantityKind.FRACTION

    def test_quantity_kind_is_required(self):
        with pytest.raises(ValidationError):
            _observable(quantity_kind=None)

    def test_assay_modality_is_required(self):
        with pytest.raises(ValidationError):
            _observable(assay_modality=None)

    def test_unregistered_quantity_kind_rejected(self):
        with pytest.raises(ValidationError):
            _observable(quantity_kind="luminosity")

    def test_unregistered_modality_rejected(self):
        with pytest.raises(ValidationError):
            _observable(assay_modality="vibes")

    def test_foldchange_requires_a_reference(self):
        with pytest.raises(ValidationError, match="no reference is declared"):
            _observable(quantity_kind=QuantityKind.FOLDCHANGE)

    def test_foldchange_with_a_timepoint_reference(self):
        obs = _observable(
            quantity_kind=QuantityKind.FOLDCHANGE,
            reference={"kind": "timepoint", "timepoint": 0.0, "timepoint_unit": "day"},
        )
        assert obs.reference.timepoint == 0.0

    def test_foldchange_with_a_scenario_reference(self):
        obs = _observable(
            quantity_kind=QuantityKind.FOLDCHANGE,
            reference={"kind": "scenario", "scenario": "baseline_no_treatment"},
        )
        assert obs.reference.scenario == "baseline_no_treatment"

    def test_absolute_quantity_may_not_declare_a_reference(self):
        with pytest.raises(ValidationError, match="absolute quantity"):
            _observable(reference={"kind": "timepoint", "timepoint": 0.0, "timepoint_unit": "day"})

    def test_timepoint_reference_needs_units(self):
        with pytest.raises(ValidationError, match="needs timepoint and timepoint_unit"):
            _observable(
                quantity_kind=QuantityKind.FOLDCHANGE,
                reference={"kind": "timepoint", "timepoint": 0.0},
            )

    def test_scenario_reference_rejects_a_timepoint(self):
        with pytest.raises(ValidationError, match="must not set timepoint"):
            _observable(
                quantity_kind=QuantityKind.FOLDCHANGE,
                reference={"kind": "scenario", "scenario": "baseline", "timepoint": 1.0},
            )

    def test_same_kind_and_modality_share_a_design_row(self):
        """No identifier needed: the attributes are the row."""
        a = _observable()
        b = _observable(code=_CODE.replace("V_T.CD8", "V_T.Treg"), species=["V_T.Treg"])
        assert (a.quantity_kind, a.assay_modality) == (b.quantity_kind, b.assay_modality)


# --------------------------------------------------------------------------- #
# Cross-target checks                                                          #
# --------------------------------------------------------------------------- #
class TestRegistryAudit:
    def test_clean_corpus_passes(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_registries({"t1": _target()}, _registry())

    def test_unknown_cohort_reported(self):
        problems = find_registry_problems({"t1": _target(cohort_id="ghost")}, _registry())
        assert [p.kind for p in problems] == ["unknown_cohort"]

    def test_pooled_target_rejected(self):
        """Several sources behind one target is a meta-analysis, not a cohort."""
        t = _target(
            empirical_data={
                "inputs": [
                    {"source_ref": "Golesworthy2022"},
                    {"source_ref": "Liu2015"},
                    {"source_ref": "Jansen2021"},
                ]
            }
        )
        pooled = [
            p for p in find_registry_problems({"t1": t}, _registry()) if p.kind == "pooled_target"
        ]
        assert len(pooled) == 1
        assert "Split into one target per source" in pooled[0].detail

    def test_mechanistic_target_exempt_from_pooling_and_cohort(self):
        t = _target(cohort_id=None, epistemic_basis="mechanistic")
        assert find_registry_problems({"t1": t}, _registry()) == []

    def test_n_evaluable_may_not_exceed_cohort(self):
        t = _target()
        t["empirical_data"]["n_evaluable"] = 99
        kinds = {p.kind for p in find_registry_problems({"t1": t}, _registry())}
        assert "n_evaluable_exceeds_cohort" in kinds

    def test_source_disagreeing_with_cohort_reported(self):
        t = _target(primary_data_source={"source_tag": "Li2022_CancerCell"})
        kinds = {p.kind for p in find_registry_problems({"t1": t}, _registry())}
        assert "source_disagrees_with_cohort" in kinds

    def test_duplicate_row_reported(self):
        """One cohort reports a quantity once."""
        dup = [
            p
            for p in find_registry_problems({"t1": _target(), "t2": _target()}, _registry())
            if p.kind == "duplicate_row"
        ]
        assert len(dup) == 1
        assert dup[0].target_ids == ("t1", "t2")

    def test_duplicate_row_keys_on_the_computed_expression(self):
        """Not on any declared id, so it cannot go stale against the code that runs."""
        t2 = _target()
        t2["observable"] = dict(
            t2["observable"],
            code=_CODE.replace("V_T.CD8", "V_T.CD4"),
        )
        problems = find_registry_problems({"t1": _target(), "t2": t2}, _registry())
        assert [p.kind for p in problems] == []

    def test_same_expression_in_different_cohorts_is_fine(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="other")])
        problems = find_registry_problems(
            {"t1": _target(), "t2": _target(cohort_id="other")}, cohorts
        )
        assert [p.kind for p in problems] == []

    def test_check_raises_with_every_problem(self):
        with pytest.raises(ValueError, match="registry problem"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                check_registries({"t1": _target(cohort_id="ghost")}, _registry())

    def test_unused_cohorts_warn(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="unused")])
        with pytest.warns(UserWarning, match="no target uses"):
            unused = warn_unused_cohorts({"t1": _target()}, cohorts)
        assert unused == ["unused"]


class TestResolveN:
    def test_falls_back_to_cohort_n(self):
        assert resolve_n(_target(), _registry()) == 9

    def test_n_evaluable_wins(self):
        t = _target()
        t["empirical_data"]["n_evaluable"] = 6
        assert resolve_n(t, _registry()) == 6

    def test_unknown_cohort_gives_none(self):
        assert resolve_n(_target(cohort_id="ghost"), _registry()) is None
