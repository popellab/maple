"""Cohort / readout registries and the cross-target checks against them."""

import warnings

import pytest
import yaml

from maple.core.calibration.cohort import Cohort, CohortRegistry, load_cohorts
from maple.core.calibration.readout import ReadoutRegistry, load_readouts
from maple.core.calibration.registry_audit import (
    check_registries,
    find_registry_problems,
    resolve_n,
    warn_unused_registry_entries,
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


def _registries(cohort_over=None, readout_over=None):
    cohorts = CohortRegistry(cohorts=[_cohort(**(cohort_over or {}))])
    readouts = ReadoutRegistry.model_validate(
        {
            "quantity_kinds": [
                {"id": "fraction", "description": "Fraction of a denominator."},
                {
                    "id": "foldchange",
                    "description": "Relative to a reference.",
                    "requires_reference": True,
                },
            ],
            "assay_modalities": [{"id": "mihc", "description": "Multiplex IHC."}],
            "readouts": [
                {
                    "readout_id": "cd8_fraction",
                    "description": "CD8 over nucleated cells.",
                    "quantity_kind": "fraction",
                    "assay_modality": "mihc",
                    "units": "dimensionless",
                    "numerator_species": ["V_T.CD8"],
                    "denominator_species": ["V_T.nucleated"],
                    **(readout_over or {}),
                }
            ],
        }
    )
    return cohorts, readouts


def _target(**over):
    base = dict(
        cohort_id="li2022_arm_a",
        epistemic_basis="literature",
        observable={
            "readout_id": "cd8_fraction",
            "code": "def f(t, species_dict, c):\n    return species_dict['V_T.CD8'] / species_dict['V_T.nucleated']\n",
        },
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
                    {"readout_id": "cd8_fraction", "units": "dimensionless", "rationale": "x"}
                ]
            )

    def test_eligibility_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="selects nobody"):
            _cohort(
                eligibility=[
                    {
                        "readout_id": "cd8_fraction",
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
# Readout                                                                      #
# --------------------------------------------------------------------------- #
class TestReadoutRegistry:
    def test_minimal_registry_validates(self):
        _, readouts = _registries()
        assert readouts.get("cd8_fraction").quantity_kind == "fraction"

    def test_unregistered_quantity_kind_rejected(self):
        with pytest.raises(ValueError, match="no 'other' bucket"):
            _registries(readout_over={"quantity_kind": "luminosity"})

    def test_unregistered_modality_rejected(self):
        with pytest.raises(ValueError, match="not registered"):
            _registries(readout_over={"assay_modality": "vibes"})

    def test_requires_reference_is_enforced(self):
        with pytest.raises(ValueError, match="requires_reference"):
            _registries(readout_over={"quantity_kind": "foldchange"})

    def test_reference_satisfies_requirement(self):
        _, readouts = _registries(
            readout_over={
                "quantity_kind": "foldchange",
                "reference": {"kind": "timepoint", "timepoint": 0.0, "timepoint_units": "day"},
            }
        )
        assert readouts.get("cd8_fraction").reference.timepoint == 0.0

    def test_timepoint_reference_needs_units(self):
        with pytest.raises(ValueError, match="needs timepoint and timepoint_units"):
            _registries(
                readout_over={
                    "quantity_kind": "foldchange",
                    "reference": {"kind": "timepoint", "timepoint": 0.0},
                }
            )

    def test_scenario_reference_rejects_timepoint(self):
        with pytest.raises(ValueError, match="must not set timepoint"):
            _registries(
                readout_over={
                    "quantity_kind": "foldchange",
                    "reference": {"kind": "scenario", "scenario": "baseline", "timepoint": 1.0},
                }
            )

    def test_unknown_reference_kind_rejected(self):
        with pytest.raises(ValueError, match="must be 'timepoint' or 'scenario'"):
            _registries(
                readout_over={
                    "quantity_kind": "foldchange",
                    "reference": {"kind": "vibe"},
                }
            )

    def test_duplicate_readout_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate readout_id"):
            ReadoutRegistry.model_validate(
                {
                    "quantity_kinds": [{"id": "fraction", "description": "d"}],
                    "assay_modalities": [{"id": "mihc", "description": "d"}],
                    "readouts": [
                        {
                            "readout_id": "x",
                            "description": "d",
                            "quantity_kind": "fraction",
                            "assay_modality": "mihc",
                            "units": "dimensionless",
                            "numerator_species": ["A"],
                        }
                    ]
                    * 2,
                }
            )

    def test_identical_numerator_and_denominator_rejected(self):
        with pytest.raises(ValueError, match="constant at 1"):
            _registries(readout_over={"denominator_species": ["V_T.CD8"]})

    def test_load_from_yaml(self, tmp_path):
        _, readouts = _registries()
        p = tmp_path / "readouts.yaml"
        p.write_text(yaml.safe_dump(readouts.model_dump()))
        assert load_readouts(p).get("cd8_fraction").units == "dimensionless"


# --------------------------------------------------------------------------- #
# Cross-target checks                                                          #
# --------------------------------------------------------------------------- #
class TestRegistryAudit:
    def test_clean_corpus_passes(self):
        cohorts, readouts = _registries()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_registries({"t1": _target()}, cohorts, readouts)

    def test_unknown_cohort_reported(self):
        cohorts, readouts = _registries()
        problems = find_registry_problems({"t1": _target(cohort_id="ghost")}, cohorts, readouts)
        assert [p.kind for p in problems] == ["unknown_cohort"]

    def test_unknown_readout_reported(self):
        cohorts, readouts = _registries()
        t = _target()
        t["observable"]["readout_id"] = "ghost"
        kinds = {p.kind for p in find_registry_problems({"t1": t}, cohorts, readouts)}
        assert "unknown_readout" in kinds

    def test_pooled_target_rejected(self):
        """Several sources behind one target is a meta-analysis, not a cohort."""
        cohorts, readouts = _registries()
        t = _target(
            empirical_data={
                "inputs": [
                    {"source_ref": "Golesworthy2022"},
                    {"source_ref": "Liu2015"},
                    {"source_ref": "Jansen2021"},
                ]
            }
        )
        problems = find_registry_problems({"t1": t}, cohorts, readouts)
        pooled = [p for p in problems if p.kind == "pooled_target"]
        assert len(pooled) == 1
        assert "Split into one target per source" in pooled[0].detail

    def test_mechanistic_target_exempt_from_pooling_and_cohort(self):
        cohorts, readouts = _registries()
        t = _target(cohort_id=None, epistemic_basis="mechanistic")
        assert find_registry_problems({"t1": t}, cohorts, readouts) == []

    def test_n_evaluable_may_not_exceed_cohort(self):
        cohorts, readouts = _registries()
        t = _target()
        t["empirical_data"]["n_evaluable"] = 99
        kinds = {p.kind for p in find_registry_problems({"t1": t}, cohorts, readouts)}
        assert "n_evaluable_exceeds_cohort" in kinds

    def test_source_disagreeing_with_cohort_reported(self):
        cohorts, readouts = _registries()
        t = _target(primary_data_source={"source_tag": "Li2022_CancerCell"})
        kinds = {p.kind for p in find_registry_problems({"t1": t}, cohorts, readouts)}
        assert "source_disagrees_with_cohort" in kinds

    def test_duplicate_row_reported(self):
        """One cohort reports a readout once."""
        cohorts, readouts = _registries()
        problems = find_registry_problems({"t1": _target(), "t2": _target()}, cohorts, readouts)
        dup = [p for p in problems if p.kind == "duplicate_row"]
        assert len(dup) == 1
        assert dup[0].target_ids == ("t1", "t2")

    def test_same_readout_must_compute_same_expression(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="other")])
        _, readouts = _registries()
        t2 = _target(cohort_id="other")
        t2["observable"] = dict(
            t2["observable"],
            code="def f(t, species_dict, c):\n    return species_dict['V_T.CD4'] / species_dict['V_T.nucleated']\n",
        )
        problems = find_registry_problems({"t1": _target(), "t2": t2}, cohorts, readouts)
        kinds = {p.kind for p in problems}
        assert "readout_composition_disagrees" in kinds

    def test_same_readout_same_expression_is_fine(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="other")])
        _, readouts = _registries()
        problems = find_registry_problems(
            {"t1": _target(), "t2": _target(cohort_id="other")}, cohorts, readouts
        )
        assert [p.kind for p in problems] == []

    def test_check_raises_with_every_problem(self):
        cohorts, readouts = _registries()
        with pytest.raises(ValueError, match="registry problem"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                check_registries({"t1": _target(cohort_id="ghost")}, cohorts, readouts)

    def test_unused_entries_warn(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="unused")])
        _, readouts = _registries()
        with pytest.warns(UserWarning, match="no target uses"):
            unused = warn_unused_registry_entries({"t1": _target()}, cohorts, readouts)
        assert unused == ["unused"]


class TestResolveN:
    def test_falls_back_to_cohort_n(self):
        cohorts, _ = _registries()
        assert resolve_n(_target(), cohorts) == 9

    def test_n_evaluable_wins(self):
        cohorts, _ = _registries()
        t = _target()
        t["empirical_data"]["n_evaluable"] = 6
        assert resolve_n(t, cohorts) == 6

    def test_unknown_cohort_gives_none(self):
        cohorts, _ = _registries()
        assert resolve_n(_target(cohort_id="ghost"), cohorts) is None
