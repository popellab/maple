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
    covariance_blocks,
    find_registry_problems,
    resolve_n,
    warn_merged_blocks,
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
    readout = dict(
        quantity_kind=QuantityKind.FRACTION,
        assay_modality=AssayModality.MIHC,
        numerator_species=["V_T.CD8"],
        denominator_species=["V_T.nucleated"],
    )
    readout.update(over.pop("readout", {}))
    for key in ("quantity_kind", "assay_modality", "reference"):
        if key in over:
            readout[key] = over.pop(key)
    base = dict(
        readout=readout,
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
        observable={
            "code": _CODE,
            "readout_time": 0.0,
            "readout": {
                "quantity_kind": "fraction",
                "assay_modality": "mihc",
                "numerator_species": ["V_T.CD8"],
                "denominator_species": ["V_T.nucleated"],
            },
        },
        empirical_data={"inputs": [{"source_ref": "Li2022_CancerCell_PDAC_AntiPD1"}]},
        primary_data_source={"source_tag": "Li2022_CancerCell_PDAC_AntiPD1"},
    )
    base.update(over)
    return base


def _fraction(numerator, denominator, center=None, **over):
    """A target whose code divides ``numerator`` by the sum of ``denominator``."""
    den = " + ".join(f"species_dict['{s}']" for s in denominator)
    t = _target(
        observable={
            "code": (
                "def compute_observable(time, species_dict, constants):\n"
                f"    return species_dict['{numerator}'] / ({den})\n"
            ),
            "readout_time": 0.0,
            "readout": {
                "quantity_kind": "fraction",
                "assay_modality": "mihc",
                "numerator_species": [numerator],
                "denominator_species": list(denominator),
            },
        },
        **over,
    )
    if center is not None:
        t["empirical_data"]["observed_distribution"] = {
            "statistics": [{"stat": "quantile", "p": 0.5, "value": center}],
            "spread_source": "across_patient",
        }
    return t


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
# Readout                                                                      #
# --------------------------------------------------------------------------- #
class TestReadoutComposition:
    """The declared composition is the row's identity; the code is audited against it."""

    def test_numerator_is_required(self):
        with pytest.raises(ValidationError):
            _observable(readout={"numerator_species": []})

    def test_denominator_defaults_to_empty(self):
        obs = _observable(readout={"denominator_species": []})
        assert obs.readout.denominator_species == []

    def test_identical_numerator_and_denominator_rejected(self):
        with pytest.raises(ValidationError, match="constant at 1"):
            _observable(
                readout={
                    "numerator_species": ["V_T.CD8"],
                    "denominator_species": ["V_T.CD8"],
                }
            )

    def test_experimental_denominator_needs_model_species(self):
        with pytest.raises(ValidationError, match="denominator_species"):
            _observable(
                readout={
                    "denominator_species": [],
                    "experimental_denominator": "all nucleated cells",
                }
            )


class TestObservableMeasurementAttributes:
    """The attributes inference reads to build the measurement-discrepancy design."""

    def test_minimal_observable_validates(self):
        assert _observable().readout.quantity_kind == QuantityKind.FRACTION

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
        assert obs.readout.reference.timepoint == 0.0

    def test_foldchange_with_a_scenario_reference(self):
        obs = _observable(
            quantity_kind=QuantityKind.FOLDCHANGE,
            reference={"kind": "scenario", "scenario": "baseline_no_treatment"},
        )
        assert obs.readout.reference.scenario == "baseline_no_treatment"

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
        b = _observable(readout={"numerator_species": ["V_T.Treg"]})
        assert (a.readout.quantity_kind, a.readout.assay_modality) == (
            b.readout.quantity_kind,
            b.readout.assay_modality,
        )


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

    def test_duplicate_row_keys_on_the_declared_composition(self):
        """Different numerators over one denominator are different rows."""
        t2 = _target()
        t2["observable"] = dict(t2["observable"])
        t2["observable"]["readout"] = dict(
            t2["observable"]["readout"], numerator_species=["V_T.CD4"]
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


class TestSingularBlocks:
    """Rows of one cohort that are deterministic functions of each other."""

    _CAF = ["V_T.iCAF", "V_T.myCAF"]

    def _kinds(self, targets, cohorts=None):
        return [p.kind for p in find_registry_problems(targets, cohorts or _registry())]

    def test_numerators_partitioning_their_denominator_flagged(self):
        targets = {
            "icaf": _fraction("V_T.iCAF", self._CAF),
            "mycaf": _fraction("V_T.myCAF", self._CAF),
        }
        problems = find_registry_problems(targets, _registry())
        assert [p.kind for p in problems] == ["singular_block"]
        assert problems[0].target_ids == ("icaf", "mycaf")
        assert "partition" in problems[0].detail

    def test_numerators_not_exhausting_the_denominator_are_fine(self):
        targets = {
            "cd8": _fraction("V_T.CD8", ["V_T.nucleated"]),
            "cd4": _fraction("V_T.CD4", ["V_T.nucleated"]),
        }
        assert self._kinds(targets) == []

    def test_centers_summing_to_one_flagged(self):
        """The model observables need not be complementary for the data to be."""
        den = self._CAF + ["V_T.apCAF"]
        targets = {
            "icaf": _fraction("V_T.iCAF", den, center=0.284),
            "mycaf": _fraction("V_T.myCAF", den, center=0.716),
        }
        problems = find_registry_problems(targets, _registry())
        assert [p.kind for p in problems] == ["singular_block"]
        assert "one number reported twice" in problems[0].detail

    def test_centers_summing_to_one_hundred_flagged(self):
        den = self._CAF + ["V_T.apCAF"]
        targets = {
            "icaf": _fraction("V_T.iCAF", den, center=28.4),
            "mycaf": _fraction("V_T.myCAF", den, center=71.6),
        }
        assert self._kinds(targets) == ["singular_block"]

    def test_centers_falling_short_of_one_are_fine(self):
        den = self._CAF + ["V_T.apCAF"]
        targets = {
            "icaf": _fraction("V_T.iCAF", den, center=0.28),
            "mycaf": _fraction("V_T.myCAF", den, center=0.61),
        }
        assert self._kinds(targets) == []

    def test_centers_in_different_cohorts_are_fine(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="other")])
        den = self._CAF + ["V_T.apCAF"]
        targets = {
            "icaf": _fraction("V_T.iCAF", den, center=0.284),
            "mycaf": _fraction("V_T.myCAF", den, center=0.716, cohort_id="other"),
        }
        assert self._kinds(targets, cohorts) == []

    def test_center_falls_back_to_the_computed_median(self):
        den = self._CAF + ["V_T.apCAF"]
        a = _fraction("V_T.iCAF", den)
        b = _fraction("V_T.myCAF", den)
        a["empirical_data"]["median"] = [0.284]
        b["empirical_data"]["median"] = [0.716]
        assert self._kinds({"icaf": a, "mycaf": b}) == ["singular_block"]

    def test_a_non_fraction_pair_is_not_checked(self):
        """Two densities summing to 1 in their own units is a coincidence, not a constraint."""
        code = "def compute_observable(time, species_dict, constants):\n    return species_dict['V_T.CD8']\n"
        targets = {}
        for tid, center in (("a", 0.284), ("b", 0.716)):
            t = _target(observable={"code": code, "readout_time": 0.0})
            t["empirical_data"]["median"] = [center]
            targets[tid] = t
        assert self._kinds(targets) == []


def _arm(role, scenario, cohort_id, numerator=("V_T.CD8_TLA",), **over):
    base = dict(
        role=role,
        scenario=scenario,
        cohort_id=cohort_id,
        required_species=list(numerator),
        observable_code="def compute_test_statistic(t, s):\n    return s['V_T.CD8_TLA']\n",
        readout={
            "quantity_kind": "density",
            "assay_modality": "mihc",
            "numerator_species": list(numerator),
        },
    )
    base.update(over)
    return base


def _contrast(**over):
    """A two-arm cross-scenario target, as parsed YAML."""
    base = dict(
        epistemic_basis="literature",
        observable={
            "inputs": [
                _arm("nivo", "gvax_nivo_neoadjuvant", "arm_b"),
                _arm("urelumab", "gvax_nivo_urelumab_neoadjuvant", "arm_c"),
            ]
        },
        empirical_data={"inputs": []},
    )
    base.update(over)
    return base


def _two_arms(**over):
    b = _cohort(cohort_id="arm_b", scenarios=["gvax_nivo_neoadjuvant"], n_c=10)
    c = _cohort(cohort_id="arm_c", scenarios=["gvax_nivo_urelumab_neoadjuvant"], n_c=8, **over)
    return CohortRegistry(cohorts=[b, c])


class TestCrossScenarioArms:
    """A contrast over disjoint arms is a derived row over several cohorts."""

    def _kinds(self, targets, cohorts):
        return [p.kind for p in find_registry_problems(targets, cohorts)]

    def test_placed_arms_pass(self):
        assert self._kinds({"inv": _contrast()}, _two_arms()) == []

    def test_unknown_cohort_on_an_arm_reported(self):
        t = _contrast()
        t["observable"]["inputs"][1]["cohort_id"] = "ghost"
        assert self._kinds({"inv": t}, _two_arms()) == ["unknown_cohort"]

    def test_arm_scenario_must_belong_to_its_cohort(self):
        t = _contrast()
        t["observable"]["inputs"][1]["scenario"] = "baseline_no_treatment"
        assert self._kinds({"inv": t}, _two_arms()) == ["scenario_not_in_cohort"]

    def test_arm_n_evaluable_may_not_exceed_its_cohort(self):
        t = _contrast()
        t["observable"]["inputs"][1]["n_evaluable"] = 99
        assert self._kinds({"inv": t}, _two_arms()) == ["n_evaluable_exceeds_cohort"]

    def test_arms_sharing_patients_are_a_paired_contrast(self):
        b = _cohort(
            cohort_id="arm_b",
            scenarios=["gvax_nivo_neoadjuvant"],
            n_c=10,
            shares_patients_with=["arm_c"],
        )
        c = _cohort(
            cohort_id="arm_c",
            scenarios=["gvax_nivo_urelumab_neoadjuvant"],
            n_c=8,
            shares_patients_with=["arm_b"],
        )
        problems = find_registry_problems({"inv": _contrast()}, CohortRegistry(cohorts=[b, c]))
        assert [p.kind for p in problems] == ["paired_contrast_as_cross_scenario"]
        assert "resampling each arm independently" in problems[0].detail

    def test_arm_duplicating_a_standalone_target_reported(self):
        """Conditioning on the constituent and the contrast counts one number twice."""
        standalone = _target(cohort_id="arm_b")
        standalone["observable"] = dict(
            standalone["observable"],
            readout={
                "quantity_kind": "density",
                "assay_modality": "mihc",
                "numerator_species": ["V_T.CD8_TLA"],
            },
        )
        kinds = self._kinds({"inv": _contrast(), "cd8_tla": standalone}, _two_arms())
        assert "redundant_cross_scenario_arm" in kinds

    def test_mechanistic_contrast_needs_no_cohorts(self):
        t = _contrast(epistemic_basis="mechanistic")
        for arm in t["observable"]["inputs"]:
            arm.pop("cohort_id")
        assert self._kinds({"inv": t}, _two_arms()) == []


class TestCovarianceBlocks:
    def test_independent_cohorts_are_their_own_blocks(self):
        assert covariance_blocks({}, _two_arms()) == [frozenset({"arm_b"}), frozenset({"arm_c"})]

    def test_a_contrast_merges_the_cohorts_it_draws_on(self):
        assert covariance_blocks({"inv": _contrast()}, _two_arms()) == [
            frozenset({"arm_b", "arm_c"})
        ]

    def test_shared_patients_merge_without_any_target(self):
        a = _cohort(cohort_id="a", shares_patients_with=["b"])
        b = _cohort(cohort_id="b", shares_patients_with=["a"])
        c = _cohort(cohort_id="c")
        blocks = covariance_blocks({}, CohortRegistry(cohorts=[a, b, c]))
        assert blocks == [frozenset({"a", "b"}), frozenset({"c"})]

    def test_both_edge_kinds_compose_into_one_component(self):
        b = _cohort(cohort_id="arm_b", scenarios=["gvax_nivo_neoadjuvant"], n_c=10)
        c = _cohort(
            cohort_id="arm_c",
            scenarios=["gvax_nivo_urelumab_neoadjuvant"],
            n_c=8,
            shares_patients_with=["arm_d"],
        )
        d = _cohort(cohort_id="arm_d", shares_patients_with=["arm_c"])
        blocks = covariance_blocks({"inv": _contrast()}, CohortRegistry(cohorts=[b, c, d]))
        assert blocks == [frozenset({"arm_b", "arm_c", "arm_d"})]

    def test_a_scalar_target_merges_nothing(self):
        cohorts = CohortRegistry(cohorts=[_cohort(), _cohort(cohort_id="other")])
        blocks = covariance_blocks({"t1": _target()}, cohorts)
        assert blocks == [frozenset({"li2022_arm_a"}), frozenset({"other"})]

    def test_merged_blocks_warn(self):
        with pytest.warns(UserWarning, match="one covariance block"):
            merged = warn_merged_blocks({"inv": _contrast()}, _two_arms())
        assert merged == [frozenset({"arm_b", "arm_c"})]

    def test_independent_blocks_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert warn_merged_blocks({}, _two_arms()) == []


class TestResolveN:
    def test_falls_back_to_cohort_n(self):
        assert resolve_n(_target(), _registry()) == 9

    def test_n_evaluable_wins(self):
        t = _target()
        t["empirical_data"]["n_evaluable"] = 6
        assert resolve_n(t, _registry()) == 6

    def test_unknown_cohort_gives_none(self):
        assert resolve_n(_target(cohort_id="ghost"), _registry()) is None
