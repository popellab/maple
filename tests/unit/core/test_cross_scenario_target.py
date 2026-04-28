#!/usr/bin/env python3
"""
Tests for CrossScenarioCalibrationTarget model.

Schema-only tests — no observable code execution, no DOI resolution. The
composer code is exercised by the qsp-hpc-tools cross_scenario_loader
tests at runtime; here we only verify pydantic validation rules.
"""

import copy

import pytest
from pydantic import ValidationError

from maple.core.calibration import (
    CrossScenarioCalibrationTarget,
    CrossScenarioInput,
    CrossScenarioObservable,
)


# ============================================================================
# Fixtures
# ============================================================================


def _scalar_empirical():
    """Minimal valid CalibrationTargetEstimates payload for a mechanistic
    cross-scenario target — single-element distribution, empty inputs."""
    return {
        "median": [0.7],
        "ci95": [[0.4, 1.1]],
        "units": "dimensionless",
        "sample_size": 1,
        "sample_size_rationale": (
            "Mechanistic prior; sample_size=1 denotes a single soft-prior " "assertion."
        ),
        "inputs": [],
        "assumptions": [],
        "distribution_code": (
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(42)\n"
            "    lo, hi = 0.4, 1.1\n"
            "    mu = 0.5 * (np.log(lo) + np.log(hi))\n"
            "    sigma = (np.log(hi) - np.log(lo)) / (2.0 * 1.96)\n"
            "    samples = rng.lognormal(mu, sigma, 100000)"
            " * ureg('dimensionless')\n"
            "    return {\n"
            "        'median_obs': np.median(samples),\n"
            "        'ci95_lower': np.percentile(samples, 2.5),\n"
            "        'ci95_upper': np.percentile(samples, 97.5),\n"
            "    }\n"
        ),
    }


@pytest.fixture
def scalar_inputs_target():
    """Two scalar test_statistic inputs composed into a fold ratio."""
    return {
        "cross_scenario_target_id": "tumor_diameter_d90_gvax_vs_untreated_fold",
        "observable": {
            "code": (
                "def compute(inputs, ureg):\n"
                "    return inputs['treated'] / inputs['untreated']\n"
            ),
            "units": "dimensionless",
            "inputs": [
                {
                    "role": "untreated",
                    "scenario": "clinical_progression_longterm",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "tumor_diameter_day_90",
                },
                {
                    "role": "treated",
                    "scenario": "gvax_neoadjuvant_longterm",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "tumor_diameter_day_90",
                },
            ],
        },
        "empirical_data": _scalar_empirical(),
        "study_interpretation": (
            "GVAX monotherapy is expected to produce at most modest growth "
            "suppression at d90 relative to untreated, per the low published "
            "ORR for single-agent GVAX in PDAC."
        ),
        "key_assumptions": [
            "Mechanistic prior; CI95 width chosen to admit both modest "
            "response and metastatic-trial baseline of unchanged growth.",
        ],
        "epistemic_basis": "mechanistic",
        "cancer_type": "PDAC",
    }


@pytest.fixture
def timeseries_inputs_target():
    """Two timeseries inputs composed into a time-to-50% fold ratio."""
    return {
        "cross_scenario_target_id": "time_to_tumor_50pct_gvax_vs_untreated_fold",
        "observable": {
            "code": (
                "def compute(inputs, ureg):\n"
                "    import numpy as np\n"
                "    def t_half(scen):\n"
                "        c = scen['V_T.C1'].to('cell').magnitude\n"
                "        t = scen['time'].to('day').magnitude\n"
                "        target = 0.5 * c[0]\n"
                "        below = np.where(c <= target)[0]\n"
                "        return float(t[below[0]])"
                " if len(below) else float('nan')\n"
                "    return (t_half(inputs['treated']) /\n"
                "            t_half(inputs['untreated'])) * ureg.dimensionless\n"
            ),
            "units": "dimensionless",
            "inputs": [
                {
                    "role": "untreated",
                    "scenario": "clinical_progression_longterm",
                    "input_kind": "timeseries",
                    "required_species": ["V_T.C1"],
                },
                {
                    "role": "treated",
                    "scenario": "gvax_neoadjuvant_longterm",
                    "input_kind": "timeseries",
                    "required_species": ["V_T.C1"],
                },
            ],
        },
        "empirical_data": _scalar_empirical(),
        "study_interpretation": (
            "Time-to-50%-tumor-shrinkage with GVAX vs untreated should "
            "exceed 1.0 (treatment-arm reaches the threshold faster) but "
            "by a modest margin given low single-agent GVAX response."
        ),
        "key_assumptions": [
            "Mechanistic prior; required_species kept minimal so the "
            "composer is robust to scenario-specific species availability.",
        ],
        "epistemic_basis": "mechanistic",
        "cancer_type": "PDAC",
    }


# ============================================================================
# Golden tests
# ============================================================================


def test_scalar_inputs_target_validates(scalar_inputs_target):
    target = CrossScenarioCalibrationTarget(**scalar_inputs_target)
    assert target.cross_scenario_target_id == "tumor_diameter_d90_gvax_vs_untreated_fold"
    assert target.epistemic_basis == "mechanistic"
    assert len(target.observable.inputs) == 2
    assert all(inp.input_kind == "test_statistic" for inp in target.observable.inputs)


def test_timeseries_inputs_target_validates(timeseries_inputs_target):
    target = CrossScenarioCalibrationTarget(**timeseries_inputs_target)
    assert all(inp.input_kind == "timeseries" for inp in target.observable.inputs)
    assert target.observable.inputs[0].required_species == ["V_T.C1"]


def test_mixed_input_kinds_validate(scalar_inputs_target):
    """One scalar input + one timeseries input — both kinds in the same target."""
    cfg = copy.deepcopy(scalar_inputs_target)
    cfg["observable"]["inputs"][1] = {
        "role": "treated",
        "scenario": "gvax_neoadjuvant_longterm",
        "input_kind": "timeseries",
        "required_species": ["V_T.C1", "V_T"],
    }
    target = CrossScenarioCalibrationTarget(**cfg)
    kinds = {inp.input_kind for inp in target.observable.inputs}
    assert kinds == {"test_statistic", "timeseries"}


# ============================================================================
# Input validation
# ============================================================================


def test_test_statistic_input_requires_test_statistic_id():
    with pytest.raises(ValidationError, match="no test_statistic_id"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            input_kind="test_statistic",
        )


def test_test_statistic_input_rejects_required_species():
    with pytest.raises(ValidationError, match="required_species is set"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            input_kind="test_statistic",
            test_statistic_id="some_id",
            required_species=["V_T.C1"],
        )


def test_timeseries_input_requires_required_species():
    with pytest.raises(ValidationError, match="required_species is empty"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            input_kind="timeseries",
        )


def test_timeseries_input_rejects_test_statistic_id():
    with pytest.raises(ValidationError, match="test_statistic_id is set"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            input_kind="timeseries",
            required_species=["V_T.C1"],
            test_statistic_id="some_id",
        )


# ============================================================================
# Observable validation
# ============================================================================


def test_observable_requires_at_least_two_inputs():
    """A 'cross-scenario' observable with only one input isn't crossing scenarios."""
    with pytest.raises(ValidationError, match="at least 2"):
        CrossScenarioObservable(
            code="def compute(inputs, ureg): return inputs['only']",
            units="dimensionless",
            inputs=[
                {
                    "role": "only",
                    "scenario": "s",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "x",
                }
            ],
        )


def test_observable_rejects_duplicate_roles():
    with pytest.raises(ValidationError, match="roles must be unique"):
        CrossScenarioObservable(
            code="def compute(inputs, ureg): return inputs['a']",
            units="dimensionless",
            inputs=[
                {
                    "role": "a",
                    "scenario": "s1",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "x",
                },
                {
                    "role": "a",
                    "scenario": "s2",
                    "input_kind": "test_statistic",
                    "test_statistic_id": "y",
                },
            ],
        )


# ============================================================================
# Cross-target validators
# ============================================================================


def test_units_must_match_empirical(scalar_inputs_target):
    cfg = copy.deepcopy(scalar_inputs_target)
    cfg["observable"]["units"] = "day"  # mismatch with empirical_data.units='dimensionless'
    with pytest.raises(ValidationError, match="must match"):
        CrossScenarioCalibrationTarget(**cfg)


def test_literature_basis_requires_primary_data_source(scalar_inputs_target):
    cfg = copy.deepcopy(scalar_inputs_target)
    cfg["epistemic_basis"] = "literature"
    # primary_data_source omitted
    with pytest.raises(ValidationError, match="requires primary_data_source"):
        CrossScenarioCalibrationTarget(**cfg)


def test_extra_fields_forbidden(scalar_inputs_target):
    cfg = copy.deepcopy(scalar_inputs_target)
    cfg["unknown_field"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted|extra"):
        CrossScenarioCalibrationTarget(**cfg)
