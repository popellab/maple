#!/usr/bin/env python3
"""
Tests for CrossScenarioCalibrationTarget model.

Schema-only tests — no observable code execution, no DOI resolution. The
per-arm observable_code and the reduction code are exercised by the
qsp-hpc-tools cross_scenario_loader / composer tests at runtime; here we
only verify pydantic validation rules.
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


def _intla_number_observable_code():
    """A within-TLA CD8 number-at-d21 per-arm observable (compute_test_statistic
    contract: (time, species_dict) -> float)."""
    return (
        "def compute_test_statistic(time, species_dict):\n"
        "    import numpy as np\n"
        "    t = np.asarray(time, dtype=float)\n"
        "    n = (np.asarray(species_dict['V_T.CD8_TLA'], dtype=float)\n"
        "         + np.asarray(species_dict['V_T.CD8_TLA_act'], dtype=float))\n"
        "    return float(np.interp(21.0, t, n))\n"
    )


@pytest.fixture
def invariance_target():
    """Two self-contained per-arm inputs composed into a cross-arm ratio."""
    code = _intla_number_observable_code()
    return {
        "cross_scenario_target_id": "cd8_intla_number_invariance_nivo_vs_urelumab",
        "observable": {
            "code": ("def compute(inputs):\n" "    return inputs['urelumab'] / inputs['nivo']\n"),
            "units": "dimensionless",
            "inputs": [
                {
                    "role": "nivo",
                    "scenario": "gvax_nivo_neoadjuvant_zheng2022",
                    "observable_code": code,
                    "required_species": ["V_T.CD8_TLA", "V_T.CD8_TLA_act"],
                },
                {
                    "role": "urelumab",
                    "scenario": "gvax_nivo_urelumab_neoadjuvant_heumann2023",
                    "observable_code": code,
                    "required_species": ["V_T.CD8_TLA", "V_T.CD8_TLA_act"],
                },
            ],
        },
        "empirical_data": _scalar_empirical(),
        "study_interpretation": (
            "Adding urelumab to GVAX+nivo changes within-TLA CD8 activation "
            "state, not within-TLA CD8 number; the day-21 ratio of within-TLA "
            "CD8 count between arms is ~1.0."
        ),
        "key_assumptions": [
            "Mechanistic prior; the per-arm absolute count is deliberately "
            "kept out of the NPE x (its unmodeled B-cell denominator cancels "
            "only in the ratio), so only the invariant constrains the fit.",
        ],
        "epistemic_basis": "mechanistic",
        "cancer_type": "PDAC",
    }


# ============================================================================
# Golden test
# ============================================================================


def test_invariance_target_validates(invariance_target):
    target = CrossScenarioCalibrationTarget(**invariance_target)
    assert target.cross_scenario_target_id == "cd8_intla_number_invariance_nivo_vs_urelumab"
    assert target.epistemic_basis == "mechanistic"
    assert len(target.observable.inputs) == 2
    nivo = target.observable.inputs[0]
    assert nivo.role == "nivo"
    assert nivo.required_species == ["V_T.CD8_TLA", "V_T.CD8_TLA_act"]
    assert "compute_test_statistic" in nivo.observable_code


def test_per_input_units_field_forbidden(invariance_target):
    """Per-input units were dropped when the runtime went pintless; the
    reduction takes raw floats. A stray per-input 'units' must fail loudly."""
    cfg = copy.deepcopy(invariance_target)
    cfg["observable"]["inputs"][0]["units"] = "cell"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted|extra"):
        CrossScenarioCalibrationTarget(**cfg)


# ============================================================================
# Input validation
# ============================================================================


def test_input_requires_observable_code():
    with pytest.raises(ValidationError, match="observable_code"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            required_species=["V_T.C1"],
        )


def test_input_requires_required_species():
    with pytest.raises(ValidationError, match="required_species"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            observable_code="def compute_test_statistic(time, species_dict): return 0.0",
        )


def test_input_rejects_empty_required_species():
    with pytest.raises(ValidationError, match="required_species|at least 1|too_short"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            observable_code="def compute_test_statistic(time, species_dict): return 0.0",
            required_species=[],
        )


def test_input_rejects_legacy_input_kind():
    """Legacy 'input_kind' / 'test_statistic_id' fields are no longer part of
    the schema; extra='forbid' should reject them so stale YAMLs fail loudly."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted|extra"):
        CrossScenarioInput(
            role="x",
            scenario="s",
            observable_code="def compute_test_statistic(time, species_dict): return 0.0",
            required_species=["V_T.C1"],
            input_kind="test_statistic",
        )


# ============================================================================
# Observable validation
# ============================================================================


def _one_input():
    return {
        "role": "only",
        "scenario": "s",
        "observable_code": "def compute_test_statistic(time, species_dict): return 0.0",
        "required_species": ["V_T.C1"],
    }


def test_observable_requires_at_least_two_inputs():
    """A 'cross-scenario' observable with only one input isn't crossing scenarios."""
    with pytest.raises(ValidationError, match="at least 2|too_short"):
        CrossScenarioObservable(
            code="def compute(inputs): return inputs['only']",
            units="dimensionless",
            inputs=[_one_input()],
        )


def test_observable_rejects_duplicate_roles():
    a = _one_input()
    a["role"] = "a"
    b = copy.deepcopy(a)
    b["scenario"] = "s2"
    with pytest.raises(ValidationError, match="roles must be unique"):
        CrossScenarioObservable(
            code="def compute(inputs): return inputs['a']",
            units="dimensionless",
            inputs=[a, b],
        )


# ============================================================================
# Cross-target validators
# ============================================================================


def test_units_must_match_empirical(invariance_target):
    cfg = copy.deepcopy(invariance_target)
    cfg["observable"]["units"] = "day"  # mismatch with empirical_data.units='dimensionless'
    with pytest.raises(ValidationError, match="must match"):
        CrossScenarioCalibrationTarget(**cfg)


def test_literature_basis_requires_primary_data_source(invariance_target):
    cfg = copy.deepcopy(invariance_target)
    cfg["epistemic_basis"] = "literature"
    # primary_data_source omitted
    with pytest.raises(ValidationError, match="requires primary_data_source"):
        CrossScenarioCalibrationTarget(**cfg)


def test_extra_fields_forbidden(invariance_target):
    cfg = copy.deepcopy(invariance_target)
    cfg["unknown_field"] = "value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted|extra"):
        CrossScenarioCalibrationTarget(**cfg)
