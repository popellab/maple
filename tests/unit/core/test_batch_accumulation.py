#!/usr/bin/env python3
"""Tests for BatchAccumulationModel: schema, Python evaluator, and Julia codegen."""

import math
import pytest
from unittest.mock import Mock

from maple.core.calibration.submodel_target import (
    BatchAccumulationModel,
    InputRef,
    ReferenceRef,
)
from maple.core.calibration.submodel_utils import (
    _evaluate_structured_model,
    run_prior_predictive,
    STRUCTURED_ALGEBRAIC_TYPES,
)


class TestBatchAccumulationSchema:
    """Test Pydantic model creation and validation."""

    def test_minimal_creation(self):
        """Create with required fields only; defaults fill in."""
        m = BatchAccumulationModel(
            secretion_rate="k_CCL2_sec",
            cell_count="100000.0",
            incubation_time="2.0",
            molecular_weight="8679.0",
            data_rationale="test",
            submodel_rationale="test",
        )
        assert m.type == "batch_accumulation"
        assert m.medium_volume == "1.0"
        assert m.unit_conversion_factor == "1.0"

    def test_with_reference_ref(self):
        """molecular_weight can be a ReferenceRef."""
        m = BatchAccumulationModel(
            secretion_rate="k_CCL2_sec",
            cell_count="100000.0",
            incubation_time="2.0",
            molecular_weight=ReferenceRef(reference_ref="ccl2_molecular_weight_mature"),
            data_rationale="test",
            submodel_rationale="test",
        )
        assert isinstance(m.molecular_weight, ReferenceRef)
        assert m.molecular_weight.reference_ref == "ccl2_molecular_weight_mature"

    def test_with_input_ref(self):
        """cell_count can be an InputRef."""
        m = BatchAccumulationModel(
            secretion_rate="k_CCL2_sec",
            cell_count=InputRef(input_ref="n_cells_prsc"),
            incubation_time="2.0",
            molecular_weight="8679.0",
            data_rationale="test",
            submodel_rationale="test",
        )
        assert isinstance(m.cell_count, InputRef)

    def test_type_in_structured_set(self):
        """batch_accumulation is in STRUCTURED_ALGEBRAIC_TYPES."""
        assert "batch_accumulation" in STRUCTURED_ALGEBRAIC_TYPES


class TestBatchAccumulationEvaluator:
    """Test Python evaluator for batch_accumulation."""

    def _make_model(self, **overrides):
        defaults = dict(
            type="batch_accumulation",
            secretion_rate="k_CCL2_sec",
            cell_count="100000.0",
            incubation_time="2.0",
            molecular_weight="8679.0",
            medium_volume="1.0",
            unit_conversion_factor="1.0",
        )
        defaults.update(overrides)
        m = Mock()
        for k, v in defaults.items():
            setattr(m, k, v)
        return m

    def test_ng_output(self):
        """k=1.1e-9 nmol/cell/day, 1e5 cells, 2 days, MW=8679 → ~1.91 ng."""
        model = self._make_model()
        # predicted = k * N * t * MW * ucf / V
        # = 1.1e-9 * 1e5 * 2.0 * 8679.0 * 1.0 / 1.0
        # = 1.1e-9 * 1e5 = 1.1e-4; * 2 = 2.2e-4; * 8679 = 1.90938
        result = _evaluate_structured_model(
            model,
            param_values={"k_CCL2_sec": 1.1e-9},
            input_values={},
        )
        expected = 1.1e-9 * 1e5 * 2.0 * 8679.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_pg_per_ml_output(self):
        """Fujita-like: k * N * t_h * MW * ucf / V where ucf=41.6667 folds h→day + ng→pg."""
        model = self._make_model(
            cell_count=InputRef(input_ref="n_cells_prsc"),
            incubation_time=InputRef(input_ref="incubation_time_hours"),
            medium_volume=InputRef(input_ref="V_medium_ml"),
            unit_conversion_factor="41.6667",
        )
        k = 1.1e-9
        inputs = {
            "n_cells_prsc": 100000.0,
            "incubation_time_hours": 48.0,
            "V_medium_ml": 1.0,
        }
        result = _evaluate_structured_model(
            model,
            param_values={"k_CCL2_sec": k},
            input_values=inputs,
        )
        # = 1.1e-9 * 1e5 * 48 * 8679 * 41.6667 / 1.0
        # = 1.1e-9 * 1e5 * 48 * 8679 * 41.6667
        expected = k * 100000.0 * 48.0 * 8679.0 * 41.6667 / 1.0
        assert result == pytest.approx(expected, rel=1e-4)
        # Should be close to 1909 pg/mL (the Fujita measurement is 1888)
        assert 1800 < result < 2000

    def test_with_reference_db(self):
        """molecular_weight via reference_ref resolves from reference_db."""
        model = self._make_model(
            molecular_weight=ReferenceRef(reference_ref="ccl2_molecular_weight_mature"),
        )
        ref_db = {"ccl2_molecular_weight_mature": 8679.0}
        result = _evaluate_structured_model(
            model,
            param_values={"k_CCL2_sec": 1.1e-9},
            input_values={},
            reference_db=ref_db,
        )
        expected = 1.1e-9 * 1e5 * 2.0 * 8679.0
        assert result == pytest.approx(expected, rel=1e-6)

    def test_run_prior_predictive_integration(self):
        """run_prior_predictive routes batch_accumulation correctly."""
        model = self._make_model()
        prior = Mock()
        prior.distribution = "lognormal"
        prior.mu = math.log(1.1e-9)
        prior.sigma = 0.5

        result = run_prior_predictive(
            model=model,
            prior=prior,
            param_name="k_CCL2_sec",
            state_variables=None,
            independent_variable=None,
            measurement=None,
            input_values={},
        )
        expected = 1.1e-9 * 1e5 * 2.0 * 8679.0
        assert result == pytest.approx(expected, rel=1e-6)


class TestBatchAccumulationJuliaCodegen:
    """Test Julia code generation for batch_accumulation."""

    def test_generates_julia_code(self):
        """_generate_steady_state_compute produces valid Julia for batch_accumulation."""
        from maple.core.calibration.julia_translator import (
            _generate_steady_state_compute,
        )

        model = BatchAccumulationModel(
            secretion_rate="k_CCL2_sec",
            cell_count=InputRef(input_ref="n_cells"),
            incubation_time="2.0",
            molecular_weight=ReferenceRef(reference_ref="ccl2_molecular_weight_mature"),
            unit_conversion_factor="1.0",
            medium_volume="1.0",
            data_rationale="test",
            submodel_rationale="test",
        )

        code = _generate_steady_state_compute(
            model=model,
            func_name="test_ccl2",
            param_names=["k_CCL2_sec"],
            inputs_code='Dict("n_cells" => 100000.0)',
            params_dict='Dict("k_CCL2_sec" => k_CCL2_sec)',
            ref_values_code='Dict("ccl2_molecular_weight_mature" => 8679.0)',
        )

        assert code is not None
        assert 'sec_rate = params["k_CCL2_sec"]' in code
        assert 'cells = inputs["n_cells"]' in code
        assert "t_inc = 2.0" in code
        assert 'mw = reference_values["ccl2_molecular_weight_mature"]' in code
        assert "ucf = 1.0" in code
        assert "vol = 1.0" in code
        assert "return sec_rate * cells * t_inc * mw * ucf / vol" in code
        assert "compute_test_ccl2(params, inputs, reference_values)" in code
