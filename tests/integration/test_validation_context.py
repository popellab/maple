#!/usr/bin/env python3
"""
Smoke tests for validation context setup.

These tests verify that runtime configuration files load correctly and contain
expected data. They catch bugs where the wrong file format is loaded or files
are missing expected content.
"""

from pathlib import Path

import pytest

from qsp_llm_workflows.core.model_structure import ModelStructure


# Path to real model_structure.json in jobs/input_data/
MODEL_STRUCTURE_PATH = Path(__file__).parent.parent.parent / "jobs/input_data/model_structure.json"


@pytest.fixture
def model_structure():
    """Load the real model_structure.json file."""
    if not MODEL_STRUCTURE_PATH.exists():
        pytest.skip(f"model_structure.json not found at {MODEL_STRUCTURE_PATH}")
    return ModelStructure.from_json(MODEL_STRUCTURE_PATH)


class TestModelStructureLoadsCorrectly:
    """Smoke tests that model_structure.json loads with expected content."""

    def test_model_structure_has_parameters(self, model_structure):
        """Verify model_structure.json loads with parameters.

        This catches the bug where model_definitions.json (flat dict format)
        was accidentally used instead of model_structure.json (array format),
        resulting in 0 parameters being loaded.
        """
        assert len(model_structure.parameters) > 100, (
            f"Expected > 100 parameters, got {len(model_structure.parameters)}. "
            "Check that model_structure.json (not model_definitions.json) is being loaded."
        )

    def test_model_structure_has_species(self, model_structure):
        """Verify model has species loaded."""
        assert (
            len(model_structure.species) > 10
        ), f"Expected > 10 species, got {len(model_structure.species)}"

    def test_model_structure_has_reactions(self, model_structure):
        """Verify model has reactions loaded."""
        assert (
            len(model_structure.reactions) > 10
        ), f"Expected > 10 reactions, got {len(model_structure.reactions)}"

    def test_key_parameters_exist(self, model_structure):
        """Verify key parameters used in tests exist in model."""
        key_params = ["k_C1_growth", "k_C1_death", "k_CD8_act"]

        for param_name in key_params:
            param = model_structure.get_parameter(param_name)
            assert param is not None, (
                f"Expected parameter '{param_name}' not found in model_structure. "
                f"Available parameters: {len(model_structure.parameters)}"
            )

    def test_parameter_has_units(self, model_structure):
        """Verify parameters have units defined."""
        # Check a sample of parameters have non-empty units
        params_with_units = [
            p for p in model_structure.parameters if p.units and p.units != "dimensionless"
        ]
        assert (
            len(params_with_units) > 50
        ), f"Expected > 50 parameters with units, got {len(params_with_units)}"

    def test_get_reactions_for_parameter_works(self, model_structure):
        """Verify reaction lookup by parameter works."""
        # Find a parameter that should be in reactions
        param = model_structure.get_parameter("k_C1_growth")
        if param is None:
            pytest.skip("k_C1_growth parameter not found")

        reactions = model_structure.get_reactions_for_parameter("k_C1_growth")
        assert (
            len(reactions) >= 1
        ), f"Expected k_C1_growth to be used in at least 1 reaction, got {len(reactions)}"
