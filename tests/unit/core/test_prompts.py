"""
Unit tests for simple prompt building functions.

Tests that prompts are correctly assembled with placeholder substitutions.
"""
import pytest

from qsp_llm_workflows.core.prompts import (
    build_parameter_extraction_prompt,
    build_test_statistic_prompt,
    build_validation_fix_prompt,
)


class TestBuildParameterExtractionPrompt:
    """Test parameter extraction prompt building."""

    def test_basic_substitution(self):
        """Test that all placeholders are replaced."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="**Parameter Name:** k_growth\n**Units:** 1/day",
            model_context="Growth rate of cancer cells",
            used_primary_studies="- Smith et al. 2020",
        )

        # Check substitutions worked
        assert "k_growth" in prompt
        assert "1/day" in prompt
        assert "Growth rate of cancer cells" in prompt
        assert "Smith et al. 2020" in prompt

        # Check no placeholders remain
        assert "{{" not in prompt
        assert "}}" not in prompt

    def test_empty_used_studies(self):
        """Test with no used studies."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="**Parameter Name:** k_death",
            model_context="Death rate",
            used_primary_studies="",
        )

        assert "k_death" in prompt
        assert "Death rate" in prompt
        assert "{{" not in prompt

    def test_includes_shared_rubrics(self):
        """Test that shared rubrics are included."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="Test param",
            model_context="Test context",
        )

        # Shared rubrics should be included (they contain validation guidance)
        assert len(prompt) > 1000  # Prompt should be substantial
        assert "{{SOURCE_AND_VALIDATION_RUBRICS}}" not in prompt


class TestBuildTestStatisticPrompt:
    """Test test statistic prompt building."""

    def test_basic_substitution(self):
        """Test that all placeholders are replaced."""
        prompt = build_test_statistic_prompt(
            model_context="PDAC tumor growth model",
            scenario_context="Baseline no treatment",
            required_species_with_units="V_T.C (mm³)",
            derived_species_description="Tumor volume at day 14",
            used_primary_studies="- Johnson et al. 2021",
        )

        # Check substitutions worked
        assert "PDAC tumor growth model" in prompt
        assert "Baseline no treatment" in prompt
        assert "V_T.C (mm³)" in prompt
        assert "Tumor volume at day 14" in prompt
        assert "Johnson et al. 2021" in prompt

        # Check no placeholders remain
        assert "{{" not in prompt
        assert "}}" not in prompt

    def test_empty_used_studies(self):
        """Test with no used studies."""
        prompt = build_test_statistic_prompt(
            model_context="Model",
            scenario_context="Scenario",
            required_species_with_units="Species",
            derived_species_description="Description",
            used_primary_studies="",
        )

        assert "Model" in prompt
        assert "{{" not in prompt

    def test_includes_shared_rubrics(self):
        """Test that shared rubrics are included."""
        prompt = build_test_statistic_prompt(
            model_context="Test",
            scenario_context="Test",
            required_species_with_units="Test",
            derived_species_description="Test",
        )

        # Shared rubrics should be included
        assert len(prompt) > 1000
        assert "{{SOURCE_AND_VALIDATION_RUBRICS}}" not in prompt


class TestBuildValidationFixPrompt:
    """Test validation fix prompt building."""

    def test_basic_substitution(self):
        """Test that all placeholders are replaced."""
        prompt = build_validation_fix_prompt(
            yaml_content="schema_version: v3\nparameter_name: k_growth",
            validation_errors="Missing field: parameter_units",
            template_content="schema_version: v3\nparameter_name: str\nparameter_units: str",
        )

        # Check substitutions worked
        assert "k_growth" in prompt
        assert "Missing field: parameter_units" in prompt
        assert "parameter_units: str" in prompt

        # Check no placeholders remain
        assert "{{" not in prompt
        assert "}}" not in prompt


class TestPromptContent:
    """Test that prompts contain expected content."""

    def test_parameter_prompt_has_key_instructions(self):
        """Test parameter prompt contains key instructions."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="Test",
            model_context="Test",
        )

        # Should contain key instruction sections
        assert "Monte Carlo" in prompt or "bootstrap" in prompt
        assert "parameter" in prompt.lower()

    def test_test_statistic_prompt_has_key_instructions(self):
        """Test test statistic prompt contains key instructions."""
        prompt = build_test_statistic_prompt(
            model_context="Test",
            scenario_context="Test",
            required_species_with_units="Test",
            derived_species_description="Test",
        )

        # Should contain key instruction sections
        assert "test statistic" in prompt.lower() or "model output" in prompt.lower()
