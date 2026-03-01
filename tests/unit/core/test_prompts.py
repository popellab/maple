"""
Unit tests for simple prompt building functions.

Tests that prompts are correctly assembled with placeholder substitutions.
"""

from maple.core.prompts import (
    build_parameter_extraction_prompt,
    build_test_statistic_prompt,
)


class TestBuildParameterExtractionPrompt:
    """Test parameter extraction prompt building."""

    def test_basic_substitution(self):
        """Test that all placeholders are replaced."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="**Parameter Name:** k_growth\n**Units:** 1/day",
            model_context="Growth rate of cancer cells",
            cancer_type="PDAC",
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
            cancer_type="melanoma",
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
            cancer_type="NSCLC",
        )

        # Shared rubrics should be included (they contain validation guidance)
        assert len(prompt) > 1000  # Prompt should be substantial
        assert "{{SOURCE_AND_VALIDATION_RUBRICS}}" not in prompt

    def test_cancer_type_substitution(self):
        """Test that cancer type is substituted throughout prompt."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="**Parameter Name:** k_growth",
            model_context="Test model context",
            cancer_type="PDAC",
        )

        # Cancer type should appear multiple times (in hierarchy, requirements, etc.)
        assert "PDAC" in prompt
        assert prompt.count("PDAC") >= 5  # Should appear in multiple sections
        assert "{{CANCER_TYPE}}" not in prompt

    def test_cancer_type_in_goal_section(self):
        """Test that cancer type appears in the goal/purpose section."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="Test param",
            model_context="Test context",
            cancer_type="melanoma",
        )

        # Should mention cancer type in goal
        assert "melanoma" in prompt
        # Should have cancer-specific guidance
        assert "melanoma-specific" in prompt or "melanoma" in prompt.lower()

    def test_cancer_type_in_source_hierarchy(self):
        """Test that cancer type appears in source hierarchy guidance."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="Test param",
            model_context="Test context",
            cancer_type="NSCLC",
        )

        # Should have tiered hierarchy mentioning the cancer type
        assert "NSCLC-specific" in prompt or "NSCLC" in prompt
        # Should mention cross-indication guidance
        assert "cross-indication" in prompt.lower() or "indication" in prompt.lower()


class TestBuildTestStatisticPrompt:
    """Test test statistic prompt building."""

    def test_basic_substitution(self):
        """Test that all placeholders are replaced."""
        prompt = build_test_statistic_prompt(
            model_context="PDAC tumor growth model",
            scenario_context="Baseline no treatment",
            required_species_with_units="V_T.C (mm³)",
            derived_species_description="Tumor volume at day 14",
            cancer_type="PDAC",
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
            cancer_type="melanoma",
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
            cancer_type="NSCLC",
        )

        # Shared rubrics should be included
        assert len(prompt) > 1000
        assert "{{SOURCE_AND_VALIDATION_RUBRICS}}" not in prompt

    def test_cancer_type_substitution(self):
        """Test that cancer type is substituted throughout prompt."""
        prompt = build_test_statistic_prompt(
            model_context="Test model",
            scenario_context="Test scenario",
            required_species_with_units="V_T.C",
            derived_species_description="Test description",
            cancer_type="melanoma",
        )

        # Cancer type should appear in data source hierarchy guidance
        assert "melanoma" in prompt
        assert "{{CANCER_TYPE}}" not in prompt


class TestPromptContent:
    """Test that prompts contain expected content."""

    def test_parameter_prompt_has_key_instructions(self):
        """Test parameter prompt contains key instructions."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="Test",
            model_context="Test",
            cancer_type="PDAC",
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
            cancer_type="PDAC",
        )

        # Should contain key instruction sections
        assert "test statistic" in prompt.lower() or "model output" in prompt.lower()

    def test_parameter_prompt_emphasizes_cancer_specificity(self):
        """Test that parameter prompt emphasizes cancer-specific data."""
        prompt = build_parameter_extraction_prompt(
            parameter_info="Test param",
            model_context="Test context",
            cancer_type="PDAC",
        )

        # Should have strong cancer-specific guidance
        assert "PDAC-specific" in prompt
        assert "prioritize" in prompt.lower() or "CRITICAL" in prompt

    def test_test_statistic_prompt_emphasizes_cancer_specificity(self):
        """Test that test statistic prompt emphasizes cancer-specific data."""
        prompt = build_test_statistic_prompt(
            model_context="Test",
            scenario_context="Test",
            required_species_with_units="Test",
            derived_species_description="Test",
            cancer_type="melanoma",
        )

        # Should have cancer-specific data source hierarchy
        assert "melanoma" in prompt
        # Count occurrences - should be many due to tiered hierarchy
        assert prompt.count("melanoma") >= 10
