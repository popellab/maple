# Goal

You are a research assistant helping to extract and formalize test statistics for quantitative systems pharmacology (QSP) model validation from scientific literature.
Your task is to create **comprehensive, reproducible test statistic definitions** that quantify expected distributions of model-derived quantities based on experimental literature.

{{EXISTING_TEST_STATISTICS}}

For this test statistic, you must:

---

## Test Statistic Definition & Quantification
1. **Mathematical formalization:** Create a precise mathematical definition of how the test statistic is computed from the specified model output formula.
2. **Biological interpretation:** Explain what the test statistic measures biologically and why it's relevant for model validation.
3. **Literature data extraction:** Identify and extract quantitative measurements from the literature that can be transformed into the expected distribution for the exact model species formula provided.

---

## Model Integration
4. **Model output computation:** Provide MATLAB code that transforms pre-extracted model species vectors into the test statistic. **CRITICAL: The test statistic must be computed from the exact model species provided, not from proxy measures.** The test harness handles SimBiology data extraction and provides named vectors.
5. **Code structure requirements:** Write a `compute_test_statistic` function that takes `time` and relevant species vectors as inputs, and returns the test statistic value. Focus on the core transformation logic.
6. **Scenario awareness:** Use the SCENARIO_CONTEXT provided below to understand the experimental context (drugs, dosing, patient parameters) that this test statistic applies to. This context is for your understanding only - do not generate a structured scenario object.

---

## Statistical Distribution Characterization
7. **Uncertainty quantification:** Generate Monte Carlo samples (≥2000) using bootstrap/resampling methods that capture all sources of experimental uncertainty. **The derivation code must transform literature data (e.g., response rates, imaging measurements) into the expected distribution for the exact model species formula.**
8. **Distribution parameters:** Calculate mean, variance, and 95% confidence interval from the resampled data for the model species, not proxy measures.
9. **Robust estimation:** Handle measurement error, biological variability, and methodological uncertainties appropriately while maintaining focus on the target model output.
10. **Code explanation:** Provide a step-by-step explanation of the R bootstrap methodology that is specific to your dataset and statistical approach.

---

## Experimental Documentation
10. **Study overview:** Provide a concise narrative explaining the measurement approach, biological rationale, and how the test statistic was derived.
11. **Technical details:** Document essential experimental details (measurement method, study design, sample size, data processing, key assumptions).
12. **Validation context:** Assess the relevance and quality of the literature data for model validation using the provided rubric.

---

## Literature Source Verification
13. **Citation accuracy:** Verify all citations come from real, accessible publications with correct DOI/URL information.
14. **Data location verification:** Confirm figure/table references contain the claimed data at specified locations.
15. **Source attribution:** Reference all sources consistently throughout with specific text snippets and locations.

---

## Quality & Reproducibility
16. **Statistical validity:** Ensure the uncertainty quantification approach is appropriate for the data type and experimental design.
17. **Biological plausibility:** Verify that test statistic values and distributions align with known biological ranges.
18. **Methodological transparency:** Document all assumptions, transformations, and analytical choices clearly.

---

## Template Structure
Use the provided test statistic template structure with these key sections:
- `test_statistic_definition`: Mathematical definition of the test statistic computation
- `model_output`: MATLAB code with `compute_test_statistic` function that transforms model species vectors into the test statistic value
- `expected_distribution`: Statistical parameters derived from literature
- `derivation_code_r`: Bootstrap/resampling code for uncertainty quantification
- `validation_weights`: Quality assessment using standardized rubrics
- `data_sources`: Detailed citation information with specific data extraction tracking
- `methodological_sources`: Background sources for methods, formulas, or context

**Note:** The scenario context (drugs, dosing, patient parameters) is provided in SCENARIO_CONTEXT below for your understanding, but you do not need to generate a structured scenario object.

---

## Provided Context

### Model Information
{{MODEL_CONTEXT}}

### Scenario Context
{{SCENARIO_CONTEXT}}

### Required Species with Units
The test statistic should be computed using these model species with their corresponding units:

{{REQUIRED_SPECIES_WITH_UNITS}}

**IMPORTANT:** These units must be used consistently in both your MATLAB `compute_test_statistic` function and your R derivation code. Ensure that your test statistic calculations, literature data transformations, and final results all use appropriate unit conversions to match these model species units.

### Derived Species Description
{{DERIVED_SPECIES_DESCRIPTION}}

### Template
{{TEMPLATE}}

### Examples
{{EXAMPLES}}

Fill out the YAML test statistic template for the specified biological expectation and experimental context.

---

## CRITICAL REQUIREMENTS

**Species Formula Compliance:** The test statistic MUST be computed from the exact model species listed in `{{REQUIRED_SPECIES}}`. Do NOT use proxy measures or clinical endpoints as substitutes. If the literature data provides proxy measures (like ORR, progression-free survival, etc.), your derivation code must transform these into the expected distribution for the actual model output described in `{{DERIVED_SPECIES_DESCRIPTION}}`.

**Example Transformation Approach:**
- If required species is `V_T.TumorVolume` and literature provides ORR data, derive the expected tumor volume reduction distribution that would produce the observed response rates
- If required species includes `V_T.T_eff,V_T.T_reg` and literature provides immune activation markers, derive the expected T cell ratio distribution
- The final test statistic samples in `mc_draws` must represent the derived species, not the literature proxy measure

**Code Implementation Requirements:**
- Write a `compute_test_statistic` function that takes `time` vector and named species vectors as inputs
- **CRITICAL**: Function inputs will be named exactly after the species in `{{REQUIRED_SPECIES}}` (e.g., if required_species is `V_T.TumorVolume,V_T.T_eff`, your function signature should be `compute_test_statistic(time, V_T_TumorVolume, V_T_T_eff)`)
- **Variable naming**: Convert dots to underscores in species names for MATLAB variables (e.g., `V_T.TumorVolume` becomes `V_T_TumorVolume`)
- Use interpolation methods (e.g., `interp1`) when evaluating at specific timepoints
- Document function inputs, outputs, and transformation logic clearly
- Focus on the core mathematical transformation from species vectors to test statistic value

---

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON):

```json
{
  "test_statistic_definition": "Mathematical definition of the test statistic...",
  "model_output": {
    "code": "function test_statistic = compute_test_statistic(time, V_T_C1)\n  % Your MATLAB code here\nend"
  },
  "study_overview": "...",
  "technical_details": "...",
  "expected_distribution": {
    "mean": 0.0,
    "variance": 0.0,
    "ci95": [0.0, 0.0],
    "units": "..."
  },
  "derivation_explanation": "...",
  "derivation_code_r": "...",
  "validation_weights": {
    "species_match": {"value": 0.0, "justification": "..."},
    "system_match": {"value": 0.0, "justification": "..."},
    "overall_confidence": {"value": 0.0, "justification": "..."},
    "indication_match": {"value": 0.0, "justification": "..."},
    "regimen_match": {"value": 0.0, "justification": "..."},
    "biomarker_population_match": {"value": 0.0, "justification": "..."},
    "stage_burden_match": {"value": 0.0, "justification": "..."}
  },
  "key_study_limitations": "...",
  "data_sources": {
    "PRIMARY_STUDY": {
      "citation": "Full citation",
      "doi": "DOI or NA",
      "data_extracted": [
        {
          "description": "What data was extracted (e.g., ORR at 8 weeks)",
          "value": 0.35,
          "units": "proportion",
          "figure_or_table": "Figure 2A",
          "text_snippet": "Exact quote from paper",
          "weight_in_synthesis": 0.6
        }
      ]
    }
  },
  "methodological_sources": {
    "METHOD_REF": {
      "citation": "Full citation",
      "doi": "DOI or NA",
      "used_for": "ORR to tumor volume conversion",
      "formula_or_method": "V = V0 * (1 - ORR * 0.7)",
      "figure_or_table": "Methods section",
      "text_snippet": "Relevant quote"
    }
  }
}
```

Requirements for JSON response:
- Wrap your entire response in ```json code block tags
- Use proper JSON syntax (all strings quoted, proper escaping)
- Numeric values should be actual numbers, not strings
- Use `\n` for line breaks in multi-line strings
- Include ALL sections from the template: test_statistic_definition, model_output, study_overview, technical_details, expected_distribution, derivation_explanation, derivation_code_r, validation_weights, key_study_limitations, data_sources, methodological_sources
- For model_output: Provide MATLAB code that computes the test statistic from the required species vectors (use SCENARIO_CONTEXT above for context on what scenario to consider)
- For data_sources: Use detailed extraction tracking with description, value, units, figure_or_table, text_snippet, and weight_in_synthesis for each data point
- For methodological_sources: Include used_for, formula_or_method (if applicable), figure_or_table, and text_snippet
