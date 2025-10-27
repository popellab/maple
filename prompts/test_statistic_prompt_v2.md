# Goal

You are a research assistant helping to extract and formalize test statistics for quantitative systems pharmacology (QSP) model validation from scientific literature.
Your task is to create **comprehensive, reproducible test statistic definitions** that quantify expected distributions of model-derived quantities based on experimental literature.

{{EXISTING_TEST_STATISTICS}}

For this test statistic, you must:

---

## Test Statistic Definition & Quantification
1. **Mathematical formalization:** Create a precise mathematical definition of how the test statistic is computed
2. **Biological interpretation:** Explain what the test statistic measures biologically and why it's relevant for model validation
3. **Literature data extraction:** Identify and extract quantitative measurements from text/tables that can be transformed into the expected distribution

---

## Statistical Distribution Characterization

**IMPORTANT:** The `model_output` code has already been provided by humans and is NOT part of your task. You only need to generate the statistical distribution from literature.

4. **Structured inputs:** Define all input values in the `inputs` list with source references
5. **Function-based code:** Provide Python code as a `derive_distribution(inputs)` function
6. **Bootstrap preferred:** Use bootstrap resampling when raw data available
7. **Uncertainty propagation:** Incorporate ALL sources of uncertainty:
   - **Multiple measurements:** Use bootstrap resampling when combining multiple data points
   - **Composite test statistics:** When test statistic depends on multiple quantities, propagate uncertainty from each component
   - **Measurement error:** Include uncertainty from assay variability when applicable
   - **Model assumptions:** Account for parametric uncertainty when making distributional assumptions
8. **Standard units:** Use standard unit formats (e.g., "percent", "mm³", "cells/µL", "dimensionless")
9. **Required outputs:** Function must return dict with:
   - `mean_stat`: Mean of Monte Carlo draws
   - `variance_stat`: Variance of Monte Carlo draws
   - `ci95_stat`: 95% percentile confidence interval as [lower, upper]

---

## Experimental Documentation

10. **Study overview (1-2 sentences):** WHAT test statistic is being measured, WHY it's biologically relevant for validation, and the overall approach
11. **Study design (1-2 sentences):** HOW the measurement was performed (assay type, sample size, key methods)
12. **Key assumptions (enumerated dict):** 3-5 critical assumptions only (e.g., distributional assumptions, model choices, data quality). Use format: `1: "Assumption text"`, `2: "Assumption text"`. Do NOT include trivial assumptions.
13. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (3-6 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
14. **Key study limitations:** List critical limitations and their specific impact on reliability

---

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

## Requirements Summary

**Important:** The `model_output` code is human-generated and NOT your responsibility.

**Code:**
- Python function `derive_distribution(inputs)` returning mean_stat, variance_stat, ci95_stat
- Bootstrap preferred for uncertainty quantification
- Set random seed via inputs for reproducibility

**Documentation:**
- `test_statistic_definition`: Mathematical definition
- `study_overview` (1-2 sentences): WHAT and WHY
- `study_design` (1-2 sentences): HOW
- `key_assumptions` (enumerated dict): 3-5 critical assumptions only
- `derivation_explanation`: Step-by-step with "ASSUMPTION N: ..." references

**Sources:**
- Separate primary/secondary/methodological sources
- All values/locations in inputs, not sources
- Text/table extraction only (no digitization)

**Validation:**
- Citations are real, accessible publications
- Text snippets match sources
- Weights follow rubric tables exactly
- Code uses exactly the defined inputs

---

## Provided Context

### Model Information
{{MODEL_CONTEXT}}

### Scenario Context
{{SCENARIO_CONTEXT}}

### Required Species with Units
The human-provided `model_output` code computes the test statistic from these model species:

{{REQUIRED_SPECIES_WITH_UNITS}}

**Note:** You do not need to write code that uses these species - that code is already provided in `model_output`. Your task is to generate the expected distribution from literature.

### Derived Species Description
{{DERIVED_SPECIES_DESCRIPTION}}

### Template
{{TEMPLATE}}

### Examples
{{EXAMPLES}}

Fill out the test statistic template for this biological expectation and experimental context.

---

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON):

```json
{
  "test_statistic_definition": "Mathematical definition...",
  "study_overview": "Brief 1-2 sentence summary of WHAT and WHY...",
  "study_design": "Brief 1-2 sentence summary of HOW...",
  "test_statistic_estimates": {
    "inputs": [
      {
        "name": "response_rate",
        "value": 0.28,
        "units": "dimensionless",
        "description": "Objective response rate from clinical trial",
        "source_ref": "TOPALIAN2012",
        "value_table_or_section": "Table 2",
        "value_snippet": "Among patients with melanoma, the objective response rate was 28%...",
        "units_table_or_section": "Table 2",
        "units_snippet": "Objective response rate was 28% (95% CI: 18-40%)..."
      }
    ],
    "derivation_code": "import numpy as np\\n\\ndef derive_distribution(inputs):\\n    ...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "dimensionless",
    "key_assumptions": {
      "1": "Binomial sampling adequately models patient heterogeneity",
      "2": "Imaging measurement error is normally distributed with CV=8%",
      "3": "Single-arm response rate is comparable to control-adjusted response"
    }
  },
  "derivation_explanation": "**Step 1:** Extract data. ASSUMPTION 1: Binomial sampling...\\n\\n**Step 2:** Calculate rate. ASSUMPTION 2: ...",
  "key_study_limitations": "- **Sample size:** ...\\n- **Measurement issues:** ...",
  "primary_data_sources": [
    {
      "source_tag": "TOPALIAN2012",
      "title": "Full article title",
      "first_author": "Topalian",
      "year": 2012,
      "doi": "10.xxxx/xxxxx"
    }
  ],
  "secondary_data_sources": [],
  "methodological_sources": [],
  "validation_weights": {
    "species_match": {"value": 1.0, "justification": "Human study"},
    "system_match": {"value": 1.0, "justification": "In vivo"},
    "overall_confidence": {"value": 0.85, "justification": "Good design, minor caveats"},
    "indication_match": {"value": 1.0, "justification": "Exact melanoma match"},
    "regimen_match": {"value": 1.0, "justification": "Anti-PD-1 monotherapy"},
    "biomarker_population_match": {"value": 0.85, "justification": "Close match"},
    "stage_burden_match": {"value": 0.65, "justification": "Advanced disease"}
  }
}
```

Requirements for JSON response:
- Wrap entire response in ```json code block tags
- Use proper JSON syntax (all strings quoted, proper escaping)
- Use `\n` for line breaks in multi-line strings
- For `derivation_code`: provide ONLY the raw Python code without ```python wrapper tags
- Use `\n\n` (double newline) to separate paragraphs or list items
- Numeric values should be actual numbers, not strings
- `inputs` is an ARRAY of objects with: name, value, units, description, source_ref, value_table_or_section, value_snippet, units_table_or_section, units_snippet
- Ensure every input with a source_ref has a corresponding source in primary/secondary/methodological sources
- Do NOT include "Exact quote:" prefix in snippets - just the text itself
- Do NOT include page numbers in value_table_or_section or units_table_or_section - just "Table X" or "Section name"
- **Do NOT generate model_output code** - that is provided separately by humans
