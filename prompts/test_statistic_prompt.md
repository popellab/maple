# Goal

You are a research assistant helping to extract and formalize test statistics for quantitative systems pharmacology (QSP) model validation from scientific literature.
Your task is to create **comprehensive, reproducible test statistic definitions** that quantify expected distributions of model-derived quantities based on experimental literature.

{{EXISTING_TEST_STATISTICS}}

**IMPORTANT:** The following primary studies have already been used for this test statistic (same test_statistic_id and context). Do NOT reuse these studies - find independent sources instead:

{{USED_PRIMARY_STUDIES}}

If no studies are listed above, this is the first derivation for this test statistic.

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
- Text snippets (values_and_units_snippet, evidence_snippet) must be VERBATIM quotes from the source - copy exact wording, do not paraphrase or summarize
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

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON).

Response structure (see template for field details):
```json
{
  "test_statistic_definition": "...",
  "study_overview": "...",
  "study_design": "...",
  "test_statistic_estimates": {
    "inputs": [{"name": "...", "value": 0.28, "units": "...", ...}],
    "derivation_code": "import numpy as np\\n...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "dimensionless",
    "key_assumptions": {"1": "...", "2": "...", "3": "..."}
  },
  "derivation_explanation": "**Step 1:** ...\\n\\n**Step 2:** ...",
  "key_study_limitations": "- **Issue:** ...\\n- **Issue:** ...",
  "primary_data_sources": [{"source_tag": "...", "title": "...", ...}],
  "secondary_data_sources": [...],
  "methodological_sources": [...],
  "validation_weights": {
    "species_match": {"value": 1.0, "justification": "..."},
    ...
  }
}
```

Key requirements:
- Wrap in ```json code block tags
- Use `\n` for line breaks, `\n\n` for paragraphs
- `derivation_code`: raw Python (no ```python wrapper)
- `inputs`: array with name, value, units, description, source_ref, value_table_or_section, value_snippet, units_table_or_section, units_snippet
- Numbers as numbers, not strings
- Every source_ref must have corresponding source entry
- **Do NOT generate model_output code** - human-provided
