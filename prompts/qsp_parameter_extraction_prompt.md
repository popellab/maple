# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

{{EXISTING_STUDIES}}

**IMPORTANT:** The following primary studies have already been used for this parameter (same name and context). Do NOT reuse these studies - find independent sources instead:

{{USED_PRIMARY_STUDIES}}

If no studies are listed above, this is the first derivation for this parameter.

For this parameter, you must:

---

## Monte Carlo Parameter Estimation

1. **Structured inputs:** Define all input values in the `inputs` list with source references
2. **Function-based code:** Provide Python code as a `derive_parameter(inputs)` function
3. **Bootstrap preferred:** Use bootstrap resampling when raw data available
4. **Uncertainty propagation:** Incorporate ALL sources of uncertainty:
   - **Multiple measurements:** Use bootstrap resampling when combining multiple data points
   - **Composite parameters:** When parameter depends on multiple quantities, propagate uncertainty from each component
   - **Unit conversions:** Include uncertainty from conversion factors when applicable
   - **Model assumptions:** Account for parametric uncertainty when making distributional assumptions
5. **Standard units:** Use standard unit formats (e.g., "1/day", "nM", "mg/L", "dimensionless" for counts/ratios)
6. **Required outputs:** Function must return dict with:
   - `mean_param`: Mean of Monte Carlo draws
   - `variance_param`: Variance of Monte Carlo draws
   - `ci95_param`: 95% percentile confidence interval as [lower, upper]

---

## Experimental Documentation

6. **Study overview (1-2 sentences):** WHAT parameter is being measured, WHY it's biologically relevant, and the overall approach
7. **Study design (1-2 sentences):** HOW the measurement was performed (assay type, sample size, key methods)
8. **Key assumptions (enumerated dict):** 3-5 critical assumptions only (e.g., distributional assumptions, model choices, data quality). Use format: `1: "Assumption text"`, `2: "Assumption text"`. Do NOT include trivial assumptions like "bootstrap samples are independent" or "conversion factors are standard".
9. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (3-6 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
10. **Key study limitations:** List critical limitations and their specific impact on reliability

---

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

## Requirements Summary

**Code:**
- Python function `derive_parameter(inputs)` returning mean_param, variance_param, ci95_param
- Bootstrap preferred for uncertainty quantification
- Set random seed via inputs for reproducibility

**Documentation:**
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

## Provided Template
{{TEMPLATE}}

## Example
{{EXAMPLES}}

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

Fill out the metadata template for this parameter.

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON).

Response structure (see template for field details):
```json
{
  "mathematical_role": "...",
  "parameter_range": "positive_reals",
  "study_overview": "...",
  "study_design": "...",
  "parameter_estimates": {
    "inputs": [{"name": "...", "value": 0.45, "units": "...", ...}],
    "derivation_code": "import numpy as np\\n...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "1/day",
    "key_assumptions": {"1": "...", "2": "...", "3": "..."}
  },
  "derivation_explanation": "**Step 1:** ...\\n\\n**Step 2:** ...",
  "key_study_limitations": "- **Issue:** ...\\n- **Issue:** ...",
  "primary_data_sources": [{"source_tag": "...", "title": "...", ...}],
  "secondary_data_sources": [...],
  "methodological_sources": [...],
  "biological_relevance": {
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
