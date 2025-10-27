# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

{{EXISTING_STUDIES}}

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

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON):

```json
{
  "mathematical_role": "Describe the mathematical role...",
  "parameter_range": "positive_reals",
  "study_overview": "Brief 1-2 sentence summary of WHAT and WHY...",
  "study_design": "Brief 1-2 sentence summary of HOW...",
  "parameter_estimates": {
    "inputs": [
      {
        "name": "median_survival",
        "value": 0.45,
        "units": "years",
        "description": "Median survival from trial endpoint",
        "source_ref": "MARCHINGO2014",
        "value_table_or_section": "Table 2",
        "value_snippet": "Median survival was 0.45 years (95% CI: 0.3-0.6) in the treatment arm.",
        "units_table_or_section": "Table 2",
        "units_snippet": "Median survival was 0.45 years (95% CI: 0.3-0.6) in the treatment arm."
      },
      {
        "name": "plasma_clearance_reference",
        "value": 0.67,
        "units": "L/h",
        "description": "Reference plasma clearance rate for similar therapeutic class",
        "source_ref": "GOODMAN2018",
        "value_table_or_section": "Table 4.3",
        "value_snippet": "Typical plasma clearance for checkpoint inhibitors ranges from 0.5-0.8 L/h, with population mean of 0.67 L/h.",
        "units_table_or_section": "Table 4.3",
        "units_snippet": "Clearance values are reported in liters per hour (L/h) for consistency with standard pharmacokinetic conventions."
      },
      {
        "name": "conversion_factor",
        "value": 365.25,
        "units": "days/year",
        "description": "Days per year for unit conversion",
        "source_ref": null,
        "value_table_or_section": null,
        "value_snippet": null,
        "units_table_or_section": null,
        "units_snippet": null
      }
    ],
    "derivation_code": "import numpy as np\\n\\ndef derive_parameter(inputs):\\n    median_survival = inputs['median_survival']['value']\\n    ...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "1/day",
    "key_assumptions": {
      "1": "Exponential survival model (constant hazard rate)",
      "2": "Parameter uncertainty follows normal distribution (CLT with n=100)",
      "3": "Two-compartment model adequately describes drug distribution"
    }
  },
  "derivation_explanation": "**Step 1:** Extract data. ASSUMPTION: Exponential survival...\\n\\n**Step 2:** Calculate rate. ASSUMPTION: ...",
  "key_study_limitations": "- **Sample size:** ...\\n- **Measurement issues:** ...",
  "primary_data_sources": [
    {
      "source_tag": "MARCHINGO2014",
      "title": "Full article title",
      "first_author": "Marchingo",
      "year": 2014,
      "doi": "10.xxxx/xxxxx"
    }
  ],
  "secondary_data_sources": [
    {
      "source_tag": "GOODMAN2018",
      "title": "Goodman & Gilman's: The Pharmacological Basis of Therapeutics",
      "first_author": "Brunton",
      "year": 2018,
      "doi": null
    }
  ],
  "methodological_sources": [
    {
      "source_tag": "COMPARTMENTAL2008",
      "title": "Pharmacokinetic compartmental analysis using exponential fitting",
      "first_author": "Gibaldi",
      "year": 2008,
      "doi": "10.xxxx/xxxxx",
      "used_for": "Two-compartment clearance formula",
      "method_description": "Used Equation 3.14 for converting elimination rate"
    }
  ],
  "biological_relevance": {
    "species_match": {"value": 1.0, "justification": "Human study"},
    "system_match": {"value": 1.0, "justification": "In vivo"},
    "overall_confidence": {"value": 0.85, "justification": "Good design, minor caveats"},
    "indication_match": {"value": 1.0, "justification": "Exact PDAC match"},
    "regimen_match": {"value": 1.0, "justification": "Baseline untreated"},
    "biomarker_population_match": {"value": 0.85, "justification": "Close match"},
    "stage_burden_match": {"value": 0.65, "justification": "Earlier stage"}
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
