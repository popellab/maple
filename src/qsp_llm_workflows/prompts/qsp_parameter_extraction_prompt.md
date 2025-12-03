# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

**IMPORTANT:** The following primary studies have already been used for this parameter (same name and context). Do NOT reuse these studies - find independent sources instead:

{{USED_PRIMARY_STUDIES}}

If no studies are listed above, this is the first derivation for this parameter.

**ADDITIONAL RESTRICTION:** Do NOT use any sources that are already cited or referenced in the parameter descriptions, species descriptions, or other context information in the MODEL_CONTEXT section below. These sources were used to define the model structure, not to estimate this specific parameter value. You must find NEW, INDEPENDENT sources for your parameter estimation.

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
6. **Required outputs:** Function must return dict with outlier-robust statistics:
   - `median_param`: Median of Monte Carlo draws
   - `iqr_param`: Interquartile range (Q3 - Q1) of Monte Carlo draws
   - `ci95_param`: 95% percentile confidence interval as [lower, upper]

---

## Experimental Documentation

6. **Study overview (1-2 sentences):** WHAT parameter is being measured, WHY it's biologically relevant, and the overall approach
7. **Study design (1-2 sentences):** HOW the measurement was performed (assay type, sample size, key methods)
8. **Key assumptions (list):** 3-5 critical assumptions only (e.g., distributional assumptions, model choices, data quality). Each assumption should have a number and text. Do NOT include trivial assumptions like "bootstrap samples are independent" or "conversion factors are standard".
9. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (3-6 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
10. **Key study limitations:** List critical limitations and their specific impact on reliability

---

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

## Requirements Summary

**Code:**
- Python function `derive_parameter(inputs)` returning median_param, iqr_param, ci95_param
- Bootstrap preferred for uncertainty quantification
- Use outlier-robust statistics (median/IQR instead of mean/variance)
- Set random seed via inputs for reproducibility

**Documentation:**
- `study_overview` (1-2 sentences): WHAT and WHY
- `study_design` (1-2 sentences): HOW
- `key_assumptions` (list): 3-5 critical assumptions only with number and text
- `derivation_explanation`: Step-by-step with "ASSUMPTION N: ..." references

**Sources:**
- Separate primary and secondary sources
- All values/locations in inputs, not sources
- Text/table extraction only (no digitization)

**Validation:**
- Citations are real, accessible publications
- Weights follow rubric tables exactly
- Code uses exactly the defined inputs

**Text Snippets (CRITICAL for automated verification):**
Text snippets (`value_snippet`, `units_snippet`) are automatically verified against the full paper text. Follow these rules strictly:

1. **VERBATIM only**: Copy exact text from the paper. Never paraphrase, summarize, or reconstruct.
2. **No table reconstruction**: Do NOT create artificial table notation like `CD8^{+} | ... | 17 (9-30)`. Tables are flattened when we extract text, so this format won't match.
3. **Use continuous text spans**: Find a short, continuous phrase that contains the value. For table data, the snippet should be just the cell value and any immediately adjacent text, e.g., `"17 (9-30)"` not a reconstructed row.
4. **Include context when helpful**: A few surrounding words help locate the snippet, e.g., `"median survival of 18.2 months"` is better than just `"18.2"`.
5. **Avoid LaTeX formatting**: Write `CD8+` not `CD8^{+}`. Write subscripts inline: `CO2` not `CO_{2}`.
6. **Keep snippets short**: 5-50 words is ideal. Long snippets are harder to match exactly.
7. **For units**: Find where units are explicitly stated, e.g., `"expressed as cells per high-power field"` or `"measured in ng/mL"`.

**Good snippet examples:**
- `"median CD8+ density was 17 (IQR 9-30) cells/HPF"` ✓
- `"n = 137 patients"` ✓
- `"tumor volume measured in mm³"` ✓

**Bad snippet examples:**
- `"CD8^{+} | No neoadjuvant | 17 (9-30)"` ✗ (reconstructed table, LaTeX)
- `"The study found elevated levels"` ✗ (no actual value)
- `"approximately 17"` ✗ (paraphrased, paper says "17 (9-30)")

---

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

---

Extract parameter metadata following all requirements above.

**Key points:**
- Use `\n` for line breaks, `\n\n` for paragraphs in text fields
- Python code should be plain text (no markdown code fences within the code strings)
- `derivation_code`: raw Python (no ```python wrapper)
- `inputs`: array with name, value, units, description, source_ref, value_table_or_section, value_snippet, units_table_or_section, units_snippet
- Numbers as numbers, not strings
- Every source_ref must have corresponding source entry
