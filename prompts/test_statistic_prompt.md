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

## Scenario & Model Integration
4. **Dosing scenario specification:** Configure the experimental scenario using the appropriate `schedule_dosing` format with appropriate drugs, doses, schedules, and patient parameters.
5. **Model output computation:** Provide MATLAB/SimBiology code that extracts and transforms the exact model species into the test statistic. **CRITICAL: The test statistic must be computed from the exact model species formula provided, not from proxy measures.** Use SimBiology's `select` function with proper compartment-qualified queries.
6. **Code structure requirements:** Include proper error handling, interpolation methods, and clear documentation of the transformation from raw model species to test statistic value.

---

## Statistical Distribution Characterization
7. **Uncertainty quantification:** Generate Monte Carlo samples (≥2000) using bootstrap/resampling methods that capture all sources of experimental uncertainty. **The derivation code must transform literature data (e.g., response rates, imaging measurements) into the expected distribution for the exact model species formula.**
8. **Distribution parameters:** Calculate mean, variance, and 95% confidence interval from the resampled data for the model species, not proxy measures.
9. **Robust estimation:** Handle measurement error, biological variability, and methodological uncertainties appropriately while maintaining focus on the target model output.

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
- `test_statistic`: Mathematical definition of the test statistic computation
- `scenario`: Dosing parameters, patient characteristics, and **nested `model_output` with MATLAB/SimBiology code**
- `expected_distribution`: Statistical parameters derived from literature
- `derivation_code_r`: Bootstrap/resampling code for uncertainty quantification
- `validation_weights`: Quality assessment using standardized rubrics
- `sources`: Detailed citation information with specific data locations

**Key Enhancement:** The `model_output` section is nested within `scenario` and contains executable MATLAB code that demonstrates how to extract and transform the target model species into the test statistic using SimBiology's `select` function.

---

## Provided Context

### Model Information
{{MODEL_CONTEXT}}

### Scenario Context
{{SCENARIO_CONTEXT}}

### Species Formula
The test statistic should be computed from this model output:
```
{{SPECIES_FORMULA}}
```

### Template
{{TEMPLATE}}

### Examples
{{EXAMPLES}}

Fill out the YAML test statistic template for the specified biological expectation and experimental context.

---

## CRITICAL REQUIREMENTS

**Species Formula Compliance:** The test statistic MUST be computed from the exact model species formula provided (`{{SPECIES_FORMULA}}`). Do NOT use proxy measures or clinical endpoints as substitutes. If the literature data provides proxy measures (like ORR, progression-free survival, etc.), your derivation code must transform these into the expected distribution for the actual model output.

**Example Transformation Approach:**
- If species formula is `V_T.TumorVolume` and literature provides ORR data, derive the expected tumor volume reduction distribution that would produce the observed response rates
- If species formula is `V_T.T_eff / V_T.T_reg` and literature provides immune activation markers, derive the expected T cell ratio distribution
- The final test statistic samples in `mc_draws` must represent the model species, not the literature proxy measure

**Code Implementation Requirements:**
- Write MATLAB functions that use `select(simData, {'Type','species', 'Name','[SPECIES_NAME]', 'Compartment','[COMPARTMENT_NAME]'})` to extract model species
- Include proper error handling for missing data or logging issues
- Use interpolation methods (e.g., `interp1`) when evaluating at specific timepoints
- Document all function inputs, outputs, and assumptions clearly
- Show the complete transformation from raw model species to final test statistic value

---

**IMPORTANT: Format your response as follows:**
```yaml
# Your complete YAML test statistic definition here
# Fill out all sections of the template above
```

Make sure to wrap your entire YAML response in ```yaml code block tags as shown above.
