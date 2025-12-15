# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

**Purpose:** These parameter extractions will be used as **informative priors for simulation-based inference (SBI)** during QSP model calibration. The extracted distributions (median, IQR, 95% CI) will inform Bayesian parameter estimation, helping constrain the parameter space during model fitting to experimental data.

**IMPORTANT:** The following primary studies have already been used for this parameter (same name and context). Do NOT reuse these studies - find independent sources instead:

{{USED_PRIMARY_STUDIES}}

If no studies are listed above, this is the first derivation for this parameter.

**ADDITIONAL RESTRICTION:** Do NOT use any sources that are already cited or referenced in the parameter descriptions, species descriptions, or other context information in the MODEL_CONTEXT section below. These sources were used to define the model structure, not to estimate this specific parameter value. You must find NEW, INDEPENDENT sources for your parameter estimation.

For this parameter, you must:

---

## Monte Carlo Parameter Estimation

1. **Structured inputs:** Define all input values in the `inputs` list with source references
2. **Function-based code:** Provide Python code as a `derive_parameter(inputs, ureg)` function
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

**Use Pint for unit-safe calculations.** The function must return Pint Quantities (not raw floats) with units matching the parameter's declared units.

**GOLDEN RULE: Keep values tethered to their units as long as possible.** Only extract `.magnitude` when absolutely necessary (e.g., for distribution parameters in `rng.lognormal`). Let Pint propagate units through calculations—this catches dimensional errors automatically.

```python
import numpy as np

def derive_parameter(inputs, ureg):
    """
    Args:
        inputs: dict mapping input name to Pint Quantity
        ureg: Pint UnitRegistry (includes custom units like 'cell', 'nanomolarity')

    Returns:
        dict with Pint Quantities: median_param, iqr_param, ci95_param
    """
    # Access inputs directly as Pint Quantities
    half_life = inputs['half_life']  # e.g., <Quantity(5, 'hour')>

    # Monte Carlo - extract magnitude only for distribution, reattach units immediately
    N = 10000
    rng = np.random.default_rng(42)
    half_life_samples = rng.lognormal(
        np.log(half_life.magnitude), 0.3, size=N
    ) * half_life.units

    # Pint handles dimensional analysis and unit conversions
    k_samples = np.log(2) / half_life_samples
    k_per_day = k_samples.to(1 / ureg.day)

    # NumPy functions preserve Pint units - no need to extract magnitudes
    return {
        'median_param': np.median(k_per_day),
        'iqr_param': np.percentile(k_per_day, 75) - np.percentile(k_per_day, 25),
        'ci95_param': [np.percentile(k_per_day, 2.5), np.percentile(k_per_day, 97.5)]
    }
```

**NumPy functions that work directly with Pint Quantities:**
- Statistics: `mean`, `median`, `std`, `var`, `percentile`, `quantile`, `min`, `max`, `sum`, `prod`
- Array ops: `concatenate`, `stack`, `hstack`, `vstack`, `reshape`, `squeeze`, `diff`
- Math: `add`, `subtract`, `multiply`, `divide`, `sqrt`, `square`, `abs`, `exp`, `log`
- Comparison: `greater`, `less`, `equal`, `isclose`, `allclose`

**Common unit aliases (defined in ureg):**
- `nanomolarity` = `nanomolar`, `cell` = cell counts, `dimensionless` = unitless ratios

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
- Python function `derive_parameter(inputs, ureg)` returning median_param, iqr_param, ci95_param
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

**Text Snippets:** See detailed rules in Source Guidelines above - snippets are automatically verified.

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
- `inputs`: array with name, value, units (Pint-parseable), description, source_ref, value_table_or_section, value_snippet
- Numbers as numbers, not strings
- Every source_ref must have corresponding source entry
