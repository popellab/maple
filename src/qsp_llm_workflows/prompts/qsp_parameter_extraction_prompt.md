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

**IMPORTANT: Use Pint for unit-safe calculations.**

The derivation code must use the Pint library for all unit conversions and calculations. This ensures dimensional consistency between literature inputs and model parameter units.

**The function must return Pint Quantities, not raw floats.** The returned values will be validated to ensure they have the correct units matching the parameter's declared units.

```python
import numpy as np
import pint

def derive_parameter(inputs, ureg):
    """
    Derive parameter distribution from literature data using Pint units.

    Args:
        inputs: List of input dicts, each with 'name', 'value', 'units', etc.
        ureg: Pint UnitRegistry (provided by validator, includes custom units)

    Returns:
        dict with Pint Quantities:
        - 'median_param': Pint Quantity with parameter units
        - 'iqr_param': Pint Quantity with parameter units
        - 'ci95_param': [lower, upper] as Pint Quantities with parameter units
    """
    # Extract input with units
    # val = float([x for x in inputs if x['name']=='half_life'][0]['value'])
    # units_str = [x for x in inputs if x['name']=='half_life'][0]['units']
    # half_life = val * ureg.parse_expression(units_str)

    # Bootstrap/Monte Carlo for uncertainty
    N = 10000
    rng = np.random.default_rng(42)

    # Perform unit-aware calculations
    # k = np.log(2) / half_life  # Pint handles units automatically
    # k_per_day = k.to(1 / ureg.day)  # Convert to model units

    # NumPy functions work directly on Pint Quantities - no need to extract magnitudes
    # samples is a Pint Quantity array with correct units

    return {
        'median_param': np.median(samples),
        'iqr_param': np.percentile(samples, 75) - np.percentile(samples, 25),
        'ci95_param': [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
    }
```

**GOLDEN RULE: Keep values tethered to their units as long as possible.**

Let Pint propagate units through your entire calculation. This catches dimensional errors automatically and makes unit conversions explicit. **Never extract `.magnitude` until absolutely necessary (e.g., for lognormal distribution parameters).**

**Key Pint usage patterns:**

1. **Inputs are pre-converted to Pint Quantities:**
   The validator automatically converts your `inputs` list to a dict of Pint Quantities keyed by input name. Just access them directly:
   ```python
   # inputs is a dict: {'half_life': <Quantity(5, 'hour')>, ...}
   half_life = inputs['half_life']  # Already a Pint Quantity!
   ```

2. **Let units flow through calculations:**
   ```python
   # Half-life to rate constant - Pint handles the dimensional analysis
   k = np.log(2) / half_life  # If half_life is in hours, k is in 1/hour
   k_per_day = k.to(1 / ureg.day)  # Convert to model units

   # Concentration conversions
   conc_nM = concentration.to(ureg.nanomolar)
   ```

3. **Dimensionless parameters emerge naturally:**
   ```python
   # Ratios, fractions, Hill coefficients - units cancel automatically
   ratio = value1 / value2  # Pint tracks that units cancel
   ```

4. **NumPy works directly on Pint Quantities:**
   ```python
   # No need to extract .magnitude - NumPy preserves units
   median = np.median(samples)  # Returns Pint Quantity
   percentile = np.percentile(samples, 75)  # Returns Pint Quantity
   ```

5. **Monte Carlo with units - keep arrays as Quantities:**
   ```python
   half_life = inputs['half_life']  # Already a Pint Quantity

   # For lognormal, extract magnitude for the distribution parameter, then reattach units
   half_life_samples = rng.lognormal(
       np.log(half_life.magnitude), 0.3, size=N
   ) * half_life.units  # Reattach units immediately!

   k = np.log(2) / half_life_samples  # Array of Quantities with 1/hour units
   k_per_day = k.to(1 / ureg.day)  # Convert entire array
   ```

**Anti-patterns to AVOID:**

```python
# BAD: Manually parsing inputs - they're already Pint Quantities!
val = float(inputs['half_life'].magnitude)  # Unnecessary extraction
units = inputs['half_life'].units  # Just use the Quantity directly!

# BAD: Keeping magnitudes separate from units
samples_mag = rng.lognormal(mean, sigma, size=N)  # Unitless samples
# ... many lines of code ...
samples = samples_mag * ureg.hour  # Easy to forget or get wrong units

# BAD: Hard-coded conversion factors
k = 0.693 / half_life_hours / 24  # Manual hour→day conversion, error-prone

# GOOD: Let Pint handle conversions explicitly
k = np.log(2) / half_life  # Pint knows half_life units
k_per_day = k.to(1 / ureg.day)  # Clear, verifiable
```

**Best practice example:**

```python
def derive_parameter(inputs, ureg):
    # Step 1: Access inputs directly - they're already Pint Quantities!
    half_life = inputs['half_life']  # e.g., <Quantity(5, 'hour')>

    # Step 2: Monte Carlo - extract magnitude only for distribution, reattach immediately
    N = 10000
    rng = np.random.default_rng(42)
    half_life_samples = rng.lognormal(
        np.log(half_life.magnitude), 0.3, size=N
    ) * half_life.units  # Samples now have correct units

    # Step 3: Let Pint handle dimensional analysis
    k_samples = np.log(2) / half_life_samples  # Units: 1/[time]

    # Step 4: Convert to model units - Pint checks compatibility
    k_per_day = k_samples.to(1 / ureg.day)  # Would error if units incompatible

    # Step 5: Return Quantities - validator checks units match parameter definition
    return {
        'median_param': np.median(k_per_day),
        'iqr_param': np.percentile(k_per_day, 75) - np.percentile(k_per_day, 25),
        'ci95_param': [np.percentile(k_per_day, 2.5), np.percentile(k_per_day, 97.5)]
    }
```

**Common unit aliases (already defined in ureg):**
- `nanomolarity` = `nanomolar` (SimBiology convention)
- `cell` = custom unit for cell counts
- `1/day`, `1/hour` = rate constants
- `dimensionless` = unitless ratios, fractions, Hill coefficients

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
