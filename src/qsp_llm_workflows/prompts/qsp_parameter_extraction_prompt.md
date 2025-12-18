# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model **for {{CANCER_TYPE}}**.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data **specific to {{CANCER_TYPE}}**.

**Purpose:** These parameter extractions will be used as **informative priors for simulation-based inference (SBI)** during QSP model calibration for {{CANCER_TYPE}}. The extracted distributions (median, IQR, 95% CI) will inform Bayesian parameter estimation, helping constrain the parameter space during model fitting to {{CANCER_TYPE}} experimental data.

**CRITICAL:** Prioritize {{CANCER_TYPE}}-specific data. See "Cross-Indication Data Handling" section for scoring and uncertainty inflation rules.

**IMPORTANT:** The following primary studies have already been used for this parameter (same name and context). Do NOT reuse these studies - find independent sources instead:

{{USED_PRIMARY_STUDIES}}

If no studies are listed above, this is the first derivation for this parameter.

**ADDITIONAL RESTRICTION:** Do NOT use any sources that are already cited or referenced in the parameter descriptions, species descriptions, or other context information in the MODEL_CONTEXT section below. These sources were used to define the model structure, not to estimate this specific parameter value. You must find NEW, INDEPENDENT sources for your parameter estimation.

For this parameter, you must:

---

## Pre-Extraction Model Equation Analysis (DO THIS FIRST)

**Before searching for literature**, analyze the model equation from MODEL_CONTEXT to understand exactly what you're estimating.

**Step 1: Write out the full rate expression**

Find where this parameter appears in `reactions_and_rules`. Write out the complete term:
- Rate expression: `[parameter] * [species] * [modulators]`
- Note all Hill functions, saturation terms, and co-factors

**Step 2: Identify the parameter's mathematical role**

| Role | Characteristics | Data Implications |
|------|-----------------|-------------------|
| **Maximum rate (kmax)** | Multiplies the entire term; rate when modulators = 1 | Need data at saturating conditions |
| **Half-saturation (K50)** | Appears in denominator of Hill/Michaelis term | Need dose-response data |
| **Scaling factor** | Converts between units or compartments | Need matched-condition data |
| **Inhibition constant** | Reduces rate; appears in (1 - H_xxx) terms | Need inhibitor dose-response |

**Step 3: Determine what experimental conditions match this role**

- For **kmax**: Studies with saturating stimulus/substrate
- For **K50**: Studies with dose-response curves near the half-point
- For **baseline rates**: Untreated/control conditions
- For **inhibition**: Graded inhibitor concentrations

**Step 4: Document in `mathematical_role` field (EXPANDED)**

Your `mathematical_role` should include:
1. The parameter's function in the equation
2. What this implies for appropriate data sources
3. Any gotchas (e.g., "model assumes saturating cytokine, but most studies use sub-saturating doses")

**Note on saturation/modulation terms:** When your derivation assumes model saturation or modulation terms equal 1 (e.g., Hill functions = 1, volume terms = max, exclusion factors = 0), explicitly document this in assumptions and note the derived value is an "effective baseline" that may not match literature values for the isolated mechanistic rate.

**Example analysis for k_M1_pol:**
```
Equation: k_M1_pol * Mac_M2 * (IL12/(IL12+IL12_50) + IFNg/(IFNg+IFNg_50))

Role: Maximum M2→M1 polarization rate when IL-12 OR IFNγ is saturating

Data implications:
- Need studies with HIGH IL-12 or IFNγ (>> IL12_50 or IFNg_50)
- Studies with CD40 agonists measure a DIFFERENT pathway
- Time-to-effect at saturating cytokine → k = ln(2)/t_half

Gotcha: Most studies use sub-saturating cytokine or different stimuli
```

This analysis should guide your literature search and help identify mechanism mismatches early.

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
    half_life = inputs['half_life']  # Pint Quantity, e.g., <Quantity(5, 'hour')>

    N, rng = 10000, np.random.default_rng(42)
    # Extract magnitude for distribution, reattach units immediately
    samples = rng.lognormal(np.log(half_life.magnitude), 0.3, size=N) * half_life.units
    k = (np.log(2) / samples).to(1 / ureg.day)  # Pint handles unit conversion

    return {
        'median_param': np.median(k),
        'iqr_param': np.percentile(k, 75) - np.percentile(k, 25),
        'ci95_param': [np.percentile(k, 2.5), np.percentile(k, 97.5)]
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

## Sanity Check (MANDATORY)

After deriving your parameter, verify it produces realistic biology:

1. **Forward prediction**: Use your parameter to predict an observable (e.g., cell density, concentration)
2. **Compare to literature**: Does the prediction match known {{CANCER_TYPE}} values?
3. **Cross-parameter check**: Compare with structurally similar parameters (e.g., CD8 vs Treg trafficking rates should be within ~10×)

If prediction is >10× off from literature or cross-parameter ratio is >100×, re-examine your derivation before proceeding.

---

## Mechanism-to-Model Alignment

**CRITICAL:** Before using a data source, verify that the **biological driver/stimulus** in the study matches the driver in the model's equation.

**Step 1: Analyze the MODEL_CONTEXT reaction/rule**
- What species DRIVE the reaction? (e.g., IL-12, IFNγ, TGFβ, IL-10)
- What species are AFFECTED? (e.g., Mac_M1 → Mac_M2)
- What modulatory terms appear? (e.g., Hill functions, saturation terms)

**Step 2: Check each candidate source for mechanism alignment**

| Alignment Level | Example | Action |
|-----------------|---------|--------|
| **Exact match** | Model uses IL-12; study uses IL-12 | Ideal - use directly |
| **Same pathway** | Model uses IL-12; study uses IFNγ (both JAK-STAT) | Good - minor uncertainty |
| **Different pathway, same outcome** | Model uses IL-12; study uses CD40 agonist (NF-κB pathway) | Requires justification + widened uncertainty |
| **Different mechanism entirely** | Model uses cytokine-driven; study uses mechanical/hypoxia | Avoid unless no alternatives |

**If mechanism differs from model:**
1. **Add explicit assumption:** "ASSUMPTION N: [Source stimulus] kinetics approximate [model driver] kinetics because [biological justification]"
2. **Widen the mapping uncertainty:** Use broader bounds (e.g., 0.3–2.0× instead of 0.5–1.5×) to account for pathway differences
3. **Use `regimen_match` to penalize:** Set to 0.65 or lower for different-pathway sources
4. **Document in `key_study_limitations`:** Explain how mechanism differences might bias the estimate

**Example of problematic mismatch (avoid this pattern):**
- Model equation: `k_M1_pol * Mac_M2 * (IL12/(IL12+IL12_50) + IFNg/(IFNg+IFNg_50))`
- Source: CD40 agonist study measuring M2→M1 repolarization time
- Problem: CD40 signals through NF-κB/TRAF; IL-12/IFNγ signal through JAK-STAT. Kinetics may differ substantially.
- If used anyway: Add assumption justifying equivalence, widen uncertainty bounds, reduce `regimen_match` to ≤0.65

### Starting State Consistency (for A→B transitions)

For state transition parameters (polarization, differentiation, exhaustion), verify sources measure transitions from the **same starting state** as the model.

| Scenario | Risk | Action |
|----------|------|--------|
| All sources match model (e.g., all M1→M2) | Low | Pool directly |
| Mixed starting states (M0→M2 + M1→M2) | High | Prefer model-matching sources; weight others lower; add assumption |
| Opposite direction (M2→M1 for M1→M2 param) | Invalid | Do not use |

If sources measure A0→B (naive differentiation) but model uses A1→B (repolarization), reduce `regimen_match` to ≤0.55 and add +50% CI inflation. These are distinct processes with potentially different kinetics.

---

## Cross-Indication Data Handling

When {{CANCER_TYPE}}-specific quantitative data is unavailable, you may use cross-indication sources **with appropriate penalties and uncertainty inflation**.

**Indication Match Scoring (use these values for `indication_match`):**

| Source Type | indication_match | CI Inflation | Required Documentation |
|-------------|------------------|--------------|------------------------|
| {{CANCER_TYPE}} human (tumor tissue/TILs) | 1.0 | None | Standard |
| {{CANCER_TYPE}} human (peripheral blood) | 0.85–0.90 | None | Note compartment difference |
| Related adenocarcinoma (e.g., CRC, lung adeno) | 0.70–0.80 | +25% CI width | Justify similarity in assumptions |
| Other solid tumor (HNSCC, melanoma, RCC) | 0.50–0.65 | +50% CI width | Explicit assumption required |
| Healthy donor / non-cancer human | 0.35–0.50 | +75–100% CI width | Strong justification + flag in limitations |
| Cell line only (no primary cells/tissue) | 0.25–0.40 | +100% CI width | Consider rejecting; last resort only |
| **Non-human species (mouse, rat)** | Multiply above `indication_match` by 0.6–0.8 | +50–100% CI width | Explicit species-transfer assumption required |

For peripheral blood sources: note that TME conditions (hypoxia, TGF-β, dense stroma) often INCREASE immunosuppressive cell potency relative to blood. Consider asymmetric uncertainty (wider toward TME-enhanced values).

**When using non-{{CANCER_TYPE}} numeric anchors:**

1. **Separate mechanistic support from quantitative anchor:**
   - State clearly: "Numeric values from [Source A] (indication_match = X)"
   - If {{CANCER_TYPE}} studies confirm the mechanism qualitatively: "{{CANCER_TYPE}} relevance supported by [Source B] (qualitative confirmation only)"

2. **Inflate uncertainty bounds:**
   - Multiply the mapping factor range by the CI inflation factor
   - Example: Normal range 0.5–1.5× becomes 0.25–2.25× for healthy donor data (+100%)

3. **Add mandatory assumption:**
   - "ASSUMPTION N: Cross-indication transfer from [indication] to {{CANCER_TYPE}} is valid because [specific biological justification]. Uncertainty inflated by [X]% to account for potential indication-specific differences."

4. **Reduce `overall_confidence`:**
   - Subtract 0.10–0.15 from what it would otherwise be for cross-indication numeric anchors

---

## Experimental Confound Identification

Flag conditions that may bias the extracted value:

| Confound | Example | Impact |
|----------|---------|--------|
| Concurrent treatment | Gemcitabine + IO study | Alters baseline immune state |
| Non-physiological dose | 10× clinical concentration | Over/underestimates effect |
| Model system mismatch | Mouse orthotopic for human param | May not translate |

Document confounds in `key_study_limitations` and inflate uncertainty accordingly.

**Source quality minimum:** Use only peer-reviewed literature or authoritative medical references (e.g., Harrison's, established physiology texts) for numeric values. Consumer health websites, Wikipedia, and non-peer-reviewed sources are not acceptable even for "standard" values like blood volume.

---

**Example (K_T_Treg extraction):**
- Numeric anchor: Healthy donor Treg suppression assay (22–77% inhibition)
- indication_match: 0.45 (not 0.65)
- CI inflation: +75% (widen mapping bounds)
- Required: Assumption citing PDAC Treg studies showing similar suppressive phenotype

---

## Input Classification and Assumed Values

**Every input in the `inputs` list must be classified by its evidence basis.**

Use the `source_ref` field to indicate the input type:

| Input Type | source_ref Value | Description | Uncertainty Handling |
|------------|------------------|-------------|----------------------|
| **Direct measurement** | Study citation (e.g., `"Smith2022_JCI"`) | Value extracted from a specific study | Use reported uncertainty (SD, IQR, range) |
| **Literature consensus** | `"Multiple_sources"` or list citations | Widely accepted value from multiple sources | Combine uncertainties; note range in description |
| **Assumed/estimated** | `"ASSUMPTION"` | Value not directly measured; based on reasoning | **Use WIDE bounds** (log-uniform preferred); flag in limitations |
| **Computational** | `"Computation"` | Seeds, draw counts, technical parameters | N/A - not propagated |

For assumed values, state whether the range is constrained by ANY measurement (even indirect) or is purely speculative. If purely speculative, use log-uniform with ≥10-fold range.

**When an assumed input dominates uncertainty:**

This is a critical quality issue. If >50% of your CI width comes from an assumed (not measured) input:

1. **Flag prominently in `key_study_limitations`:**
   - "The assumed [parameter name] (range: X–Y) dominates uncertainty; no {{CANCER_TYPE}}-specific measurements available"

2. **Use conservative (wide) bounds:**
   - Prefer log-uniform distributions for assumed values spanning >2× range
   - Default to at least 4-fold range (e.g., 0.5–2 days) unless literature constrains tighter

3. **Reduce `overall_confidence` by 0.10–0.15:**
   - An estimate dominated by assumptions is less reliable than one anchored to data

4. **Consider whether the parameter can be estimated at all:**
   - If the assumed input is completely unconstrained, state this clearly
   - Better to report "insufficient data for reliable estimate" than to generate a false-precision prior

**Example (q_CD8_T_in extraction):**
```yaml
inputs:
  - name: tau_residence_time
    value: 1.5
    units: day
    source_ref: "ASSUMPTION"  # <-- Clearly flagged
    description: |
      Assumed intratumoral CD8+ residence time. No PDAC-specific measurements exist.
      Range 0.5–4 days based on cross-indication intravital imaging.
```
In `key_study_limitations`: "Residence time (0.5–4 days) is assumed, not measured, and contributes ~80% of CI width. PDAC-specific residence time data would substantially improve this estimate."

---

## Proxy and Surrogate Measurements

When the measured quantity is not identical to the model parameter, you are using a **proxy**. This is common but requires explicit documentation and uncertainty inflation.

**Common proxy relationships in QSP:**

| Measured (Proxy) | Model Parameter | Relationship Strength | Typical Uncertainty |
|------------------|-----------------|----------------------|---------------------|
| Proliferation inhibition | Cytotoxic killing | Moderate | ±50% |
| Endothelial adhesion | Extravasation rate | Weak-moderate | ±100% |
| mRNA expression | Protein concentration | Variable | ±50–200% |
| In vitro half-life | In vivo clearance | Weak | ±100–300% |
| Time-to-marker-change | Rate constant | Moderate | ±50% |
| Cell death (apoptosis) | Exhaustion/dysfunction | Weak | ±100% — death ≠ survival with dysfunction |
| Marker expression timing | Functional state transition | Moderate | ±50% — markers lag functional changes |

**Note on adhesion → extravasation proxies:** Literature reports only 10–50% of firmly adhered leukocytes successfully transmigrate. Unless measuring actual transmigration, apply a transmigration efficiency factor (default: 0.1–0.5) and document as a separate assumption.

**Document the proxy chain explicitly:**

For each proxy relationship, trace the logical chain from measurement to parameter:

```
Measured: [what was actually measured in the study]
    ↓ Assumption: [relationship/conversion]
Intermediate: [derived quantity, if any]
    ↓ Assumption: [relationship/conversion]
Parameter: [model parameter being estimated]
```

**Example (q_Treg_T_in):**
```
Measured: Treg adhesion to tumor endothelium (5.2%)
    ↓ Assumption: Adhesion fraction ≈ per-pass extravasation probability
Parameter: q_Treg_T_in (transport rate constant)
```
This is a WEAK proxy: not all adherent cells extravasate (some detach). Add ±100% uncertainty.

**Example (K_T_Treg):**
```
Measured: T cell proliferation inhibition at 1:4 Treg:CD8 ratio
    ↓ Assumption: Proliferation inhibition ∝ cytotoxic killing inhibition
Parameter: K_T_Treg (killing rate dependence on Treg ratio)
```
This is a MODERATE proxy: both reflect suppressed T cell function, but mechanisms differ.

**Required actions when using proxies:**

1. **Document the proxy chain** in `derivation_explanation` (as shown above)

2. **Add explicit assumption for each proxy step:**
   - "ASSUMPTION N: [Proxy measure] approximates [target quantity] because [mechanistic justification]"

3. **Inflate uncertainty for proxy relationships:**
   - Each proxy step adds uncertainty
   - Weak proxy: multiply CI width by 1.5–2.0×
   - Moderate proxy: multiply CI width by 1.2–1.5×
   - Strong proxy: no additional inflation needed

4. **Adjust `biomarker_population_match` based on proxy strength:**
   - Direct measurement of model species: 1.0
   - Strong proxy (validated surrogate): 0.85
   - Moderate proxy: 0.65–0.75
   - Weak proxy with known confounders: 0.45–0.55

5. **Flag in `key_study_limitations`:**
   - "Estimate relies on [proxy] as surrogate for [parameter]; the proxy relationship adds uncertainty because [specific concern]"

---

## Direct Measurements vs. Constructed Derivations

**Always search for direct measurements before constructing a derived estimate.**

### Derivation Hierarchy (prefer higher levels)

| Level | Description | Example | Reliability |
|-------|-------------|---------|-------------|
| **1. Direct measurement** | Study explicitly measures the parameter | "Kill probability per contact was 15%" | ⭐⭐⭐⭐⭐ |
| **2. Simple conversion** | One-step unit conversion from measured value | k = ln(2) / measured_half_life | ⭐⭐⭐⭐ |
| **3. Derived from related measurements** | Combine 2-3 measured quantities | rate = concentration / volume / time | ⭐⭐⭐ |
| **4. Constructed from assumed relationships** | Requires assumptions about biological mechanisms | "Time X ÷ duration Y = number of events" | ⭐⭐ |
| **5. Assumed with literature bounds** | No measurement; informed guess | "Based on similar systems, assume 0.5–4 days" | ⭐ |

### Before Using a Constructed Derivation (Level 4+)

**Ask: "Has anyone directly measured this parameter?"**

Many QSP parameters have been measured in imaging, flow cytometry, or kinetic studies — but these measurements may be published under different terminology or in specialized fields (biophysics, immunology methods papers, intravital imaging studies).

**Search strategies for direct measurements:**
- Search for the biological process + "quantitative" or "kinetics"
- Search for imaging modalities: "intravital," "two-photon," "live-cell imaging"
- Search for method papers in the relevant field
- Check reviews that compile parameter values across studies

### Warning Signs of Problematic Constructed Derivations

**🚩 Red flags that your derivation may be conceptually flawed:**

1. **Dividing unrelated quantities:** If your derivation involves dividing quantity A by quantity B to get a count or probability, verify this relationship is biologically real, not just dimensionally convenient.

2. **Assuming additivity without evidence:** "Cumulative X leads to outcome Y" requires evidence that the process is actually additive (not threshold-based, cooperative, or saturating).

3. **Converting snapshots to rates without timescale data:** A prevalence or fraction (e.g., "40% exhausted") only becomes a rate if you have a well-characterized exposure time AND the system is at steady state.

4. **Mixing incompatible data sources:** Combining a "time to effect" from system A with a "duration per event" from system B assumes the underlying processes are equivalent.

**When red flags are present:**

1. **Search harder for direct measurements** — they often exist
2. **If direct data truly doesn't exist**, document the conceptual assumptions explicitly
3. **Widen uncertainty substantially** (2-3× wider CI) to reflect model uncertainty
4. **Reduce `overall_confidence`** by 0.15-0.20
5. **Flag in `key_study_limitations`**: "Derivation assumes [relationship] which has not been directly validated"

### Example: Good vs. Problematic Derivation

**Problematic approach (avoid):**
```
Parameter: p_kill_per_contact (probability)
Derivation: "8 hours to kill" ÷ "15 min per contact" = 32 contacts → p = 1/32 = 0.03
Problem: Assumes killing requires accumulating contact time, which may not match biology
```

**Better approach:**
```
Parameter: p_kill_per_contact (probability)
Derivation: Direct measurement from intravital imaging: "15% of CTL-tumor contacts
            resulted in target cell death" → p = 0.15
Source: Studies that tracked individual contact outcomes
```

**If direct measurement unavailable:**
```
Parameter: p_kill_per_contact (probability)
Derivation: From bulk cytotoxicity: "50% killing at 10:1 E:T after 4h"
            → Estimate contact rate from density + search volume (separate data)
            → Back-calculate p_kill that produces observed bulk killing
Caveat: Multiple parameters involved; p_kill estimate is coupled to contact rate assumption
```

---

## Experimental Documentation

6. **Study overview (1-2 sentences):** WHAT parameter is being measured in {{CANCER_TYPE}}, WHY it's biologically relevant to {{CANCER_TYPE}}, and the overall approach
7. **Study design (1-2 sentences):** HOW the measurement was performed (assay type, sample size, key methods). Note if data is from {{CANCER_TYPE}} patients or cross-indication.
8. **Key assumptions (list):** 3-5 critical assumptions only (e.g., distributional assumptions, model choices, data quality). Each assumption should have a number and text. **If using non-{{CANCER_TYPE}} data, include an assumption justifying the cross-indication transfer.** Do NOT include trivial assumptions like "bootstrap samples are independent" or "conversion factors are standard".
9. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (3-6 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
10. **Key study limitations:** List critical limitations and their specific impact on reliability. **If not {{CANCER_TYPE}}-specific, note how cancer type differences might affect the parameter value.**

---

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

## Quick Checklist

Before submitting, verify:
- [ ] **{{CANCER_TYPE}}-specific sources** prioritized; cross-indication use justified in assumptions
- [ ] **`derive_parameter(inputs, ureg)`** returns median_param, iqr_param, ci95_param as Pint Quantities
- [ ] **All inputs** have source_ref, value_table_or_section, and value_snippet
- [ ] **Assumptions** numbered and referenced in derivation_explanation as "ASSUMPTION N: ..."
- [ ] **Citations** are real, accessible publications (no hallucinated DOIs)
- [ ] **Weights** follow rubric tables (especially `indication_match` for cross-indication sources)

---

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

---

Extract parameter metadata for **{{CANCER_TYPE}}** following all requirements above.

**Formatting:** Use `\n` for line breaks. `derivation_code` is raw Python (no ```python wrapper). Numbers as numbers, not strings. Every source_ref must have a corresponding source entry.
