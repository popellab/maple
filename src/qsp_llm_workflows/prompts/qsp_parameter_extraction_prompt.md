# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model **for {{CANCER_TYPE}}**.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data **specific to {{CANCER_TYPE}}**.

**Purpose:** These parameter extractions will be used as **informative priors for simulation-based inference (SBI)** during QSP model calibration for {{CANCER_TYPE}}. The extracted distributions (median, IQR, 95% CI) will inform Bayesian parameter estimation, helping constrain the parameter space during model fitting to {{CANCER_TYPE}} experimental data.

**CRITICAL:** Prioritize {{CANCER_TYPE}}-specific data. See "Cross-Indication Data Handling" section for scoring and uncertainty inflation rules.

**IMPORTANT:** The following primary studies have already been used for this parameter (same name and context). Do NOT reuse these studies - find independent sources instead:

{{USED_PRIMARY_STUDIES}}

If no studies are listed above, this is the first derivation for this parameter.

**ADDITIONAL RESTRICTION:** Do NOT use any sources that are already cited or referenced in the parameter descriptions, species descriptions, or other context information in the MODEL_CONTEXT section below. These sources were used to define the model structure, not to estimate this specific parameter value. You must find NEW, INDEPENDENT sources for your parameter estimation.

---

## Uncertainty Inflation Reference

Use this table when determining how much to widen confidence intervals. Effects are cumulative when multiple factors apply.

| Factor | CI Width Multiplier | When to Apply |
|--------|---------------------|---------------|
| **Cross-indication** | | |
| Related adenocarcinoma | 1.25× | CRC, lung adeno for PDAC |
| Other solid tumor | 1.5× | HNSCC, melanoma, RCC |
| Healthy donor / non-cancer | 1.75–2.0× | No cancer context |
| Cell line only | 2.0× | Last resort |
| Non-human species | 1.5–2.0× | Add to above |
| **Mechanism mismatch** | | |
| Different pathway, same outcome | 1.25–1.5× | e.g., CD40 vs IL-12 |
| Starting state mismatch (A0→B vs A1→B) | 1.5× | Naive vs repolarization |
| **Measurement gaps** | | |
| Weak proxy | 1.5–2.0× | Adhesion→extravasation, mRNA→protein |
| Moderate proxy | 1.2–1.5× | Proliferation inhibition→killing |
| Composite measurement | 1.5–2.0× | Total apoptosis→intrinsic death |
| Snapshot-to-rate (non-equilibrium) | 1.25–1.5× | Resected tumor samples |
| **Derivation quality** | | |
| Constructed (Level 4+) | 2.0–3.0× | Assumed biological relationships |
| Assumed input dominates (>50% CI) | Use log-uniform | Mark as low confidence |

---

## Pre-Extraction Model Equation Analysis (DO THIS FIRST)

**Before searching for literature**, analyze the model equation from MODEL_CONTEXT to understand exactly what you're estimating.

**Step 1: Write out the full rate expression** from `reactions_and_rules`:
- Rate expression: `[parameter] * [species] * [modulators]`
- Note all Hill functions, saturation terms, and co-factors

**Step 2: Identify the parameter's mathematical role**

| Role | Characteristics | Data Implications |
|------|-----------------|-------------------|
| **Maximum rate (kmax)** | Multiplies entire term; rate when modulators = 1 | Need saturating conditions |
| **Half-saturation (K50)** | Appears in denominator of Hill term | Need dose-response data |
| **Scaling factor** | Converts between units or compartments | Need matched-condition data |
| **Inhibition constant** | Appears in (1 - H_xxx) terms | Need inhibitor dose-response |

**Step 3: Document in `mathematical_role` field:**
1. The parameter's function in the equation
2. What this implies for appropriate data sources
3. Any gotchas (e.g., "model assumes saturating cytokine, but most studies use sub-saturating doses")

**Note:** When assuming saturation/modulation terms = 1, document this in assumptions and note the derived value is an "effective baseline."

---

## Monte Carlo Parameter Estimation

1. **Structured inputs:** Define all input values in the `inputs` list with source references
2. **Function-based code:** Provide Python code as a `derive_parameter(inputs, ureg)` function
3. **Bootstrap preferred:** Use bootstrap resampling when raw data available
4. **Uncertainty propagation:** Incorporate ALL sources of uncertainty (measurements, composite parameters, unit conversions, model assumptions)
5. **Standard units:** Use standard unit formats (e.g., "1/day", "nM", "mg/L", "dimensionless")
6. **Required outputs:** Function must return dict with:
   - `median_param`: Median of Monte Carlo draws
   - `iqr_param`: Interquartile range (Q3 - Q1)
   - `ci95_param`: 95% percentile CI as [lower, upper]

**Use Pint for unit-safe calculations.** Return Pint Quantities with units matching the parameter's declared units.

**GOLDEN RULE:** Sampling strips units - reattach IMMEDIATELY after. Then units propagate through numpy operations (median, percentile, etc.).

```python
import numpy as np

def derive_parameter(inputs, ureg):
    half_life = inputs['half_life']  # Pint Quantity
    N, rng = 10000, np.random.default_rng(42)
    samples = rng.lognormal(np.log(half_life.magnitude), 0.3, size=N) * half_life.units
    k = (np.log(2) / samples).to(1 / ureg.day)
    return {
        'median_param': np.median(k),
        'iqr_param': np.percentile(k, 75) - np.percentile(k, 25),
        'ci95_param': [np.percentile(k, 2.5), np.percentile(k, 97.5)]
    }
```

**NumPy functions that work with Pint:** `mean`, `median`, `std`, `percentile`, `concatenate`, `sqrt`, `exp`, `log`

**Common unit aliases:** `nanomolarity` = `nanomolar`, `cell` = cell counts, `dimensionless` = unitless ratios

---

## Sanity Check (MANDATORY)

After deriving your parameter, verify it produces realistic biology:

1. **Forward prediction**: Use your parameter to predict an observable
2. **Compare to literature**: Does the prediction match known {{CANCER_TYPE}} values?
3. **Cross-parameter check**: Similar parameters should be within ~10× (e.g., CD8 vs Treg trafficking)

If prediction is >10× off from literature, re-examine your derivation before proceeding.

---

## Mechanism-to-Model Alignment

**CRITICAL:** Verify the **biological driver/stimulus** in the study matches the driver in the model's equation.

**Step 1:** Analyze the MODEL_CONTEXT reaction—what species DRIVE and are AFFECTED?

**Step 2:** Check each source for alignment:

| Alignment Level | Example | Action |
|-----------------|---------|--------|
| **Exact match** | Model uses IL-12; study uses IL-12 | Use directly |
| **Same pathway** | Model uses IL-12; study uses IFNγ (both JAK-STAT) | Minor uncertainty |
| **Different pathway** | Model uses IL-12; study uses CD40 (NF-κB) | Justify + widen CI (see table) |
| **Different mechanism** | Model uses cytokine; study uses mechanical | Avoid unless no alternatives |

**If mechanism differs:** Add assumption, widen CI per reference table, reduce `regimen_match` to ≤0.65.

### Starting State Consistency (for A→B transitions)

For state transitions, verify sources measure from the **same starting state** as the model:
- Mixed starting states (M0→M2 + M1→M2): Prefer model-matching; add assumption for others
- Opposite direction: Do not use
- A0→B for A1→B parameter: Reduce `regimen_match` to ≤0.55, apply 1.5× CI inflation

---

## Cross-Indication Data Handling

When {{CANCER_TYPE}}-specific data is unavailable, use cross-indication sources **with penalties**.

**Indication Match Scoring:**

| Source Type | indication_match | Required Documentation |
|-------------|------------------|------------------------|
| {{CANCER_TYPE}} human (tumor tissue/TILs) | 1.0 | Standard |
| {{CANCER_TYPE}} human (peripheral blood) | 0.85–0.90 | Note compartment difference |
| Related adenocarcinoma | 0.70–0.80 | Justify in assumptions |
| Other solid tumor | 0.50–0.65 | Explicit assumption required |
| Healthy donor / non-cancer | 0.35–0.50 | Strong justification + flag |
| Cell line only | 0.25–0.40 | Last resort only |
| Non-human species | Multiply above by 0.6–0.8 | Species-transfer assumption |

**Peripheral blood note:** TME conditions often INCREASE immunosuppressive potency vs blood. Consider asymmetric uncertainty toward TME-enhanced values.

**When using non-{{CANCER_TYPE}} numeric anchors:**
1. State clearly: "Numeric values from [Source A] (indication_match = X)"
2. Apply CI inflation per reference table
3. Add assumption: "ASSUMPTION N: Cross-indication transfer valid because [justification]. Uncertainty inflated by [X]%."
4. Reduce `overall_confidence` by 0.10–0.15

**Source quality minimum:** Use only peer-reviewed literature or authoritative references (Harrison's, established physiology texts). Consumer health websites and Wikipedia are not acceptable.

---

## Measurement Gaps and Proxies

When the measured quantity doesn't exactly match the model parameter, you must document and inflate uncertainty.

### Types of Measurement Gaps

**1. Proxy measurements** — measured quantity correlates with but isn't identical to parameter:

| Proxy → Parameter | Strength | Notes |
|-------------------|----------|-------|
| Proliferation inhibition → Cytotoxic killing | Moderate | |
| Endothelial adhesion → Extravasation rate | Weak | Only 10–50% of adhered cells transmigrate |
| mRNA → Protein concentration | Variable | |
| Time-to-marker → Rate constant | Moderate | Markers lag functional changes |
| Apoptosis → Exhaustion/dysfunction | Weak | Death ≠ survival with dysfunction |

**2. Composite/contaminated measurements** — measured quantity includes multiple sources but parameter represents one:
- Total apoptosis vs intrinsic death only
- Tumor volume vs cancer cell count (PDAC is 50–80% stroma)

**3. Snapshot-to-rate conversions** — converting prevalence to rate constant:
- Requires well-characterized exposure time AND steady-state assumption
- Resected tumors may not be at equilibrium; add assumption and widen CI

**4. Auxiliary parameter mismatches** — derivation requires intermediate value (apoptosis duration, exposure time) from different tissue/context

### Required Actions

1. **Document the proxy chain** in `derivation_explanation`:
   ```
   Measured: [what was measured]
       ↓ Assumption: [relationship]
   Parameter: [model parameter]
   ```

2. **Add explicit assumption** for each gap: "ASSUMPTION N: [Proxy] approximates [target] because [justification]"

3. **Apply CI inflation** per reference table (effects cumulative)

4. **Adjust `biomarker_population_match`:**
   - Direct measurement: 1.0
   - Strong proxy: 0.85
   - Moderate proxy: 0.65–0.75
   - Weak proxy: 0.45–0.55

5. **Flag in `key_study_limitations`**

---

## Input Classification

**Every input must be classified by evidence basis:**

| Input Type | source_ref Value | Uncertainty Handling |
|------------|------------------|----------------------|
| Direct measurement | Study citation | Use reported uncertainty |
| Literature consensus | `"Multiple_sources"` | Combine uncertainties |
| Assumed/estimated | `"ASSUMPTION"` | Wide bounds (log-uniform preferred) |
| Computational | `"Computation"` | N/A |

**When an assumed input dominates uncertainty (>50% of CI width):**
1. Flag in `key_study_limitations`
2. Use log-uniform with ≥4-fold range
3. Reduce `overall_confidence` by 0.10–0.15
4. Consider reporting "insufficient data" if completely unconstrained

**Auxiliary parameter validity:** Verify auxiliary parameters (apoptosis duration, exposure time) were measured in conditions matching your target tissue/context. If from heterogeneous sources, note in limitations and widen uncertainty.

---

## Direct vs. Constructed Derivations

**Always search for direct measurements before constructing derived estimates.**

### Derivation Hierarchy

| Level | Description | Reliability |
|-------|-------------|-------------|
| 1. Direct measurement | Study measures the parameter | ⭐⭐⭐⭐⭐ |
| 2. Simple conversion | k = ln(2) / half_life | ⭐⭐⭐⭐ |
| 3. Derived from measurements | Combine 2-3 measured quantities | ⭐⭐⭐ |
| 4. Constructed | Requires biological assumptions | ⭐⭐ |
| 5. Assumed | No measurement; informed guess | ⭐ |

**Search strategies for direct measurements:**
- Search biological process + "quantitative" or "kinetics"
- Search imaging: "intravital," "two-photon," "live-cell"
- Check reviews that compile parameter values

### Warning Signs (Level 4+ Derivations)

🚩 **Red flags:**
1. Dividing unrelated quantities for dimensional convenience
2. Assuming additivity without evidence
3. Converting snapshots to rates without timescale/equilibrium justification
4. Mixing incompatible data sources

**When red flags present:** Search harder for direct measurements. If unavailable, document assumptions, apply 2–3× CI inflation, reduce `overall_confidence` by 0.15–0.20.

---

## Experimental Documentation

6. **Study overview (1-2 sentences):** WHAT, WHY, and overall approach
7. **Study design (1-2 sentences):** HOW (assay, sample size, methods). Note if cross-indication.
8. **Key assumptions (list):** 3-5 critical assumptions only. Number each. Include cross-indication justification if applicable.
9. **Derivation explanation:** Step-by-step plain-language explanation. Reference assumptions as "ASSUMPTION N: ..."
10. **Key study limitations:** Critical limitations and specific impact on reliability.

---

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

## Quick Checklist

Before submitting, verify:
- [ ] **{{CANCER_TYPE}}-specific sources** prioritized; cross-indication justified
- [ ] **`derive_parameter(inputs, ureg)`** returns median_param, iqr_param, ci95_param as Pint Quantities
- [ ] **All inputs** have source_ref, value_table_or_section, and value_snippet
- [ ] **Assumptions** numbered and referenced as "ASSUMPTION N: ..."
- [ ] **Citations** are real, accessible publications
- [ ] **Weights** follow rubric tables

---

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

---

Extract parameter metadata for **{{CANCER_TYPE}}** following all requirements above.

**Formatting:** Use `\n` for line breaks. `derivation_code` is raw Python (no ```python wrapper). Numbers as numbers, not strings. Every source_ref must have a corresponding source entry.
