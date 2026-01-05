# Calibration Target Extraction

Extract **raw observables** from scientific literature for QSP model calibration.

**Cancer type:** {{CANCER_TYPE}}

**Observable:** {{OBSERVABLE_DESCRIPTION}}

---

## What is a Calibration Target?

A biological observable measured in a **specific experimental scenario**, used to calibrate QSP model parameters via Bayesian inference.

**Critical concept:** Each observable has an **experimental context** (species, indication, compartment, system, treatment) that may differ from the **model context**.

**Your task:**
1. **Find the closest match** to the model context (specified below) when searching literature
2. **Document the experimental context accurately** - mismatches will be quantified later via formal distance metrics
3. Prefer exact matches (same species, indication, compartment, system), but accept reasonable mismatches when necessary

Example: For a human PDAC model, prefer human PDAC data > mouse PDAC data > related adenocarcinoma data > other solid tumors.

---

## Model Context (Target to Match)

**Species:** {{MODEL_SPECIES}}
**Indication:** {{MODEL_INDICATION}}
**Compartment:** {{MODEL_COMPARTMENT}}
**System:** {{MODEL_SYSTEM}}
**Treatment history:** {{MODEL_TREATMENT_HISTORY}}
**Stage/burden:** {{MODEL_STAGE_BURDEN}}

Find observables that match this context as closely as possible. Document the actual experimental context in your extraction.

---

## Task

{{PRIMARY_SOURCE_TITLE}}
1. Extract measurement with uncertainty (mean ± SD/SE, 95% CI, IQR, or range)
2. Specify the experimental scenario (interventions + measurement timing/location)
3. Document experimental context (species, compartment, system, treatment history, stage)
4. Provide verbatim text snippets for all extracted values

---

## Critical Requirements

### Scenario Specification

**Interventions** (may be empty for untreated state):
- Drug dosing: specify agent, dose + units, schedule timepoints, patient weight/BSA if needed
- Surgical resection: timing, fraction removed, affected species
- Leave empty list for baseline/natural measurements

**Measurements** (at least ONE required, all biomarker-triggered):

All measurements MUST be **biomarker-triggered** - anchored to an observable biological state, NOT arbitrary time points. This ensures:
- **Biological interpretability**: "When tumor reaches resectable size" vs. vague "at diagnosis"
- **Simulation reproducibility**: Unambiguous instructions for when to query the model
- **Reality**: There is no absolute "time zero" in biology

**Biomarker trigger selection (CRITICAL):**

⏰ **`trigger_species`** determines **WHEN** to measure
📊 **`measurement_code`** determines **WHAT** to measure

**These are usually DIFFERENT species!**

- **Use clinical/standard triggers**: Tumor burden, treatment response, clinical milestones
- **DON'T use the measured observable as the trigger**: If measuring a cytokine concentration, trigger on tumor size or treatment timepoint, not the cytokine level itself
- **Clinical logic**: ⏰ "When tumor reaches 500 mm³" → 📊 "measure CD8 density" ✓
- **Wrong (circular)**: ⏰ "When CD8 exceeds threshold" → 📊 "measure CD8" ✗

**Most common triggers**:
1. Tumor burden (V_T.C1) at resection/diagnosis/establishment
2. Biomarker concentration reaching clinical threshold
3. Treatment milestone (post-infusion, post-resection)

**Example 1: Simple identity mapping (trigger in natural units)**
```yaml
trigger_species: V_T.TGFb  # TGF-beta concentration
threshold_conversion_code: |
  def compute_threshold_value(species_dict, inputs, ureg):
      """Identity mapping - threshold in trigger's natural units."""
      return species_dict['V_T.TGFb']
threshold: 100.0
threshold_units: nanomolarity
threshold_input_name: high_tgfb_threshold
comparison: ">"
timepoints: [0.0]
inputs:
  - name: high_tgfb_threshold
    value: 100.0
    units: nanomolarity
    description: "High TGF-beta threshold from cohort stratification"
    source_ref: smith_2020
    value_snippet: "High TGF-β patients defined as >100 ng/mL serum concentration"
```

**Example 2: Unit conversion (tumor burden trigger with cells → volume)**
```yaml
trigger_species: V_T.C1  # Tumor cell count
threshold_conversion_code: |
  def compute_threshold_value(species_dict, inputs, ureg):
      """Convert tumor cells to volume for threshold comparison."""
      tumor_cells = species_dict['V_T.C1']
      cell_density = inputs['cell_packing_density']
      volume = tumor_cells / cell_density
      return volume.to(ureg.mm**3)
threshold: 500.0
threshold_units: millimeter**3
threshold_input_name: resection_tumor_volume
comparison: ">"
inputs:
  - name: resection_tumor_volume
    value: 500.0
    units: millimeter**3
    description: "Tumor volume at time of resection"
    source_ref: smith_2020
    value_snippet: "Tumors were resected at mean volume of 500 mm³"
  - name: cell_packing_density
    value: 1.0e9
    units: cell / millimeter**3
    description: "Typical solid tumor cell packing density"
    source_ref: modeling_assumption
    value_snippet: null
```

**Threshold source tracking:**

**⚠️ CRITICAL RULE: Threshold value MUST come from the SAME paper as the calibration target.**

The threshold defines WHEN the observable was measured. It is part of the experimental context, not a modeling choice.

**What MUST come from the primary paper:**
- ✅ Threshold values (tumor size at resection, enrollment criteria, cohort stratification cutoffs)

**What CAN use modeling_assumption:**
- ✅ Conversion factors (cell packing density, geometric constants)
- ✅ Universal constants (π, 2, percentiles)

**Extract threshold from the primary source:**

**Direct statements:**
- "Tumors resected at mean volume of 500 mm³" → extract 500 mm³ from this paper
- "Biopsies at 1 cm diameter" → extract 1 cm from this paper
- "High IL-2 patients (>100 pg/mL)" → extract 100 pg/mL from this paper

**Temporal phrases - find threshold in the same paper:**
- "At clinical presentation" → search this paper for patient demographics (median tumor size at presentation)
- "At diagnosis" → extract from this paper's cohort characteristics table
- "At enrollment" → find in this paper's methods (enrollment criteria)

**If threshold not in paper:**
The paper may not be suitable for extraction. The threshold is part of the experimental context and must be documented in the primary source.

**Remember the rule:**
- ❌ Threshold values → NEVER modeling_assumption
- ✅ Conversion factors → CAN use modeling_assumption

**Notes:**
- **Derivatives**: Use central differences from timepoints. Don't assume analytical derivatives exist.
- **Multiple timepoints**: Use `timepoints: [-7.0, 0.0, 7.0]` for kinetics around trigger event.

### Distribution Code

**CRITICAL:**
- **Extract biological/experimental values via inputs** with source traceability (measurements, conversion factors, reference values)
- **Universal constants are OK as literals**: percentiles (2.5, 25, 75, 97.5), mathematical constants (2, π), MC sample sizes (10000)
- **Use MC methods** (parametric bootstrap), NOT analytical approximations
- **Use Pint units** - inputs are pre-converted Pint Quantities, return Pint Quantities
- Function signature: `derive_distribution(inputs, ureg)` returns dict with `median_obs`, `iqr_obs`, `ci95_obs`

**GOLDEN RULE: Keep values tethered to their units throughout calculations.**

Let Pint propagate units through your entire calculation. This catches dimensional errors automatically. **Never extract `.magnitude` early - only when absolutely necessary (e.g., for lognormal distribution parameters).**

**Good example (fraction with unit validation):**
```python
def derive_distribution(inputs, ureg):
    # Inputs are already Pint Quantities - use directly
    treg_fraction_mean = inputs['treg_fraction_mean']  # e.g., 0.15 dimensionless
    treg_fraction_sd = inputs['treg_fraction_sd']  # e.g., 0.03 dimensionless
    n_samples = int(inputs['n_mc_samples'].magnitude)  # Only extract for integer conversion

    # Bounds stay as Quantities
    lower_bound = inputs['lower_bound_fraction']  # 0.0 dimensionless
    upper_bound = inputs['upper_bound_fraction']  # 1.0 dimensionless

    # Monte Carlo with truncated normal - extract magnitude only for distribution
    rng = np.random.default_rng(42)
    samples = rng.normal(
        treg_fraction_mean.magnitude,  # Extract for distribution params
        treg_fraction_sd.magnitude,
        size=n_samples
    ) * treg_fraction_mean.units  # Reattach units immediately!

    # Clip with Pint Quantities - units validated automatically
    samples = np.clip(samples, lower_bound, upper_bound)

    # Return Quantities - validator checks dimensionality
    return {
        'median_obs': np.median(samples),
        'iqr_obs': np.percentile(samples, 75) - np.percentile(samples, 25),
        'ci95_obs': [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]
    }
```

### Source Requirements

- **Primary source (singular)**: One paper, real DOI that resolves
- **No reuse**: Avoid studies already used for this observable: {{USED_PRIMARY_STUDIES}}
- **Verbatim snippets**: Exact quotes containing values (automatically verified)
- **Secondary sources**: Reference values, conversion factors (can be multiple). Threshold and calibration target MUST come from the same primary source.

---

## Source Hierarchy

**Prefer (in order):**
1. Same indication + compartment + system (e.g., PDAC tumor IHC)
2. Same indication + adjacent compartment OR different modality
3. Related indication + same compartment (document mismatch justification)
4. Pan-cancer or non-cancer reference (last resort, document limitations)

**Never acceptable:**
- In vitro data for clinical observables
- Pure assumptions without literature anchor

---

## Context

**Available model species:**
{{MODEL_SPECIES_WITH_UNITS}}

---

Generate calibration target metadata following all requirements above.
