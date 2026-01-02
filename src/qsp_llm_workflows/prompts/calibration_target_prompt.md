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

The `biomarker_species` determines **WHEN** to measure, NOT **WHAT** to measure.

- **Use clinical/standard triggers**: Tumor burden, treatment response, clinical milestones
- **DON'T use the measured observable as the trigger**: If measuring a cytokine concentration, trigger on tumor size or treatment timepoint, not the cytokine level itself
- **Clinical logic**: "At resection when tumor reaches threshold volume, measure the observable" ✓
- **Wrong (circular)**: "When observable X exceeds threshold, measure observable X" ✗

**Most common trigger**: Tumor burden (V_T.C1) at resection/diagnosis/establishment.

**Example 1: Tumor burden trigger (most common - with unit conversion)**
```yaml
biomarker_species: V_T.C1  # Tumor cell count
threshold_computation_code: |
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

**CRITICAL: Extract threshold from paper, NOT assumptions.**

Look for explicit statements about when measurements were taken:
- "Tumors resected at mean volume of 500 mm³"
- "Biopsies taken when tumor reached 1 cm diameter"
- "Analysis performed in high IL-2 patients (>100 pg/mL)"
- "At clinical presentation" (then extract typical presentation volume/size)

If not explicitly stated, look for implicit information:
- Cohort characteristics (e.g., "newly diagnosed patients" → extract median tumor size at diagnosis)
- Experimental protocol details (e.g., "resectable tumors" → find typical resection criteria)
- Companion papers describing methods

Add threshold as an input with source tracking (`threshold_input_name` references it). Conversion factors (cell density, diameter→volume formulas) can use `modeling_assumption` as source.

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
