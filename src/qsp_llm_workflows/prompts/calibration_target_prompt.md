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

1. Find 1 peer-reviewed paper reporting this observable in {{CANCER_TYPE}}
2. Extract measurement with uncertainty (mean ± SD/SE, 95% CI, IQR, or range)
3. Specify the experimental scenario (interventions + measurement timing/location)
4. Document experimental context (species, compartment, system, treatment history, stage)
5. Provide verbatim text snippets for all extracted values

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

**Required fields for ALL measurements:**
- **timing_type**: Always `"biomarker_triggered"`
- **biomarker_species**: Model species to monitor (e.g., `"V_T.C1"` for tumor cells)
- **threshold**: Threshold value (MUST extract from paper - see threshold tracking requirements below)
- **threshold_units**: Units of threshold (must match paper's reported units)
- **threshold_input_name**: Name of input providing threshold with source tracking
- **comparison**: `">"` (exceeds threshold) or `"<"` (falls below threshold)
- **timepoints**: Days relative to trigger. `[0.0]` for single point, `[-1.0, 0.0, 1.0]` for derivatives
- **required_species**: Model species needed for measurement computation
- **computation_code**: Python function `compute_measurement(time, species_dict, ureg)` returning Pint Quantity

**Common biomarker choices:**

1. **Tumor burden** (most common for untreated/baseline measurements):

   *Simple case - threshold in natural units (cells):*
   ```yaml
   biomarker_species: V_T.C1  # Tumor cell count (natural units: cells)
   threshold_computation_code: |
     def compute_threshold_value(species_dict, inputs, ureg):
         # Identity mapping - use raw cell count
         return species_dict['V_T.C1']
   threshold: 5e8  # cells
   threshold_units: cell
   threshold_input_name: tumor_burden_at_resection
   comparison: ">"
   inputs:
     - name: tumor_burden_at_resection
       value: 5.0e8
       units: cell
       description: "Tumor cell count at resection"
       source_ref: smith_2020
       value_snippet: "Tumors resected at approximately 5×10⁸ cells"
   ```

   *Common case - paper reports volume, need conversion:*
   ```yaml
   biomarker_species: V_T.C1  # Tumor cell count
   threshold_computation_code: |
     def compute_threshold_value(species_dict, inputs, ureg):
         """Convert tumor cells to volume for threshold comparison."""
         tumor_cells = species_dict['V_T.C1']
         cell_density = inputs['cell_packing_density']
         volume = tumor_cells / cell_density
         return volume.to(ureg.mm**3)
   threshold: 500.0  # mm³ (from paper: "resection at 500 mm³")
   threshold_units: millimeter**3
   threshold_input_name: resection_tumor_volume
   comparison: ">"
   inputs:
     - name: resection_tumor_volume
       value: 500.0
       units: millimeter**3
       description: "Tumor volume at time of resection"
       source_ref: smith_2020
       value_table_or_section: "Methods"
       value_snippet: "Tumors were resected at mean volume of 500 mm³"
     - name: cell_packing_density
       value: 1.0e9
       units: cell / millimeter**3
       description: "Typical solid tumor cell packing density"
       source_ref: modeling_assumption
       value_snippet: null
   ```
   **Use when:** Paper states "at resection", "in established tumors", "at detectable disease"
   **Threshold source:** Extract from paper ("resection at mean tumor volume of 500 mm³") OR document as modeling assumption

2. **Treatment response** (tumor shrinkage):
   ```yaml
   biomarker_species: V_T.C1
   threshold_computation_code: |
     def compute_threshold_value(species_dict, inputs, ureg):
         """Compute tumor volume for response criteria."""
         tumor_cells = species_dict['V_T.C1']
         cell_density = inputs['cell_packing_density']
         return (tumor_cells / cell_density).to(ureg.mm**3)
   threshold: 250.0  # mm³ (50% of 500 mm³ baseline)
   threshold_units: millimeter**3
   threshold_input_name: partial_response_threshold
   comparison: "<"
   inputs:
     - name: partial_response_threshold
       value: 250.0
       units: millimeter**3
       description: "Tumor volume threshold for partial response"
       source_ref: recist_criteria
       value_table_or_section: "RECIST 1.1 guidelines"
       value_snippet: "Partial response defined as 50% reduction from baseline 500 mm³"
     - name: cell_packing_density
       value: 1.0e9
       units: cell / millimeter**3
       description: "Typical solid tumor cell packing density"
       source_ref: modeling_assumption
       value_snippet: null
   ```
   **Use when:** Paper states "at partial response", "when tumor shrinks below X mm³"

3. **Circulating biomarker level** (identity mapping):
   ```yaml
   biomarker_species: V_P.IL2  # Serum IL-2 (natural units: pg/mL)
   threshold_computation_code: |
     def compute_threshold_value(species_dict, inputs, ureg):
         # Identity mapping - use raw IL-2 concentration
         return species_dict['V_P.IL2']
   threshold: 100.0  # pg/mL
   threshold_units: picogram/milliliter
   threshold_input_name: high_il2_threshold
   comparison: ">"
   inputs:
     - name: high_il2_threshold
       value: 100.0
       units: picogram / milliliter
       description: "IL-2 concentration threshold for high expressors"
       source_ref: jones_2019
       value_table_or_section: "Results, patient stratification"
       value_snippet: "High IL-2 patients defined as >100 pg/mL serum concentration"
   ```
   **Use when:** Paper states "in high IL-2 patients", "when cytokine >100 pg/mL"

**Threshold specification rules:**

- **Always provide `threshold_computation_code`** - even for identity mappings (makes intent explicit)
- **Set `threshold_input_name`** - must reference an input in the `inputs` list
- **Referenced input must have matching value/units** - same as `threshold`/`threshold_units` fields
- **If paper reports threshold in biomarker's natural units**: Use identity mapping (return species_dict['...'])
- **If paper reports threshold in different units** (common for tumor size):
  - Provide conversion code: biomarker → threshold space
  - Reference conversion factors from `inputs` (with proper source tracking)
  - Set `threshold` and `threshold_units` to match paper's reported units

**Threshold source tracking (REQUIRED):**

**CRITICAL: The threshold value MUST be extracted from the paper, NOT assumed.**

- **ALWAYS extract threshold from paper** - look for explicit statements about when measurements were taken:
  - "Tumors resected at mean volume of 500 mm³"
  - "Biopsies taken when tumor reached 1 cm diameter"
  - "Analysis performed in high IL-2 patients (>100 pg/mL)"
  - "At clinical presentation" (then extract typical presentation volume/size)
- **NEVER use `modeling_assumption` for threshold values** - if the paper doesn't state the threshold explicitly:
  - Look for implicit information (e.g., "resectable tumors" → find typical resection criteria)
  - Use cohort characteristics (e.g., "newly diagnosed patients" → extract median tumor size at diagnosis from results)
  - Search for companion papers describing the experimental protocol
  - **Only as absolute last resort**: Document the assumption AND explain why the paper doesn't provide this information
- Add to `inputs` list with full source tracking:
  - `name`: Descriptive name (referenced by `threshold_input_name`)
  - `value`: Same as `threshold` field (MUST be from paper)
  - `units`: Same as `threshold_units` field
  - `source_ref`: Reference to paper (almost never "modeling_assumption")
  - `value_snippet`: Verbatim quote containing threshold value (REQUIRED if from paper)
- **Document conversions**: Cell density, diameter→volume formulas, etc. as inputs with sources (modeling assumptions OK here)

**Common patterns:**

*Untreated baseline measurement:*
```yaml
# Paper: "CD8+ density in treatment-naive resected PDAC at 500 mm³"
biomarker_species: V_T.C1
threshold_computation_code: |
  def compute_threshold_value(species_dict, inputs, ureg):
      tumor_cells = species_dict['V_T.C1']
      cell_density = inputs['cell_packing_density']
      return (tumor_cells / cell_density).to(ureg.mm**3)
threshold: 500.0
threshold_units: millimeter**3
threshold_input_name: resection_tumor_volume
comparison: ">"
timepoints: [0.0]
inputs:
  - name: resection_tumor_volume
    value: 500.0
    units: millimeter**3
    description: "Tumor volume at resection"
    source_ref: smith_2020
    value_snippet: "resected at 500 mm³"
  - name: cell_packing_density
    value: 1.0e9
    units: cell / millimeter**3
    source_ref: modeling_assumption
    value_snippet: null
```

*Multiple timepoints around trigger:*
```yaml
# Paper: "Cytokine kinetics around tumor establishment (1e8 cells)"
biomarker_species: V_T.C1
threshold_computation_code: |
  def compute_threshold_value(species_dict, inputs, ureg):
      return species_dict['V_T.C1']  # Identity - threshold in cells
threshold: 1.0e8
threshold_units: cell
threshold_input_name: tumor_establishment_threshold
comparison: ">"
timepoints: [-7.0, 0.0, 7.0]  # Week before, at, week after
inputs:
  - name: tumor_establishment_threshold
    value: 1.0e8
    units: cell
    description: "Tumor cell count at establishment"
    source_ref: jones_2021
    value_snippet: "tumor establishment defined as 10^8 cells"
```

**Derivatives:** Use central differences from timepoints. Don't assume analytical derivatives exist.

### Distribution Code

**CRITICAL:**
- **NO magic numbers** - every constant comes through `inputs` with source traceability
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

**Anti-patterns to AVOID:**
```python
# BAD: Converting to dimensionless and stripping units early
mean = inputs['treg_fraction_mean'].to(ureg.dimensionless).magnitude  # Loses unit checking!
sd = inputs['treg_fraction_sd'].to(ureg.dimensionless).magnitude
# ... calculations with bare floats ...
# Easy to mix incompatible quantities without Pint catching errors

# GOOD: Keep as Quantities, only extract magnitude when necessary
mean = inputs['treg_fraction_mean']  # Keep as Quantity
samples = rng.normal(mean.magnitude, sd.magnitude, N) * mean.units  # Extract only for RNG
```

### Source Requirements

- **Primary source (singular)**: One paper, real DOI that resolves
- **No reuse**: Avoid studies already used for this observable: {{USED_PRIMARY_STUDIES}}
- **Verbatim snippets**: Exact quotes containing values (automatically verified)
- **Secondary sources**: Reference values, conversion factors (can be multiple)

### Experimental Context Enums

Use exact enum values for:
- **Indication**: PDAC, melanoma, NSCLC, breast, colorectal, RCC, etc.
- **Compartment**: tumor.primary, tumor.metastatic_lesion, lymph_node.tumor_draining, blood.PBMC, etc.
- **System**: clinical.resection, clinical.core_biopsy, animal_in_vivo.orthotopic, ex_vivo.organoid, etc.

---

## Source Hierarchy

**Prefer (in order):**
1. Same indication + compartment + system (e.g., PDAC tumor IHC)
2. Same indication + adjacent compartment OR different modality
3. Related indication + same compartment (document mismatch justification)
4. Pan-cancer or non-cancer reference (last resort, document limitations)

**Never acceptable:**
- Cell lines for clinical observables
- Mouse data without species scaling
- Pure assumptions without literature anchor

---

## Context

**Available model species:**
{{MODEL_SPECIES_WITH_UNITS}}

---

Generate calibration target metadata following all requirements above.
