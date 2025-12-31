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

**Measurements** (at least ONE required):
- **timing_type**: `at_diagnosis` (most common) or `biomarker_triggered`
- **timepoints**: Days relative to timing event. Single point `[0.0]` for direct measurements, multiple points `[-1.0, 0.0, 1.0]` for derivatives
- **required_species**: Model species needed for computation
- **computation_code**: Python function `compute_measurement(time, species_dict, ureg)` returning Pint Quantity

For biomarker-triggered timing, also specify: `biomarker_species`, `threshold`, `comparison` ('>' or '<')

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
