# Calibration Target Extraction

Extract **raw observables** from scientific literature for QSP model calibration.

**Cancer type:** {{CANCER_TYPE}}

**Observable:** {{OBSERVABLE_DESCRIPTION}}

---

## What is a Calibration Target?

A biological observable measured in a **specific experimental scenario**. Unlike test statistics (model-derived), calibration targets are **directly measured** values used to calibrate model parameters via Bayesian inference.

Example: "CD8+/tumor ratio at day 14 post-anti-PD-1 treatment in resected PDAC tumors"

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
- Absolute timing: list of days [7, 14, 21]
- Biomarker-triggered: species to monitor, threshold value, comparison ('>' or '<'), offset days
- Required species: which model species needed
- Computation code: Python function converting species → observable (returns Pint Quantity)

### Distribution Code

**CRITICAL:**
- **NO magic numbers** - every constant comes through `inputs` with source traceability
- **Use MC methods** (parametric bootstrap), NOT analytical approximations
- **Use Pint units** - inputs are pre-converted Pint Quantities, return Pint Quantities
- Function signature: `derive_distribution(inputs, ureg)` returns dict with `median_obs`, `iqr_obs`, `ci95_obs`

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
3. Related indication + same compartment (justify, reduce `indication_match` ≤0.65)
4. Pan-cancer or non-cancer reference (last resort, `indication_match` ≤0.5)

**Never acceptable:**
- Cell lines for clinical observables
- Mouse data without species scaling
- Pure assumptions without literature anchor

---

## Validation Rubrics

Assign weights [0-1] with brief justification:

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

## Context

**Available model species:**
{{MODEL_SPECIES_WITH_UNITS}}

---

Generate calibration target metadata following all requirements above.
