# Calibration Target Extraction

Extract **raw observables** from scientific literature for QSP model calibration.

**Cancer type:** {{CANCER_TYPE}}

**Observable:** {{OBSERVABLE_DESCRIPTION}}

---

## What is a Calibration Target?

A biological observable measured in a **specific experimental scenario**, used to calibrate QSP model parameters via Bayesian inference.

**Critical concept:** Each observable has an **experimental context** (species, indication, compartment, system, treatment) that may differ from the **model context**.

**Your task:**
1. **Find data that matches the model context** (specified below) as closely as possible
2. **Document the experimental context accurately** - mismatches will be quantified later via formal distance metrics
3. Follow the strict matching requirements below

---

## Strict Matching Requirements

**CRITICAL - The following MUST match exactly (NO exceptions):**

1. **Species** - Use ONLY the exact species specified (e.g., human for human models). NEVER substitute mouse, rat, or other species.

2. **Indication** - Use ONLY the exact cancer type specified (e.g., PDAC). NEVER substitute related cancers or pan-cancer data.

3. **Compartment** - Use ONLY the exact compartment (e.g., tumor.primary, blood.peripheral). NEVER substitute serum for tissue or peripheral blood for tumor-infiltrating cells.

4. **Source** - Use ONLY in vivo patient data (biopsies, resections, blood draws). NEVER use cell culture, organoids, or in vitro measurements.

5. **Measurement Type** - Use ONLY direct measurements with absolute units. NEVER use statistical effect sizes (hazard ratios, odds ratios) or fold-changes as calibration values.

**Acceptable flexibility (document mismatches):**
- System (clinical.resection vs clinical.biopsy) - document timing differences
- Treatment history (treatment_naive vs post-treatment) - document and justify
- Measurement modality (IHC vs flow cytometry) - if measuring same quantity

**If strict requirements cannot be met:**
- Do NOT fabricate or substitute incompatible data
- Note the limitation in your response

---

## Validation Requirements

**Your response will be automatically validated. Ensure the following requirements are met to avoid validation failures:**

### 1. DOI Must Be Valid and Resolve
- Use real DOIs from actual papers (format: `10.xxxx/journal.year.id`)
- Verify DOI exists before submitting - search PubMed, Google Scholar, or journal websites
- Common DOI prefixes: `10.1038` (Nature), `10.1126` (Science), `10.1371` (PLOS), `10.1200` (JCO)
- **Validation:** DOI will be resolved via CrossRef API

### 2. Paper Title Must Match DOI Metadata
- Use the EXACT title from the paper (will be cross-checked with CrossRef)
- Copy the title character-for-character from the paper or CrossRef metadata
- **Validation:** Title similarity must be ≥75% match with CrossRef

### 3. Code Must Execute Without Errors
- `measurement_code` and `distribution_code` will both be executed with mock data
- Syntax errors or runtime errors will cause validation failure
- **Validation:** Code is executed with mock species data and Pint unit registry

### 4. Measurement Code Units Must Match
- `measurement_code` output units must match `calibration_target_estimates.units`
- **Validation:** Pint dimensional analysis checks measurement code output

### 5. Computed Values Must Match Reported Values
- `distribution_code` computed median/IQR/CI95 must match reported values within 1% tolerance
- **Validation:** Code is executed and outputs compared to reported statistics

### 6. Text Snippets Must Contain Declared Values
- Each `value_snippet` must contain the actual numeric value being reported
- Include enough context so the number is clearly visible in snippet text
- **Validation:** Automated string matching checks snippet content

### 7. All Source References Must Be Defined
- Every `source_ref` must point to either `primary_data_source.source_tag` or a `secondary_data_sources` entry
- Exception: `modeling_assumption` is valid for conversion factors and thresholds
- **Validation:** Cross-reference checking against defined sources

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

**Interventions** (may be empty list for untreated state):

Provide a text description for each intervention including:
- **What**: Agent/procedure name
- **How much**: Dose and units (mg/kg, mg/m², etc.)
- **When**: Schedule/timing
- **Additional details**: Patient weight/BSA if relevant

**Examples:**
- "Anti-PD-1 antibody 3 mg/kg IV every 2 weeks starting on day 0"
- "Surgical resection on day 14, removing 90% of tumor burden"
- "No intervention (natural disease progression)" → use empty list `[]`

**Measurements** (at least ONE required):

Each measurement requires:

1. **measurement_description** (text) - Describes WHAT is being measured and HOW:
   - Observable: What biological quantity
   - Method: How it's measured (e.g., "via IHC", "by flow cytometry")
   - Location: Where in the body
   - Units: Expected units

2. **measurement_species** (list) - Species accessed by measurement_code:
   - Format: `['compartment.species']` (e.g., `['V_T.CD8', 'V_T.C1']`)
   - Must match species names in model

3. **measurement_code** (executable Python) - Computes observable from simulation:
   - Function signature: `compute_measurement(time, species_dict, ureg)`
   - `time`: numpy array with day units (Pint Quantity)
   - `species_dict`: dict mapping species names to numpy arrays (Pint Quantities)
   - `ureg`: Pint UnitRegistry for conversions
   - Must return Pint Quantity with units matching `calibration_target_estimates.units`

4. **threshold_description** (text) - Describes WHEN the measurement occurs:
   - Trigger: What biological/clinical event triggers measurement
   - Threshold: Specific value if stated in paper
   - Context: Clinical context (resection, diagnosis, enrollment, etc.)

**Example measurement:**
```yaml
measurements:
  - measurement_description: "CD8+ T cell density measured via IHC in tumor tissue sections, reported as dimensionless ratio"
    measurement_species: ['V_T.CD8', 'V_T.C1']
    measurement_code: |
      def compute_measurement(time, species_dict, ureg):
          """Compute CD8/tumor ratio at baseline (single timepoint)."""
          cd8 = species_dict['V_T.CD8']
          tumor = species_dict['V_T.C1']
          ratio = cd8 / tumor
          return ratio.to(ureg.dimensionless)
    threshold_description: "At tumor resection when tumor burden reaches ~1e9 cells (~500 mm³)"
```

**Key principles for measurement_code:**
- Keep Pint units throughout calculation (see Pint Golden Rule below)
- Access only species listed in `measurement_species`
- Return scalar or array Pint Quantity
- Handle time array properly (index if single timepoint needed)

### Distribution Code

**Data Flow (CRITICAL to understand):**

1. **Paper reports** statistics (mean, SD, median, IQR, range, etc.) → Put these in `inputs[]`
2. **distribution_code** uses inputs to run Monte Carlo → Produces median, IQR, CI95
3. **calibration_target_estimates** contains the COMPUTED values (from step 2), NOT the paper's reported values

**Example:** Paper reports "150 ± 25 cells/mm²" but doesn't report median/IQR/CI95:
- `inputs`: `[{name: "mean", value: 150}, {name: "sd", value: 25}]` ← Paper's values
- `distribution_code`: Runs MC sampling from normal(150, 25)
- `calibration_target_estimates.median`: 149.94 ← Computed from MC, matches code output
- `calibration_target_estimates.iqr`: 33.59 ← Computed from MC, matches code output
- `calibration_target_estimates.ci95`: [100.79, 199.35] ← Computed from MC

**Validation:** Code is executed and outputs must match declared median/iqr/ci95 within 1% tolerance.

**Requirements:**
- **Extract biological/experimental values via inputs** with source traceability
- **Universal constants OK as literals**: percentiles (2.5, 25, 75, 97.5), mathematical constants (π, 2), MC sample sizes (10000)
- **Use MC methods** (parametric bootstrap), NOT analytical approximations
- **Use Pint units** - inputs are pre-converted Pint Quantities, return Pint Quantities
- Function signature: `derive_distribution(inputs, ureg)` returns dict with `median_obs`, `iqr_obs`, `ci95_obs`

**GOLDEN RULE: Keep values tethered to their units throughout calculations.**

```python
def derive_distribution(inputs, ureg):
    import numpy as np
    # Inputs are already Pint Quantities - use directly
    mean = inputs['cd8_density_mean']
    sd = inputs['cd8_density_sd']
    n_samples = int(inputs['n_mc_samples'].magnitude)  # Only extract for integer conversion

    # Extract magnitude only for numpy functions, reattach units immediately
    rng = np.random.default_rng(42)
    samples = rng.normal(mean.magnitude, sd.magnitude, n_samples) * mean.units

    # Return Quantities - validator checks dimensionality
    return {
        'median_obs': np.median(samples),
        'iqr_obs': np.percentile(samples, 75) - np.percentile(samples, 25),
        'ci95_obs': np.percentile(samples, [2.5, 97.5])
    }
```

**Key principle:** Extract `.magnitude` ONLY when absolutely necessary (numpy distribution functions, integer conversion), then immediately reattach units.

### Source Requirements

- **Primary source (singular)**: One paper, real DOI that resolves
- **No reuse**: Avoid studies already used for this observable: {{USED_PRIMARY_STUDIES}}
- **Verbatim snippets**: Exact quotes containing values (automatically verified)
- **Secondary sources**: Reference values, conversion factors (can be multiple)

---

## Source Hierarchy

**Priority order (given that species/indication/compartment MUST match exactly):**

1. **Ideal:** Exact match on all contexts (species, indication, compartment, system, treatment status)
2. **Acceptable:** Core match (species/indication/compartment), minor variations in system or treatment (document differences)
3. **Never:** Different species, indication, compartment, or in vitro data (violates strict requirements)

---

## Context

**Available model species:**
{{MODEL_SPECIES_WITH_UNITS}}

---

Generate calibration target metadata following all requirements above.
