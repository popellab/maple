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

5. **Measurement Type** - Use ONLY direct measurements with absolute units. NEVER use statistical effect sizes (hazard ratios, odds ratios) as calibration values. Fold-changes are acceptable ONLY when explicitly requested AND computed from paired pre/post data within the same patients (see "Fold-Change Targets" section below).

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
- `measurement_code` output units must match `empirical_data.units`
- **Validation:** Pint dimensional analysis checks measurement code output

### 5. Computed Values Must Match Reported Values
- `distribution_code` computed median/IQR/CI95 must match reported values within 1% tolerance
- **Validation:** Code is executed and outputs compared to reported statistics

### 6. Text Snippets Must Contain Declared Values
- Each `value_snippet` must contain the actual numeric value being reported
- Include enough context so the number is clearly visible in snippet text
- **Validation:** Automated string matching checks snippet content
- **Exception for figures:** When values come from figures (plots, charts, bar graphs), use `source_type: 'figure'` with `figure_id` (e.g., `'Figure 2A'`) and `extraction_method: 'manual'`. Snippet validation is relaxed for figure sources — the `value_snippet` should contain the figure caption or relevant axis labels instead of the exact numeric value.

### 7. All Observable Constant Sources Must Be Traceable
- Every observable constant must have a verifiable source via `source_type`:
  - `reference_db`: Value from curated reference_values.yaml → requires `reference_db_name`
  - `derived_from_reference_db`: Computed from reference DB entries → requires `reference_db_names` list
  - `literature`: From a specific paper → requires `source_tag` matching a defined source
- **No ungrounded constants allowed.** Do NOT use "modeling_assumption" — every numeric must trace to a specific reference DB entry or literature source.
- **Validation:** Cross-reference checking against defined sources and reference DB entries

### 8. Measurement Code Output Scale Must Match Calibration Target Scale
- Ensure measurement_code output range is on the same scale as empirical_data
- Example: Don't mix 0-1 ratios with 0-N scores - they must use consistent scaling
- **Validation:** Code executed with mock data; output range compared to target range

### 9. Avoid Control Characters
- Do not include control characters in any text fields (causes YAML parsing errors)
- Common source: copying from PDFs or word processors with invisible formatting
- **Validation:** All text fields scanned for control characters

### 10. Population Aggregation (Optional)

Some clinical endpoints (ORR, median OS, 1-year OS, MPR rate) are population summary statistics that cannot be expressed as single-patient observables. Use `observable.aggregation` when the calibration target requires aggregating across a virtual patient cohort.

**When to use:**
- Overall response rate (ORR) — fraction of patients meeting RECIST criteria
- Median overall survival (OS) / progression-free survival (PFS)
- Landmark survival rates (e.g., 1-year OS)
- Major pathological response (MPR) rate

**Aggregation types:**
- `response_rate`: Requires `threshold_code` defining per-patient binary classification
- `median_time_to_event`: Observable code computes per-patient event times
- `survival_rate`: Requires `time_point` and `time_unit`
- `none`: Default, no aggregation (per-patient observable)

**Example (ORR via RECIST):**
```yaml
observable:
  code: |
    def compute_observable(time, species_dict, constants, ureg):
        tumor = species_dict['V_T.C1']
        baseline = tumor[0]
        return ((baseline - tumor) / baseline).to('dimensionless')
  units: dimensionless
  species: ['V_T.C1']
  support: unit_interval
  aggregation:
    type: response_rate
    threshold_code: |
      def classify_patient(time, species_dict, constants, ureg):
          tumor = species_dict['V_T.C1']
          baseline = tumor[0]
          nadir = min(tumor)
          return (baseline - nadir) / baseline >= 0.3
    rationale: "ORR requires classifying each virtual patient as responder/non-responder per RECIST 1.1 (>=30% decrease in longest diameter)"
```

**Validation:** `response_rate` and `survival_rate` aggregations warn if `support` is not `unit_interval`.

### 11. Source Relevance Assessment (Optional)

When the source data does not perfectly match the model context (different species, proxy indication, perturbed conditions), use `source_relevance` to formally document the translation quality.

**When to use:**
- Source indication is proxy or unrelated (e.g., melanoma data for PDAC model)
- Cross-species data (e.g., mouse measurements for human model)
- Pharmacological or genetic perturbation (drug-induced or knockout measurements)
- Low TME compatibility (e.g., immunogenic tumor model for immunosuppressive PDAC)

**When NOT needed:** Exact-match human clinical data for the target indication (the typical CalibrationTarget case).

**Key fields:**
| Field | Options |
|---|---|
| `indication_match` | `exact`, `related`, `proxy`, `unrelated` |
| `source_quality` | `primary_human_clinical`, `primary_human_in_vitro`, `primary_animal_in_vivo`, `primary_animal_in_vitro`, `review_article`, `textbook`, `non_peer_reviewed` |
| `perturbation_type` | `physiological_baseline`, `pathological_state`, `pharmacological`, `genetic_perturbation` |
| `tme_compatibility` | `high`, `moderate`, `low` (optional, for immune/stromal parameters) |
| `estimated_translation_uncertainty_fold` | 1.0-1000.0 (fold-uncertainty from source-to-target translation) |

**Example:**
```yaml
source_relevance:
  indication_match: proxy
  indication_match_justification: "Melanoma TIL data used as proxy for PDAC. Both solid tumors with CD8 infiltration, but PDAC has denser stroma and more T cell exclusion."
  species_source: human
  species_target: human
  source_quality: primary_human_clinical
  perturbation_type: physiological_baseline
  tme_compatibility: low
  tme_compatibility_notes: "Melanoma is T cell-permissive; PDAC is T cell-excluded. Expect 10-100x overestimation of infiltration rates."
  estimated_translation_uncertainty_fold: 10.0
```

**Validation:** Warnings (not errors) flag insufficient uncertainty for proxy/cross-species data, missing perturbation relevance documentation, and non-peer-reviewed sources.

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

## Statistic Selection Rules

### Prefer Descriptive Statistics Over Analytical Thresholds

**CRITICAL:** When a paper reports BOTH descriptive statistics (mean, median, SD, SEM, IQR) AND analytical cutoffs (X-tile optimal cutpoints, ROC thresholds, survival stratification values), ALWAYS use the descriptive statistics. Analytical cutoffs are optimized for binary classification (e.g., high vs low survival groups) and do NOT represent population central tendency.

**Priority order when the SAME paper offers multiple statistics:**
1. Tabulated descriptive statistics (Table with mean/median ± SD/SEM/IQR) — **always prefer this**
2. Text-reported summary statistics (e.g., "mean CD8 density was 15.2 ± 3.3 cells/mm²")
3. Figure-extracted values (read from bar/scatter plot with error bars)
4. **NEVER use as central values:** survival cutoffs, X-tile values, ROC thresholds, hazard ratio denominators, Youden index cutpoints

**Why this matters:** An X-tile cutoff of 11.8 cells/mm² and a mean of 15.2 ± 3.3 cells/mm² from the same paper represent fundamentally different quantities. The cutoff maximizes survival discrimination; the mean estimates population central tendency. Only the mean (with its SEM/SD) is appropriate for calibration.

### Standard Deviation vs Standard Error of the Mean

**CRITICAL:** When a paper reports mean ± value, you MUST determine whether the ± value is a **standard deviation (SD)** or a **standard error of the mean (SEM)**. Misidentifying SEM as SD underestimates population variability by a factor of √n, producing unrealistically narrow confidence intervals.

**How to determine SD vs SEM:**

1. **Check the paper text.** Look for explicit labels ("mean ± SD", "mean ± SEM", "mean ± SE") in tables, figure legends, methods section, or statistical methods.

2. **Apply the √n test.** If the paper reports ± values for multiple subgroups with similar biology:
   - Compute SD = ± value × √n for each group
   - If the derived SDs are approximately equal across groups, the ± values are SEMs
   - If the ± values themselves are approximately equal across groups, they may be SDs

3. **Apply the biological plausibility (CV) test.** For immune cell densities, cell counts, and biomarker concentrations:
   - CV = SD / mean. Typical biological CV is 50–200% for immune cell densities
   - If treating ± as SD gives CV < 20%, it is almost certainly SEM
   - If treating ± as SEM gives biologically plausible CV (50–200%), this confirms SEM

4. **Default assumption.** If NONE of the above resolves the ambiguity:
   - Clinical papers and large-cohort studies (n > 30) more commonly report SEM
   - Basic science papers more commonly report SD
   - When uncertain, state the ambiguity explicitly in `key_assumptions`

**When you identify the ± value as SEM:**
- Name the input with `sem_` prefix (e.g., `sem_cd8_density`), NOT `sd_`
- Set `dispersion_type: se` and provide `dispersion_type_rationale` explaining your evidence
- Convert to SD in `distribution_code`: `sd = sem * np.sqrt(n)`
- Document the determination in `key_assumptions`

**When you identify the ± value as SD:**
- Name the input with `sd_` prefix (e.g., `sd_cd8_density`)
- Set `dispersion_type: sd` and provide `dispersion_type_rationale`
- Use SD directly in `distribution_code` (no conversion needed)

**Example (SEM identified via √n test):**
```yaml
inputs:
  - name: sem_cd8_density
    value: 15.0
    units: cell / millimeter**2
    description: "SEM for CD8 density, identified via √n test"
    dispersion_type: se
    dispersion_type_rationale: |
      Paper reports 227.7 ± 15.0 (n=368). SD = 15.0 × √368 = 287.7.
      Second group: 220.1 ± 33.0 (n=76). SD = 33.0 × √76 = 287.7.
      Identical SDs confirm these are SEMs. CV if SD would be 6.6% (implausible).
```

### Ratio and Composite Targets

**CRITICAL:** When the calibration target is a RATIO of two quantities (e.g., M1/M2 macrophage ratio, CD8/Treg ratio):

1. **Extract numerator and denominator from the SAME patients and analysis level.** Same table, same stratification, same statistical summary. NEVER divide a summary statistic from one patient subgroup by a statistic from a different subgroup — this destroys within-patient correlation and produces biologically meaningless ratios.

2. **Prefer directly reported ratios.** If the paper reports the ratio itself (e.g., "M1/M2 ratio was 0.45 ± 0.12"), use that value directly rather than reconstructing it from separate M1 and M2 densities.

3. **Biological plausibility check.** Before finalizing a ratio target, verify against known literature consensus. For example:
   - M2 macrophages typically outnumber M1 in untreated solid tumors (M1/M2 ≈ 0.3–0.7 in PDAC)
   - Tregs are a minority of CD4+ T cells (Treg/CD4 ≈ 0.1–0.3)
   - If your computed ratio contradicts established biology, re-examine your extraction method.

### Fold-Change Targets

When the calibration target requests a fold-change (pre-to-post treatment change):

1. **Use ONLY paired pre-to-post data from the SAME patients.** The fold change must be (post-treatment value) / (pre-treatment baseline value) within a single treatment arm.

2. **NEVER compute fold change as a cross-arm ratio** (Arm A value / Arm B value). Cross-arm comparisons confound the treatment effect with baseline differences and inter-patient variability. They are a completely different quantity from within-patient fold changes.

3. **If no paired pre/post data exists** for the requested comparison, clearly state this limitation rather than substituting a cross-arm ratio.

### Scaling Factor Red Flag

**CRITICAL:** If you find yourself needing a dimensionless scaling factor > 10× in `observable.constants` to reconcile model output with literature data, this almost certainly indicates an extraction error.

**Common root causes of needing large scaling factors:**
- Wrong statistic type (analytical threshold instead of descriptive statistic)
- Wrong quantity (cross-arm ratio instead of pre/post fold change)
- Unit mismatch or conversion error
- Measurement from a non-comparable subset (e.g., TLA-restricted density vs whole-tumor density)

**What to do:** STOP and re-examine your data source. Find a study that reports a quantity directly comparable to the model species, or explicitly document in `key_study_limitations` why no direct comparison is possible.

Legitimate conversion factors have clear physical meaning (e.g., cell cross-sectional area = 2.27e-4 mm²/cell from geometry) and should never be dimensionless fudge factors.

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

3. **measurement_constants** (list) - All conversion factors and reference values with units:
   - **REQUIRED** for any numeric constant with units used in measurement_code
   - Each constant requires: `name`, `value`, `units`, `biological_basis`, `source_type`, and source-specific fields
   - **source_type** must be one of:
     - `reference_db` → also provide `reference_db_name` (must match reference_values.yaml entry)
     - `derived_from_reference_db` → also provide `reference_db_names` list (each must match)
     - `literature` → also provide `source_tag` (must match a defined source in this target)
   - **Never hardcode numbers with units** like `23.0 * ureg('mg/mL')` in measurement_code
   - **No "modeling_assumption" allowed** — every constant must trace to a verifiable source
   - Access in code via: `constants['constant_name']`

4. **measurement_code** (executable Python) - Computes observable from species time series:
   - Function signature: `compute_measurement(time, species_dict, ureg, constants)`
   - `time`: numpy array with day units (Pint Quantity)
   - `species_dict`: dict mapping species names to numpy arrays (Pint Quantities, one value per timepoint)
   - `ureg`: Pint UnitRegistry for conversions
   - `constants`: dict mapping constant names to Pint Quantities (from measurement_constants)
   - Must return Pint Quantity (scalar or array) with units matching `empirical_data.units`
   - **IMPORTANT**: Do NOT hardcode numbers with units. Use `constants` dict for all conversion factors.
   - **IMPORTANT**: Do NOT include time filtering logic. This function computes WHAT to measure. WHEN to measure is handled via threshold_description.

5. **support** (required) - Declares the mathematical support of the output:
   - `positive`: Output must be > 0 (densities, concentrations, volumes)
   - `non_negative`: Output must be ≥ 0 (counts)
   - `unit_interval`: Output must be in [0, 1] (fractions, proportions)
   - `positive_unbounded`: Output must be > 0, no upper bound (fold-changes, ratios)
   - `real`: Any real value (log-ratios, change scores)

6. **threshold_description** (text) - Describes WHEN the measurement occurs:
   - **What triggers measurement**: The biological or clinical event/condition that prompts observation
   - **Timing context**: When in disease progression or treatment timeline
   - **Avoid circular reasoning**: Don't define timing by the observable being measured (e.g., don't say "when tumor reaches X cm" if measuring tumor size)
   - **Be specific about causality**: Distinguish between biological thresholds (disease progression) and clinical decisions (institutional protocols, symptom-triggered interventions)

**Example measurement:**
```yaml
measurements:
  - measurement_description: "CD8+ T cell density per tumor area via IHC, cells/mm²"
    measurement_species: ['V_T.CD8', 'V_T.C1']
    measurement_constants:
      - name: area_per_cancer_cell
        value: 2.27e-4
        units: mm**2/cell
        biological_basis: "From reference DB pdac_cancer_cell_diameter (17 μm) → π×(8.5 μm)² = 2.27e-4 mm²"
        source_type: derived_from_reference_db
        reference_db_names: [pdac_cancer_cell_diameter]
    measurement_code: |
      def compute_measurement(time, species_dict, ureg, constants):
          cd8 = species_dict['V_T.CD8']
          c_cells = species_dict['V_T.C1']
          tumor_area = c_cells * constants['area_per_cancer_cell']
          return (cd8 / tumor_area).to('cell/mm**2')
    support: positive
    threshold_description: "At tumor resection"
```

**Key principles for measurement_code:**
- Keep Pint units throughout calculation (see Pint Golden Rule below)
- Access only species listed in `measurement_species`
- Return Pint Quantity (typically array with one value per timepoint)
- Do NOT include time filtering logic - compute over entire time series

### Distribution Code

**Data Flow (CRITICAL to understand):**

1. **Paper reports** statistics (mean, SD, median, IQR, range, etc.) → Put these in `inputs[]`
2. **distribution_code** uses inputs to run Monte Carlo → Produces median, IQR, CI95
3. **empirical_data** contains the COMPUTED values (from step 2), NOT the paper's reported values

**Example:** Paper reports "150 ± 25 cells/mm²" but doesn't report median/IQR/CI95:
- `inputs`: `[{name: "mean", value: 150}, {name: "sd", value: 25}]` ← Paper's values
- `distribution_code`: Runs MC sampling from normal(150, 25)
- `empirical_data.median`: 149.94 ← Computed from MC, matches code output
- `empirical_data.iqr`: 33.59 ← Computed from MC, matches code output
- `empirical_data.ci95`: [100.79, 199.35] ← Computed from MC

**Validation:** Code is executed and outputs must match declared median/ci95 within tolerance (1% for median, 10% for CI bounds due to Monte Carlo variance).

**Requirements:**
- **Extract biological/experimental values via inputs** with source traceability
- **Universal constants OK as literals**: percentiles (2.5, 25, 75, 97.5), mathematical constants (π, 2), MC sample sizes (10000)
- **Use MC methods** (parametric bootstrap), NOT analytical approximations
- **Use Pint units** - inputs are pre-converted Pint Quantities, return Pint Quantities
- Function signature: `derive_distribution(inputs, ureg)` returns dict with `median_obs`, `ci95_lower`, `ci95_upper`

**IMPORTANT: What distribution_code is (and is NOT) for:**

distribution_code is **STATISTICAL ONLY** - it converts literature values (mean ± SD) into
sampling distributions. It should NEVER contain:
- Model compartment volumes (V_T, V_C, V_P) - these belong in observable.code
- Physical constants or conversion factors - these belong in observable.constants
- ODE parameters or dynamics - these belong in submodel.code

**If you find yourself writing `V_T = 1.0 * ureg.milliliter` in distribution_code, STOP.**
You're confusing statistical derivation with model simulation.

**GOLDEN RULE: Reattach units immediately after sampling → units propagate naturally.**

```python
def derive_distribution(inputs, ureg):
    import numpy as np
    # Inputs are already Pint Quantities - use directly
    mean = inputs['cd8_density_mean']
    sd = inputs['cd8_density_sd']
    n_samples = int(inputs['n_mc_samples'].magnitude)  # Only extract for integer conversion

    # Sampling strips units - reattach IMMEDIATELY
    rng = np.random.default_rng(42)
    samples = rng.normal(mean.magnitude, sd.magnitude, n_samples) * mean.units

    # np.median, np.percentile, np.mean, np.std all preserve units!
    return {
        'median_obs': np.median(samples),
        'ci95_lower': np.percentile(samples, 2.5),
        'ci95_upper': np.percentile(samples, 97.5),
    }
```

**Key principle:** Sampling strips units. Reattach immediately after, then units propagate through numpy operations (median, percentile, etc.).

### Choosing Probability Distributions

**CRITICAL:** Select the distribution family that matches the measurement type to avoid non-physical values and bias.

**Size/Volume/Mass Measurements (always positive, often right-skewed):**
- **Prefer lognormal distribution** for tumor diameter/volume, cell counts, organ weights, concentrations
- Normal distributions often yield negative draws → require clipping → **introduces bias**
- If paper reports mean ± SD for size data, convert to lognormal parameters:
  ```python
  # Convert normal(mean, sd) to lognormal parameters
  mu_log = np.log(mean.magnitude**2 / np.sqrt(mean.magnitude**2 + sd.magnitude**2))
  sigma_log = np.sqrt(np.log(1 + sd.magnitude**2 / mean.magnitude**2))
  samples = rng.lognormal(mu_log, sigma_log, n_samples) * mean.units
  ```
- **Red flag:** If you need `np.clip()` to avoid negatives, you probably need lognormal
- Document in key_assumptions if using normal for size data (explain why clipping is justified)

**Proportions/Fractions (bounded 0-1):**
- Use Beta distribution or logit-normal for percentages, response rates, cell fractions

**Count Data:**
- Poisson or negative binomial for event counts, discrete measurements

**Symmetric Continuous Data:**
- Normal distribution appropriate for measurements without natural bounds (e.g., change scores, log-ratios)

### Conversion Factors and Uncertainty Propagation

When `measurement_code` includes conversion factors (cells → volume, IHC score → density, cellularity adjustments):

1. **Document all conversion assumptions in `key_assumptions`**
   - Cell sizes, tissue densities, cellularity fractions, spherical approximations
   - Include numeric values and cite sources

2. **Extract uncertain conversion factors as inputs**
   - If cellularity varies widely (e.g., 10-50% in PDAC stroma), consider sampling in `distribution_code`
   - If cell size has reported uncertainty, propagate it through the calculation
   - **Example:** Sample cellularity from uniform(0.15, 0.35) rather than fixing at 0.25

3. **Cite sources for conversion factors**
   - Use `source_type: reference_db` or `source_type: derived_from_reference_db` for values from the curated reference database
   - Use `source_type: literature` with a `source_tag` for values from papers (add paper to `secondary_data_sources`)
   - **Never use ungrounded constants** — every numeric must trace to a specific reference DB entry or literature source

4. **Assess impact of conversion uncertainty**
   - Note in `key_study_limitations` if conversion assumptions dominate uncertainty
   - **Example:** "Cellularity varies 10-50% in PDAC; assuming fixed 25% introduces unquantified error"

### Source Requirements

- **Primary source (singular)**: One paper, real DOI that resolves
- **No reuse**: Avoid studies already used for this observable: {{USED_PRIMARY_STUDIES}}
- **Verbatim snippets**: Exact quotes containing values (automatically verified)
- **Secondary sources**: Reference values, conversion factors (can be multiple)

### Sample Size Requirement

You MUST extract the sample size (n) for each measurement. This is critical
for proper uncertainty quantification and pooling across studies.

**Look for:**
- "n = X" or "N = X" in methods/results sections
- Sample sizes in figure legends (e.g., "n=5 per group")
- Patient/subject counts in study design
- Number of samples/biopsies in clinical studies

**Required fields in `empirical_data`:**
- `sample_size`: int or List[int] - the numeric value(s)
- `sample_size_rationale`: str - explanation of how sample size was determined

**If sample size is not explicitly reported:**
- Check figure error bars - if SEM is reported, can sometimes back-calculate n from SD/SEM
- Note uncertainty in rationale: "Sample size not explicitly reported; n≈X inferred from methods"
- Use conservative estimate based on study type

**Example:**
```yaml
empirical_data:
  median: 149.94
  iqr: 33.59
  ci95: [100.79, 199.35]
  units: cell / mm**2
  sample_size: 42
  sample_size_rationale: "n=42 patients with resected PDAC tumors, stated in Table 1"
```

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

**Available reference values (curated constants for conversion factors):**
{{REFERENCE_DATABASE}}

Use these reference values in `observable.constants` when applicable (e.g., cell diameters, molecular weights, tissue densities). This avoids re-deriving standard physical/biological constants and ensures consistency across targets.

---

## Viewing Figures from Papers

You have a `view_figure` tool that can fetch and display figure images from scientific papers. When a paper reports key data only in figures (bar charts, scatter plots, survival curves) and not in tabulated form, use this tool to view the figure and read numeric values directly from the plot axes.

**Usage:**
- Call `view_figure(paper_url="https://...", figure_label="Figure 2A")` with the paper URL (PMC URLs work best) and the figure label
- The tool returns the figure image so you can read values from axes, error bars, and data points
- After reading values from a figure, set `source_type: 'figure'`, `figure_id`, and `extraction_method: 'manual'` on the corresponding input

**When to use:**
- The paper's abstract/text mentions data shown in a figure but doesn't tabulate values
- You need to verify values visible in plots but not stated in text
- Clinical endpoints are shown as Kaplan-Meier curves or waterfall plots

---

Generate calibration target metadata following all requirements above.
