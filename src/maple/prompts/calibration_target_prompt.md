# Calibration Target Extraction

Extract **raw observables** from scientific literature for QSP model calibration.

**Cancer type:** {{CANCER_TYPE}}

**Observable:** {{OBSERVABLE_DESCRIPTION}}

---

## What is a Calibration Target?

A biological observable measured in a **specific experimental scenario**, used to calibrate QSP model parameters via Bayesian inference.

Each observable has an **experimental context** (species, indication, system, treatment) that may differ from the **model context**. Your job: find data that matches the model context, document the actual experimental context, and follow the strict matching requirements below.

---

## Output YAML Structure

You produce a single `CalibrationTarget` YAML with these top-level fields. **Required** unless marked optional.

| Field | Type | Notes |
|---|---|---|
| `study_interpretation` | str | One-paragraph narrative: what is being measured, how the experimental context maps to the model, key methodological points |
| `key_assumptions` | List[str] (≥1) | Biological + statistical assumptions made in extraction (e.g., cell-type equivalence, normal-distribution assumption) |
| `key_study_limitations` | List[str] | Issues that bias estimates or limit generalizability (e.g., small cohort, figure-extracted values) |
| `observable` | dict | How to compute the observable from QSP species. See "Observable" section below |
| `experimental_context` | dict | The source paper's context: `{species, system, indication, treatment, stage?, mouse_subspecifier?, cell_lines?, culture_conditions?, tissue_source?, assay_type?}` — describes WHERE the data came from, not the model target |
| `scenario` | dict (optional) | Interventions + measurement timing. Omit if untreated baseline and no perturbations |
| `empirical_data` | dict | Computed `median`, `ci95`, `units`, `sample_size`, `inputs[]`, `assumptions[]`, `distribution_code`. See "Empirical Data" section |
| `primary_data_source` | dict | Single paper with verified DOI. See "Source Requirements". Required when `epistemic_basis: literature` (default) |
| `secondary_data_sources` | List[dict] | Reference values / conversion-factor sources. Empty list OK |
| `epistemic_basis` | `"literature"` (default) or `"mechanistic"` | Use `"mechanistic"` only for biological-invariant priors with no primary measurement (live in `calibration_targets/mechanistic/`); requires deliberately wide CIs and rationale in `key_assumptions`. Otherwise leave at default |

The `experimental_context` block is REQUIRED — it documents where the data came from. The `Model Context (Target to Match)` section below describes the *target* you're matching against; `experimental_context` describes the *actual* source. They may legitimately differ (cross-species, proxy indication, etc.) — document mismatches via `primary_data_source.source_relevance` (required whenever a `primary_data_source` is provided).

---

## Strict Matching Requirements

**CRITICAL - The following MUST match exactly (NO exceptions):**

1. **Species** - Use ONLY the exact species specified (e.g., human for human models). NEVER substitute mouse, rat, or other species.

2. **Indication** - Use ONLY the exact cancer type specified (e.g., PDAC). NEVER substitute related cancers or pan-cancer data.

3. **Source** - Use ONLY in vivo patient data (biopsies, resections, blood draws). NEVER use cell culture, organoids, or in vitro measurements.

4. **Measurement Type** - Use ONLY direct measurements with absolute units. NEVER use statistical effect sizes (hazard ratios, odds ratios) as calibration values. Fold-changes are acceptable ONLY when explicitly requested AND computed from paired pre/post data within the same patients (see "Fold-Change Targets" section below).

**Acceptable flexibility (document mismatches):**
- System (clinical.resection vs clinical.biopsy) - document timing differences
- Treatment history (treatment_naive vs post-treatment) - document and justify
- Measurement modality (IHC vs flow cytometry) - if measuring same quantity
- **Compartment mismatch (e.g., serum vs tumor tissue)** - acceptable ONLY when a matching auxiliary-parameter group is available (see "Available Auxiliary Parameter Groups" below). The bridging factor must be declared via `observable.auxiliary_parameters` and applied in `observable.code` so the inference absorbs the bridging uncertainty. If no matching group is declared for the relevant bridge, treat the compartment mismatch as a strict-rejection criterion.

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
- `observable.code` and `empirical_data.distribution_code` will both be executed with mock data
- Syntax errors or runtime errors will cause validation failure
- **Validation:** Code is executed with mock species data and Pint unit registry

### 4. Observable Code Units Must Match
- `observable.code` output units must match `observable.units` and `empirical_data.units`
- **Validation:** Pint dimensional analysis checks measurement code output

### 5. Computed Values Must Match Reported Values
- `distribution_code` computed median/CI95 must match reported values within 1% (median) and 10% (CI bounds) tolerance
- **Validation:** Code is executed and outputs compared to reported statistics

### 6. Text Snippets Must Contain Declared Values

For values **in body text or numeric mentions** (the default case), set `value_snippet` to a verbatim quote that contains the numeric value. Automated string matching verifies the value appears in the snippet.

For values **read from a figure or table** — where PDF text extraction is unreliable or unreadable — set `value_snippet: null` and instead populate one of these structured excerpt blocks, both of which are **all-string** with `extra="forbid"`:

`figure_excerpt` (use when value comes from a figure — plot, chart, scatter, bar, dose-response curve):

```yaml
figure_excerpt:
  figure_id: "Figure 2A"                  # str — e.g. 'Figure 1C', 'Supplementary Figure S2A'
  value: "~5 pg/mg"                       # str — what you READ from the figure (text annotation, NOT the digitized number; e.g., '~5', '2-5 range', 'lower whisker at ~2 pg/mg')
  description: "highest data point in scatter plot at 16 h"   # str — what was read and from where in the panel
  context: "Figure 2A: IL-10 ELISA on tumor lysates (pg/mg total protein); error bars = SD across n=3 mice"  # str — caption / axis labels / panel conditions
```

`table_excerpt` (use when value comes from a table cell):

```yaml
table_excerpt:
  table_id: "Table 2"                     # str — e.g. 'Table 2', 'Supplementary Table S1'
  column: "PDAC tumor"                    # str — column header the value falls under
  row: "IL-1β (pg/mL)"                    # str — row label / identifier
  value: "29 ± 10"                        # str — value AS IT APPEARS in the cell (preserve formatting like '29 ± 10', '<0.1', 'n/d')
  context: "Table caption: 'Cytokine concentrations in resected PDAC vs adjacent normal pancreas; n=12 per group, mean ± SD'"  # str — caption, units in column header, surrounding text
```

**CRITICAL — common mistakes:**

- `figure_excerpt.value` and `table_excerpt.value` are **strings**, NOT numbers. Do NOT emit `value: 5.263` (numeric). The actual digitized numeric value belongs in the parent `EstimateInput.value` field; the excerpt's `value` is a *string description of what's visible* (e.g., `"~5"`, `"5.263"` quoted as a string, or `"lower whisker at ~5 pg/mg"`).
- These are the ONLY fields each schema accepts. Do not add `panel`, `group`, `caption_excerpt`, or other made-up fields — they will fail with `extra_forbidden` validation errors.
- All five `table_excerpt` fields and all four `figure_excerpt` fields are **required**.

When figure/table excerpts are populated, snippet validation is relaxed (figure-derived values cannot be text-matched, so they are flagged for manual review instead of failing validation). For non-figure / non-table sources, populate `value_snippet` with the verbatim quote as before.

### 7. All Observable Constant Sources Must Be Traceable
- Every observable constant must have a verifiable source via `source_type`:
  - `reference_db`: Value from curated reference_values.yaml → requires `reference_db_name`
  - `derived_from_reference_db`: Computed from reference DB entries → requires `reference_db_names` list
  - `literature`: From a specific paper → requires `source_tag` matching a defined source
- **No ungrounded constants allowed.** Do NOT use "modeling_assumption" — every numeric must trace to a specific reference DB entry or literature source.
- **Validation:** Cross-reference checking against defined sources and reference DB entries

### 8. Observable Code Output Scale Must Match Calibration Target Scale
- Ensure `observable.code` output range is on the same scale as `empirical_data`
- Example: Don't mix 0-1 ratios with 0-N scores - they must use consistent scaling
- **Validation:** Code executed with mock data; output range compared to target range

### 9. Avoid Control Characters
- Do not include control characters in any text fields (causes YAML parsing errors)
- Common source: copying from PDFs or word processors with invisible formatting
- **Validation:** All text fields scanned for control characters

---

## Optional Features

### Population Aggregation

Some clinical endpoints (ORR, median OS, 1-year OS, MPR rate) are population summary statistics, not per-patient observables. When the target requires aggregating across a virtual patient cohort, populate `observable.aggregation`:

- `response_rate` — needs `threshold_code` for per-patient binary classification
- `median_time_to_event` — observable code computes per-patient event times
- `survival_rate` — needs `time_point` and `time_unit`
- `none` (default) — per-patient observable, no aggregation

`response_rate` / `survival_rate` aggregations warn if `support` ≠ `unit_interval`.

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
    rationale: "ORR per RECIST 1.1 (≥30% decrease in longest diameter)"
```

### Population Spread (across-patient variability)

For hierarchical / virtual-patient inference, a target declares whether its reported width is genuine patient-to-patient spread (usable as the population-spread / omega signal) or just uncertainty on the mean. Default is `center_only` (excluded from omega); opt in explicitly. Two interoperable ways to declare it:

1. **`population_spread` + a `samples` array.** Set `empirical_data.population_spread: across_patient` and have `distribution_code` ALSO return a `samples` key — the across-patient population draw (one value per patient-equivalent); its empirical spread is the omega signal. Keep the default `center_only` (and do NOT return `samples`) when the width is a pooled-mean / SEM CI that shrinks with n. `median_obs` / `ci95` are unaffected either way.

2. **`observed_distribution`** (general representation, shared with submodel targets). Author it in whichever form the paper reports — **prefer `moments`** (mean +/- SD, median +/- IQR, CV, CI); the framework expands it to quartiles, so do not hand-convert:

```yaml
empirical_data:
  population_spread: across_patient
  observed_distribution:
    moments:
      center: 17
      center_type: median
      scale: 21            # full IQR here
      scale_type: iqr      # sd | sem | cv | iqr | ci95_halfwidth
      shape: lognormal     # lognormal | normal
    spread_source: across_patient       # or center_only (SEM/CI on the mean; default)
    n_biological: 40
    experimental_unit_type: biological
```

Use the `quantiles` form when the paper gives quartiles/percentiles/samples directly:

```yaml
  observed_distribution:
    quantiles:
      - {p: 0.25, value: 9}
      - {p: 0.5,  value: 17}
      - {p: 0.75, value: 30}
    spread_source: across_patient
    n_biological: 40
    experimental_unit_type: biological
```

Provide EXACTLY ONE of `moments` / `quantiles`. A population spread (`spread_source: across_patient`) REQUIRES `n_biological` + `experimental_unit_type: biological`. When both `observed_distribution` and `population_spread` are present they must agree that the width is (or is not) genuine spread — a validator enforces this.

### Source Relevance Assessment

`primary_data_source.source_relevance` (and the same field on each `secondary_data_sources` entry) is **required** whenever a Source is provided. Even for exact-match human clinical data, fill the fields — the indication_match=`exact` / species_target=species_source / `tme_compatibility=high` path is the common case, not a skip:

| Field | Options |
|---|---|
| `indication_match` | `exact`, `related`, `proxy`, `unrelated` |
| `source_quality` | `primary_human_clinical`, `primary_human_in_vitro`, `primary_animal_in_vivo`, `primary_animal_in_vitro`, `review_article`, `textbook`, `non_peer_reviewed` |
| `perturbation_type` | `physiological_baseline`, `pathological_state`, `pharmacological`, `genetic_perturbation` |
| `tme_compatibility` | `high`, `moderate`, `low` |
| `heterogeneity_transfer` | `high`, `moderate`, `low` — spread-transfer grade (+ `heterogeneity_transfer_justification`). OPTIONAL: omit for direct patient measurements (they measure the target spread directly); set only for proxy / preclinical sources whose spread understates patient heterogeneity. |
| `validation_warnings` | List[str] — free-text caveats (e.g., "Values digitized from scatter plots; precision ±0.5%"). Optional but commonly used. |

**Example:**
```yaml
primary_data_source:
  ...
  source_relevance:
    indication_match: proxy
    indication_match_justification: "Melanoma TIL data used as proxy for PDAC. Both solid tumors with CD8 infiltration, but PDAC has denser stroma and more T cell exclusion."
    species_source: human
    species_target: human
    source_quality: primary_human_clinical
    perturbation_type: physiological_baseline
    tme_compatibility: low
    tme_compatibility_notes: "Melanoma is T cell-permissive; PDAC is T cell-excluded. Expect 10–100× overestimation of infiltration rates."
```

---

## Model Context (Target to Match)

**Species:** {{MODEL_SPECIES}}
**Indication:** {{MODEL_INDICATION}}
**System:** {{MODEL_SYSTEM}}
**Treatment history:** {{MODEL_TREATMENT_HISTORY}}
**Stage/burden:** {{MODEL_STAGE_BURDEN}}

Find observables that match this context as closely as possible. Document the actual experimental context in your extraction.

---

## Task

{{PRIMARY_SOURCE_TITLE}}
1. Extract measurement with uncertainty (mean ± SD/SE, 95% CI, IQR, or range)
2. Specify the experimental scenario (interventions + measurement timing/location)
3. Document experimental context (species, system, indication, treatment history, stage)
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

When a paper reports `mean ± value`, you must determine whether the dispersion is **SD** or **SEM**. Misidentifying SEM as SD underestimates population variability by √n.

**Detection (in order of preference):**

1. **Explicit label** in tables, figure legends, or methods ("mean ± SD", "mean ± SEM", "± SE").
2. **√n test** across subgroups: compute `SD = ± × √n` for each group; if derived SDs are ~equal across groups, the ± values are SEMs.
3. **CV plausibility test** for immune-cell / biomarker measurements: typical biological CV is 50–200%. If `CV = SD/mean < 20%`, the ± is almost certainly SEM.
4. If still unresolved, state the ambiguity in `key_assumptions` and pick the more conservative interpretation (SD, wider distribution).

**Encoding:**

| Identified as | Input name prefix | `dispersion_type` | In `distribution_code` |
|---|---|---|---|
| SD | `sd_*` | `sd` | use directly |
| SEM | `sem_*` | `se` | convert: `sd = sem * np.sqrt(n)` |

Always populate `dispersion_type_rationale` with your evidence (label, √n test arithmetic, or CV check).

**Example (SEM via √n test):**
```yaml
inputs:
  - name: sem_cd8_density
    value: 15.0
    units: cell / millimeter**2
    dispersion_type: se
    dispersion_type_rationale: |
      Paper reports 227.7 ± 15.0 (n=368) and 220.1 ± 33.0 (n=76).
      Derived SDs = 287.7 and 287.7 — equal across groups → SEMs.
      Treating as SD would give CV 6.6% (implausibly narrow for immune cell density).
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

### Denominator Audit

**CRITICAL:** When your observable computes a ratio (density, fraction, percentage), verify that the numerator and denominator match the experimental measurement:

1. **State the experimental denominator explicitly.** What does the paper divide by?
   - "per mm² of tumor tissue" (includes stroma + cancer + immune cells)
   - "% of all cells in ROI" (all nucleated cells)
   - "% of CD45+ cells" (all leukocytes)
   - "% of CD3+ cells" (all T cells)

2. **State the model denominator.** What does your observable code divide by?
   - `V_T.C1 * area_per_cell` (cancer-cell-only area — **ignores stroma**)
   - Sum of T cell species only (excludes myeloid, B cells)
   - Sum of all immune species (excludes cancer cells, fibroblasts)

3. **Confirm they match.** If not, either:
   - Add correction constants (e.g., `stromal_fraction`) with reference DB source
   - Restructure the observable to eliminate the mismatch (e.g., use a ratio of two densities from the same paper — see "Prefer Ratios" below)
   - Document the residual mismatch magnitude in `key_assumptions` and in `observable.unmodeled_denominator_components`

**Common pitfall — desmoplastic tumors (PDAC, cholangiocarcinoma):** PDAC is 60–90% stroma. Using `C1 * area_per_cancer_cell` as tumor tissue area underestimates the true section area by 3–10×, systematically overpredicting cell densities. Always include `pdac_stromal_fraction` from the reference DB, or — better — prefer dimensionless ratios.

**Common pitfall — lymphoid aggregates / tertiary lymphoid structures:** LAs are immune-cell-dense regions where B cells constitute 50–70% of cells. B cells are typically not modeled. If the measurement denominator is "all cells in LA ROI", using only modeled immune species in the denominator will overestimate the predicted fraction by 2–3×. Document this in `unmodeled_denominator_components`.

### Prefer Dimensionless Ratios Over Absolute Densities

When the same paper reports densities for multiple cell types measured on the same tissue sections (e.g., CD3+ and Foxp3+ cells/mm²), **prefer computing their RATIO** rather than calibrating against absolute densities. Ratios:

- **Cancel the tissue area factor** (no stroma correction or section thickness needed)
- **Are independent of counting field size** and magnification
- **Use paper-specific denominators** (more accurate than generic reference values)

**Example:** A paper reports Foxp3+ = 48.5 cells/mm² and CD3+ = 942.5 cells/mm² from the same cohort and sections. Use Foxp3+/CD3+ = 0.051 (dimensionless) rather than trying to convert 48.5 cells/mm² to model cell counts via tissue area estimation.

**When to use ratios:**
- Paper reports ≥2 cell-type densities from the same sections/cohort
- The ratio maps cleanly to model species (e.g., CD3+ → all T cells, CD8+/CD3+ → effector fraction)
- The absolute density target would require uncertain conversion constants (stromal fraction, section volume, cellularity)

**When absolute densities are unavoidable:**
- Only one cell type measured (no denominator available from the same paper)
- The calibration specifically targets absolute cell count (e.g., tumor burden in cells)
- The ratio would require species not tracked in the model

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

**Observable** (singular, required):

The top-level `observable:` block describes a single experimental observable. (There is no `measurements:` array — exactly one observable per CalibrationTarget.) It contains:

1. **observable.species** (list) - Full model species accessed by `observable.code`:
   - Format: `['compartment.species']` (e.g., `['V_T.CD8', 'V_T.C1']`)
   - Must match species names in the model

2. **observable.constants** (list) - All conversion factors and reference values with units:
   - **REQUIRED** for any numeric constant with units used in `observable.code`
   - Each constant requires: `name`, `value`, `units`, `biological_basis`, `source_type`, and source-specific fields
   - **source_type** must be one of:
     - `reference_db` → also provide `reference_db_name` (must match reference_values.yaml entry)
     - `derived_from_reference_db` → also provide `reference_db_names` list (each must match)
     - `literature` → also provide `source_tag` (must match a defined source in this target)
   - **Never hardcode numbers with units** like `23.0 * ureg('mg/mL')` in `observable.code`
   - **No "modeling_assumption" allowed** — every constant must trace to a verifiable source
   - Access in code via: `constants['constant_name']`

3. **observable.auxiliary_parameters** (list, optional) - Measurement-bridging parameters that are *jointly inferred* with QSP parameters at calibration time:
   - Use ONLY when a real compartment / cross-species / measurement-scale gap exists between the literature measurement and the QSP species (e.g., serum cytokine concentration vs tumor-compartment concentration; mouse activity vs human equivalent; IHC H-score vs nM concentration).
   - Each auxiliary parameter requires: `name` (globally unique Python identifier; same name across cal targets ⇒ same theta draw), `group` (must match a declared group — see "Available Auxiliary Parameter Groups" below), `units` (Pint-parseable, defaults to `dimensionless`), and `biological_basis` (≥20 chars; explain *which* gap is being bridged and *how* `observable.code` uses the parameter).
   - Access in `observable.code` via the same `constants` dict — auxiliary draws are merged into `constants` per-simulation by the inference workflow before invoking the observable. **Do NOT also list the same name in `observable.constants`** — that's a hard validation error (name collision).
   - Cross-target consistency: if `f_serum_to_tumor_TGFb` appears on both the TGFβ and IL-12 cal targets, both must declare the same `group` and `units`. Different bridges across cytokines should use distinct names (`f_serum_to_tumor_TGFb`, `f_serum_to_tumor_IL12`, ...).
   - Naming convention: use a leading `f_` for ratios/dimensionless factors, and embed the bridged quantity in the name (`f_serum_to_tumor_<species>`, `f_mouse_to_human_<param>`, `f_ihc_to_nM_<species>`). The leading `f_` keeps the variable distinguishable from fixed `observable.constants`.
   - **Do NOT invent auxiliary parameters when no compartment / measurement gap exists.** A target whose source already matches the model compartment must NOT declare any auxiliary parameters — extra members would inject unidentified slack into the joint posterior.

4. **observable.code** (executable Python) - Computes the observable from species time series:
   - Function signature: `compute_observable(time, species_dict, constants, ureg)`
   - `time`: numpy array with day units (Pint Quantity)
   - `species_dict`: dict mapping species names to numpy arrays (Pint Quantities, one value per timepoint)
   - `constants`: dict mapping constant names to Pint Quantities (from `observable.constants` AND `observable.auxiliary_parameters` — both are looked up by the same key)
   - `ureg`: Pint UnitRegistry for conversions
   - Must return Pint Quantity (scalar or array) with units matching `observable.units` and `empirical_data.units`
   - **IMPORTANT**: Do NOT hardcode numbers with units. Use `constants` dict for all conversion factors.
   - **IMPORTANT**: Do NOT include time filtering logic. This function computes WHAT to measure. WHEN to measure is handled at the scenario/timepoint level.

5. **observable.units** (str) - Pint-parseable units of the observable output. Must match both `observable.code` return units and `empirical_data.units`.

6. **observable.support** (required) - Declares the mathematical support of the output:
   - `positive`: Output must be > 0 (densities, concentrations, volumes)
   - `non_negative`: Output must be ≥ 0 (counts)
   - `unit_interval`: Output must be in [0, 1] (fractions, proportions)
   - `positive_unbounded`: Output must be > 0, no upper bound (fold-changes, ratios)
   - `real`: Any real value (log-ratios, change scores)

7. **observable.experimental_denominator** / **observable.model_denominator_species** — describe what the experiment normalizes by, and which model species form the matching model-side denominator. **Conditionally required:** when the observable is a density or per-mass concentration (units like `cell/mm**2`, `pg/mg`, `cell/g`, etc., with `support: positive`), validation REQUIRES `experimental_denominator` to be set. Omitting it triggers a `value_error: "Observable with units='pg/mg' and support='positive' is a density but experimental_denominator is not set"` failure. Optional only for unitless ratios (`support: unit_interval`) or absolute counts.

**Example observable (absolute density with stroma correction):**
```yaml
observable:
  code: |
    def compute_observable(time, species_dict, constants, ureg):
        cd8 = species_dict['V_T.CD8']
        c_cells = species_dict['V_T.C1']
        area_per_cell = constants['area_per_cancer_cell']
        stroma_frac = constants['stromal_fraction']
        # Tissue area includes cancer cells AND stroma
        tumor_area = c_cells * area_per_cell / (1 - stroma_frac)
        return (cd8 / tumor_area).to('cell/mm**2')
  units: cell/mm**2
  species: ['V_T.CD8', 'V_T.C1']
  constants:
    - name: area_per_cancer_cell
      value: 2.27e-4
      units: mm**2/cell
      biological_basis: "From reference DB pdac_cancer_cell_diameter (17 μm) → π×(8.5 μm)² = 2.27e-4 mm²"
      source_type: derived_from_reference_db
      reference_db_names: [pdac_cancer_cell_diameter]
    - name: stromal_fraction
      value: 0.75
      units: dimensionless
      biological_basis: "PDAC tumors are highly desmoplastic with 60-90% stromal content. Reference DB consensus value 0.75."
      source_type: reference_db
      reference_db_name: pdac_stromal_fraction
  support: positive
  experimental_denominator: "mm^2 of tumor tissue section (cancer cells + stroma)"
  model_denominator_species: ['V_T.C1']
```

**Example observable (dimensionless ratio — preferred when available):**
```yaml
observable:
  code: |
    def compute_observable(time, species_dict, constants, ureg):
        treg = species_dict['V_T.Treg']
        total_t = (treg + species_dict['V_T.CD8'] + species_dict['V_T.Th']
                   + species_dict['V_T.CD8_exh'] + species_dict['V_T.Th_exh'])
        return (treg / total_t).to('dimensionless')
  units: dimensionless
  species: ['V_T.Treg', 'V_T.CD8', 'V_T.Th', 'V_T.CD8_exh', 'V_T.Th_exh']
  constants: []
  support: unit_interval
  experimental_denominator: "CD3+ T cells (all T cell subsets)"
  model_denominator_species: ['V_T.Treg', 'V_T.CD8', 'V_T.Th', 'V_T.CD8_exh', 'V_T.Th_exh']
```

**Example observable (compartment bridge via auxiliary parameter — serum→tumor TGFβ):**
```yaml
observable:
  code: |
    def compute_observable(time, species_dict, constants, ureg):
        # Model species V_T.TGFb is the BIOACTIVE pool in the tumor compartment.
        # The literature value is total TGFβ1 in human serum, so we (1) scale
        # to the bioactive fraction, (2) apply the serum:tumor compartment
        # bridge inferred jointly with QSP theta.
        tgfb_tumor = species_dict['V_T.TGFb']                       # nanomolar
        f_active = constants['active_fraction']                      # 0.10 (fixed)
        f_serum_to_tumor = constants['f_serum_to_tumor_TGFb']        # AUXILIARY draw
        predicted_serum = (tgfb_tumor * f_active) / f_serum_to_tumor
        return predicted_serum.to('nanomolar')
  units: nanomolar
  species: ['V_T.TGFb']
  constants:
    - name: active_fraction
      value: 0.10
      units: dimensionless
      biological_basis: "Typical bioactive TGFβ1 fraction in vivo (~10% of total)."
      source_type: literature
      source_tag: <SOME_REFERENCE>
  auxiliary_parameters:
    - name: f_serum_to_tumor_TGFb
      group: serum_to_tumor
      units: dimensionless
      biological_basis: >
        Serum:tumor concentration ratio for TGF-β1. The literature reports
        bioactive TGFβ1 in human PDAC serum, while the QSP species is the
        tumor-compartment bioactive pool — `observable.code` divides the
        compartment-corrected V_T.TGFb by this auxiliary factor to predict
        the serum measurement, with the inference jointly absorbing the
        bridging uncertainty.
  support: positive
```

The auxiliary parameter is consumed via `constants['f_serum_to_tumor_TGFb']` in the same dict as the fixed-constant `active_fraction`. The prior on the ratio (lognormal, location, spread) lives in `auxiliary_config.yaml` on the inference side and is NOT specified here — `observable.auxiliary_parameters` only declares the *member* and which group's prior it draws from.

**Key principles for `observable.code`:**
- Keep Pint units throughout calculation (see Pint Golden Rule below)
- Access only species listed in `observable.species`
- Return Pint Quantity (typically array with one value per timepoint)
- Do NOT include time filtering logic - compute over entire time series

### Distribution Code

**Data Flow (CRITICAL to understand):**

1. **Paper reports** statistics (mean, SD, median, IQR, range, etc.) → Put these in `inputs[]`
2. **distribution_code** uses inputs to run Monte Carlo → Produces median + CI95
3. **empirical_data** contains the COMPUTED values (from step 2), NOT the paper's reported values

**Example:** Paper reports "150 ± 25 cells/mm²" but doesn't report median/CI95:
- `inputs`: `[{name: "mean", value: 150}, {name: "sd", value: 25}]` ← Paper's values
- `distribution_code`: Runs MC sampling from normal(150, 25)
- `empirical_data.median`: `[149.94]` ← Computed from MC, matches code output (length-1 list for scalar)
- `empirical_data.ci95`: `[[100.79, 199.35]]` ← Computed from MC (list of `[lo, hi]` pairs)

**Requirements:**
- **Extract biological/experimental values via inputs** with source traceability
- **Universal constants OK as literals**: percentiles (2.5, 25, 75, 97.5), mathematical constants (π, 2), MC sample sizes (10000)
- **Use MC methods** (parametric bootstrap), NOT analytical approximations
- **Use Pint units** - inputs are pre-converted Pint Quantities, return Pint Quantities
- Function signature: `derive_distribution(inputs, ureg)` returns dict with `median_obs`, `ci95_lower`, `ci95_upper`

**IMPORTANT: What distribution_code is (and is NOT) for:**

`distribution_code` is **STATISTICAL ONLY** — it converts literature values (mean ± SD) into sampling distributions. It should NEVER contain:
- Model compartment volumes (V_T, V_C, V_P) → belong in `observable.code`
- Physical constants or conversion factors → belong in `observable.constants`
- ODE / dynamics — there is no ODE machinery on a CalibrationTarget; if your data needs ODE simulation to fit, it's a SubmodelTarget, not a CalibrationTarget

If you find yourself writing `V_T = 1.0 * ureg.milliliter` in `distribution_code`, STOP — you're confusing statistical derivation with model simulation.

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

**Dimensionless / digitized-patient pattern.** When inputs are dimensionless magnitudes (e.g., individual patient values digitized from a figure), use `ureg('dimensionless')` to attach units after computing on raw magnitudes:

```python
def derive_distribution(inputs, ureg):
    import numpy as np
    rng = np.random.default_rng(42)
    values = np.array([inputs[f'val_{i:02d}'].magnitude for i in range(1, 10)])
    n = len(values)
    boot_medians = [np.median(rng.choice(values, size=n, replace=True)) for _ in range(10000)]
    dl = ureg('dimensionless')
    return {
        'median_obs': np.median(values) * dl,
        'ci95_lower': np.percentile(boot_medians, 2.5) * dl,
        'ci95_upper': np.percentile(boot_medians, 97.5) * dl,
    }
```

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

When `observable.code` includes conversion factors (cells → volume, IHC score → density, cellularity adjustments):

1. **Document all conversion assumptions in `key_assumptions`**
   - Cell sizes, tissue densities, cellularity fractions, spherical approximations
   - Include numeric values and cite sources

2. **Extract uncertain conversion factors as inputs**
   - If cellularity varies widely (e.g., 10-50% in PDAC stroma), consider sampling in `distribution_code`
   - If cell size has reported uncertainty, propagate it through the calculation
   - **Example:** Sample cellularity from uniform(0.15, 0.35) rather than fixing at 0.25

3. **Assess impact of conversion uncertainty**
   - Note in `key_study_limitations` if conversion assumptions dominate uncertainty
   - **Example:** "Cellularity varies 10-50% in PDAC; assuming fixed 25% introduces unquantified error"

### Source Requirements

- **Primary source (singular)**: One paper, real DOI that resolves
- **No reuse**: Avoid studies already used for this observable: {{USED_PRIMARY_STUDIES}}
- **Verbatim snippets**: Exact quotes containing values (automatically verified)
- **Secondary sources**: Reference values, conversion factors (can be multiple)

### Sample Size

`empirical_data.sample_size` (int or List[int]) and `empirical_data.sample_size_rationale` (str) are required. Look for `n =`, `N =`, figure legends, patient counts, replicate counts. If unreported, back-calculate from SD/SEM (if both given), or use a conservative type-based estimate and note the uncertainty in `sample_size_rationale`.

**Reference example with all `empirical_data` fields:**
```yaml
empirical_data:
  median: [149.94]                       # List[float] — length 1 for scalar; matches index_values length for vector data
  ci95: [[100.79, 199.35]]               # List[List[float]] — one [lo, hi] pair per index point
  units: cell / mm**2
  sample_size: 42
  sample_size_rationale: "n=42 patients with resected PDAC tumors, stated in Table 1"
  inputs: [...]                           # List[EstimateInput] — paper-reported values used by distribution_code
  assumptions: [...]                      # Optional List[ModelingAssumption] — values NOT from the paper but needed for computation (e.g., n_mc_samples=10000, assumed_cv when not reported); each requires a rationale
  distribution_code: |
    def derive_distribution(inputs, ureg):
        ...
        return {'median_obs': ..., 'ci95_lower': ..., 'ci95_upper': ...}
```

There is **no `iqr` field** on `empirical_data`. `distribution_code` returns only `median_obs`, `ci95_lower`, `ci95_upper`.

### Experimental Context Block

`experimental_context` documents WHERE the source data came from. Required, even when source matches model context exactly. Common shape:

```yaml
experimental_context:
  species: human                          # SOURCE species — may differ from model
  system: clinical.resection              # how the data was obtained (clinical.resection, clinical.biopsy, in_vitro, etc.)
  indication: PDAC                        # cancer/disease the source paper studied
  treatment:
    history: [treatment_naive]            # list — see TreatmentHistory enum
    status: off_treatment                 # off_treatment | on_treatment
    specifier: null                       # optional drug name/class
  stage:                                  # optional, for clinical contexts
    extent: resectable                    # resectable | borderline_resectable | locally_advanced | metastatic
    burden: moderate                      # low | moderate | high
  mouse_subspecifier: null                # only when species=mouse: wild_type | immunocompromised | transgenic
  cell_lines: null                        # for in_vitro / cell-line studies
  culture_conditions: null                # for in_vitro studies (medium, duration_hours)
  tissue_source: null                     # e.g., "fresh resection", "frozen biopsy", "FFPE"
  assay_type: ELISA                       # IHC, flow_cytometry, ELISA, qPCR, scRNA-seq, etc.
```

The `Model Context (Target to Match)` section above describes the *target*; `experimental_context` describes the *actual* source. Mismatches are documented (not rejected) — flag them via `primary_data_source.source_relevance` (see "Optional Features").

### Narrative Fields

Three top-level narrative fields are required:

- **`study_interpretation`** (str, ≥1 paragraph) — what observable is being measured, how the experimental context maps to the model, key methodological points. Focus on scientific interpretation, not assumptions/limitations.
- **`key_assumptions`** (List[str], min 1 entry) — biological + statistical assumptions (e.g., `"CD8+ T cells include both effector and exhausted phenotypes"`, `"Normal distribution assumed for positive-only data"`, `"Mouse data assumed transferable to human with 5–10× rate reduction"`).
- **`key_study_limitations`** (List[str], can be empty) — issues that bias estimates or limit generalizability (e.g., `"Single-center cohort (n~15)"`, `"Values estimated from figures, not tabulated data"`, `"Bulk CD3+ assay, not CD8-specific"`).

### Epistemic Basis

Default `epistemic_basis: literature` covers the standard case (a primary publication directly measures the observable). Use `epistemic_basis: mechanistic` ONLY for biological-invariant priors that cannot be tied to a primary measurement (e.g., "untreated tumors do not spontaneously regress"). Mechanistic targets:
- May have `primary_data_source: null`
- Skip `value_snippet` validation
- Must include the mechanistic rationale in `key_assumptions`
- Should use deliberately wide CIs so the target nudges rather than dominates the likelihood
- Live in `calibration_targets/mechanistic/`, not the scenario directories

`mechanistic` is NOT a backdoor for unverified citations. If a paper exists, use `literature`.

**Worked mechanistic example** (from `tumor_diameter_fold_d90_gvax_vs_untreated`):

```yaml
epistemic_basis: mechanistic
primary_data_source: null
secondary_data_sources: []
key_assumptions:
  - >
    Mechanistic prior on within-θ fold-of-fold (cancels shared baseline growth).
    CI95 [0.70, 1.20] admits both modest growth suppression and no-effect baselines.
  - >
    Distribution: lognormal on the ratio. Composing two priors expands effective
    uncertainty consistent with the wider CI95 here.
empirical_data:
  median: [0.95]
  ci95: [[0.70, 1.20]]
  units: dimensionless
  sample_size: 1
  sample_size_rationale: >
    Mechanistic prior; sample_size=1 denotes a single soft-prior assertion.
  inputs: []                            # no extracted values
  assumptions: []
  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        rng = np.random.default_rng(42)
        lo, hi = 0.70, 1.20
        mu = 0.5 * (np.log(lo) + np.log(hi))
        sigma = (np.log(hi) - np.log(lo)) / (2.0 * 1.96)
        samples = rng.lognormal(mu, sigma, 100000) * ureg('dimensionless')
        return {
            'median_obs': np.median(samples),
            'ci95_lower': np.percentile(samples, 2.5),
            'ci95_upper': np.percentile(samples, 97.5),
        }
```

Note: `inputs: []` is allowed for mechanistic targets because the CI95 itself is the assertion. Numeric literals inside `distribution_code` (the CI bounds) are acceptable here only because they define the prior; they would be illegal in a literature target.

---

## Context

**Available model species:**
{{MODEL_SPECIES_WITH_UNITS}}

**Available reference values (curated constants for conversion factors):**
{{REFERENCE_DATABASE}}

Use these reference values in `observable.constants` when applicable (e.g., cell diameters, molecular weights, tissue densities). This avoids re-deriving standard physical/biological constants and ensures consistency across targets.

**Available Auxiliary Parameter Groups (for measurement-bridging — see `observable.auxiliary_parameters`):**
{{AUXILIARY_GROUPS}}

Each group declares a hierarchical prior shared by every member that references it. To use one, declare the bridging factor under `observable.auxiliary_parameters` with `group: <name>` (matching one of the listed groups exactly). Use these ONLY when a real compartment / cross-species / measurement-scale gap exists between the source measurement and the QSP species — never to introduce slack on a target that already matches the model context.

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
