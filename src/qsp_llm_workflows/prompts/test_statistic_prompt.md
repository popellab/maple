# Task

Extract test statistics for QSP model validation from scientific literature. You'll find papers with experimental data, extract measurements, and create reproducible statistical distributions with uncertainty quantification.

---

# Finding Sources

**Most test statistics are DERIVED from underlying measurements, not directly reported.**

For this test statistic, find 1-2 real published papers that report:
- Raw measurements you can use to compute the test statistic (e.g., tumor volumes, cell counts, time points)
- Sample sizes and variability information for uncertainty quantification

**Common derivation patterns:**
- Tumor doubling time → Tumor volumes at multiple timepoints → Fit exponential growth
- Cell population ratios → Individual cell counts for each population → Compute ratio with uncertainty
- Response rates → Number of responders + total sample size → Binomial uncertainty
- Fold changes → Baseline + endpoint values → Compute change with propagated uncertainty

**Source requirements:**
- Use REAL DOIs that resolve at https://doi.org/ (I will validate these)
- Verify title, first author, and year match the DOI metadata
- Extract verbatim text snippets showing the values you use
- Find NEW sources - avoid studies already used for this test statistic:

{{USED_PRIMARY_STUDIES}}

---

# Data Completeness Requirement

**You MUST extract actual numeric values from literature. Pure placeholders are NOT acceptable.**

**Minimum requirements for a usable estimate:**
- [ ] At least ONE numeric value for each key component (e.g., numerator AND denominator for ratios)
- [ ] Sample size for at least one component

**Preferred but NOT required:**
- Variability measure (SD, SEM, IQR, range, or CI) - if missing, you MAY assume reasonable CV based on similar biological measurements (document this assumption)
- Same-study data - cross-study combinations ARE acceptable with proper documentation

**Handling missing variability:**
If a study reports mean/median but no dispersion:
1. Search for variability in similar PDAC studies with the same marker
2. If found, borrow the CV (coefficient of variation) and document the source
3. If not found, assume a conservative CV (e.g., 50-100% for immune cell counts) and document this assumption
4. Reduce `overall_confidence` by 0.1-0.15 for borrowed/assumed variability

**What IS acceptable:**
- Cross-study combinations with documented independence assumption
- Borrowed variability from related studies
- Wide confidence intervals reflecting genuine uncertainty
- Using min/median/max when only range is reported

**What is NOT acceptable (red flags):**
- "pending numeric extraction" - extract now or report failure
- "conservative prior" with no literature anchor whatsoever
- "placeholder distribution" - every distribution must trace to at least one extracted value
- Hard-coded parameters (e.g., `Beta(3,22)`) not derived from any input values

**If truly no usable data exists:**
Only after exhausting the options above, report: "Insufficient data available" and suggest alternative test statistics that ARE measurable.

---

# Data Source Hierarchy

When searching for measurements, prefer sources in this order:

**Tier 1 (Best):** Same indication, same compartment, same measurement modality
- Example: PDAC tumor IHC for intratumoral cell densities

**Tier 2 (Good):** Same indication, adjacent compartment OR different modality
- Example: PDAC blood flow cytometry (for tumor, need to justify transfer)
- Example: PDAC tumor scRNA-seq (different modality than IHC)

**Tier 3 (Acceptable with justification):** Related indication, same compartment
- Example: Other GI cancers for PDAC (e.g., colorectal)
- Requires explicit biology justification

**Tier 4 (Last resort):** Pan-cancer or non-cancer reference
- Example: General solid tumor immune composition
- Reduce `indication_match` to ≤0.65

**Never acceptable:**
- Cell line data for in vivo test statistics
- Mouse data without explicit species scaling factors
- Pure assumptions without any literature anchor

## Cytokine/Chemokine Data Sources

For tumor cytokine measurements (V_T.IL6, V_T.IL10, V_T.IFNg, V_T.TGFb, V_T.CCL2, V_T.CXCL12, V_T.IL12, V_T.PDGF, etc.), the compartment/matrix hierarchy below applies **in addition to** the indication hierarchy above.

**CRITICAL: Always search for PDAC-specific data first within each compartment tier.** Only fall back to cross-indication data (e.g., CRC, breast) if PDAC data is truly unavailable for that compartment type.

**Compartment preference order:**

**Tier 1 (Best):** PDAC tumor interstitial fluid (TIF) - direct TME measurement
- Obtained by centrifugation or microdialysis of resected PDAC tumors
- Units: pg/mL - directly comparable to model (after nM→pg/mL conversion)

**Tier 2 (Good):** PDAC tumor tissue homogenate/lysate
- Common in literature, reported as pg/mg protein or pg/mg tissue
- Requires unit conversion (see below)
- Includes both extracellular and some intracellular cytokine pools
- **Known PDAC sources exist** (e.g., Bellone et al. 2006 measured IL-6, IL-10, TGF-β in PDAC tissue homogenates)

**Tier 3 (Acceptable with caveats):** Tumor tissue/TIF from related indications (CRC, breast, ovarian)
- Only use if PDAC-specific data unavailable after thorough search
- Requires biological justification for cross-indication transfer
- Reduce `indication_match` to ≤0.65

**Tier 4 (Last resort for cytokines):** Serum/plasma
- Reflects systemic levels, NOT tumor microenvironment concentrations
- Tumor:serum ratios vary widely (10-1000×) depending on cytokine
- Only use if no tumor tissue data exists

**Confidence score adjustments for cytokine data tiers:**

| Data Tier | `system_match` | `indication_match` | `overall_confidence` penalty |
|-----------|----------------|-------------------|------------------------------|
| Tier 1 (PDAC TIF) | 1.0 | 1.0 | None |
| Tier 2 (PDAC Homogenate) | 0.85-0.9 | 1.0 | -0.05 to -0.1 (unit conversion) |
| Tier 3 (Cross-indication) | 0.8-0.9 | ≤0.65 | -0.15 to -0.25 (indication + compartment) |
| Tier 4 (Serum/plasma) | ≤0.5 | varies | -0.2 to -0.3 (compartment mismatch) |

**Note:** Cross-indication data (Tier 3) should ONLY be used after exhausting PDAC-specific sources. If using CRC or other GI cancer data for PDAC, explicitly document why PDAC data was not found.

**Unit conversion for tissue homogenate data:**

To convert pg/mg protein → pg/mL (model-compatible):
```
pg/mL ≈ pg/mg protein × [mg protein / mL tumor volume]
```

Typical conversion factor: **~100 mg protein/mL** for dense tumors like PDAC (range: 50-150).

Example: If literature reports IL-6 = 50 pg/mg protein:
- Point estimate: 50 × 100 = 5,000 pg/mL
- Propagate uncertainty: assume lognormal with σ ≈ 0.4 (reflecting 2-3× uncertainty in protein content)

```python
# Example derivation code for tissue homogenate conversion
protein_content = rng.lognormal(np.log(100), 0.4, size=N)  # mg protein/mL, ~2-3x uncertainty
cytokine_pg_per_mg = 50  # from literature
cytokine_pg_per_mL = cytokine_pg_per_mg * protein_content
```

Document this conversion in `key_assumptions` and note the additional uncertainty in `key_study_limitations`.

---

# Scientific Soundness Checklist

Before finalizing your extraction, verify:

**1. Biological Validity**
- Does the measured entity actually represent what the test statistic claims? (e.g., cDC1/cDC2 are lineage subsets, NOT maturation states)
- If using a proxy, is it well-established in the literature?
- Weak proxies should reduce overall_confidence to ≤0.7 and be documented in limitations

**2. Proxy Marker Validation**

If using a proxy marker (e.g., PD-1 for exhaustion, iNOS for M1 macrophages):
- [ ] State the gold standard definition (e.g., "true exhaustion = PD-1+TIM-3+LAG-3+")
- [ ] Cite literature justifying the proxy (e.g., "PD-1 enriches for tumor-reactive CD8+")
- [ ] Quantify the proxy's limitations:
  - Does it OVER-estimate? (e.g., PD-1 includes activated cells)
  - Does it UNDER-estimate? (e.g., iNOS misses some M1 macrophages)
  - Does it capture NON-TARGET cells? (e.g., iNOS+ neutrophils)
- [ ] Reduce `biomarker_population_match` to ≤0.75 for single-marker proxies
- [ ] Document in `key_assumptions` which marker(s) define the proxy

**3. Cross-Modality Harmonization**
- Avoid combining flow cytometry + IHC unless absolutely necessary
- If unavoidable: document ALL conversion factors in assumptions, set overall_confidence ≤0.6

**4. Cross-Study Integration**

Combining measurements from different patient cohorts is ACCEPTABLE when necessary, but requires documentation.

When combining data from different studies:
- [ ] Document explicitly: "Numerator from Study A (n=X), denominator from Study B (n=Y)"
- [ ] Acknowledge independence assumption in `key_assumptions`
- [ ] Reduce `overall_confidence` by 0.1-0.15 for cross-study combinations
- [ ] Note in limitations: potential demographic/methodological differences

Preferred hierarchy (but lower tiers ARE acceptable):
1. Same study, same patients (best) - no confidence penalty
2. Same study, different patients (good) - no confidence penalty
3. Different studies, same institution (acceptable) - reduce confidence by 0.05
4. Different studies, different institutions (acceptable with documentation) - reduce confidence by 0.1-0.15

**Cross-study combinations require extra caution when:**
- The two quantities are known to be correlated (e.g., CD8 and Treg infiltration) - note this in limitations
- Sample sizes differ by >10x - weight toward larger study or note limitation
- Measurement modalities differ (flow vs IHC) - see Cross-Modality Harmonization

**Cross-study is often the only option** for ratios where no single paper reports both components. This is acceptable - just document it properly.

**5. Cascading Assumptions and Unit Conversions**
- Count your inferred values and assumed conversion factors
- **Unit conversions count as assumptions** (e.g., area fraction → cell count)
- 2-3 assumptions: acceptable with documentation
- 4+ assumptions: high risk - consider finding better data
- Each assumption should reduce overall_confidence by ~0.05-0.1

Required documentation for unit conversions:
- Source of conversion factor (literature reference OR explicit assumption)
- Uncertainty in conversion factor (must be propagated in derivation_code)
- Alternative approaches considered

Example (BAD - hard-coded without uncertainty):
```python
tot_cells = 8500  # assumed
```

Example (GOOD - documented with uncertainty propagation):
```python
# Total nucleated cell density prior: 8500 cells/mm² (95% CI: 5000-14000)
# Based on PDAC histology literature (no single source; methodological prior)
# Propagated as lognormal with sigma=0.35
tot_cells = rng.lognormal(np.log(8500), 0.35, size=N)
```

**6. Plausibility Check**
- Does your final median value make biological sense?
- Does ratio of input medians ≈ output median?
- Cross-check against other studies if possible
- Red flags: extreme values, fractions >1, ratios that seem off by orders of magnitude

Confidence interval reasonableness:
- CI95 range (upper/lower) should typically be <100x for biological ratios
- If CI95 range >100x:
  - This is acceptable but note in limitations that uncertainty is high
  - Set `overall_confidence` ≤ 0.6
  - Wide CIs are BETTER than refusing to provide an estimate
- If CI95 range >1000x:
  - Still provide the estimate, but note in limitations
  - Consider whether alternative test statistics might be more constraining

**Important:** A wide CI reflecting genuine biological/measurement uncertainty is scientifically honest and useful for SBI. Do NOT refuse to provide estimates just because uncertainty is high.

**7. Honest Confidence Scores**
- 0.85-1.0: Direct measurements, no proxies, large sample
- 0.70-0.84: Minor proxy OR small sample, otherwise solid
- 0.50-0.69: Weak proxy OR cross-modality OR 2-3 cascading assumptions
- <0.50: Multiple significant issues - consider if extraction is justified

---

# What You'll Generate

1. **model_output** - Python function computing test statistic from model simulation
2. **test_statistic_definition** - Mathematical definition
3. **study_overview** - What's measured and why (1-2 sentences)
4. **study_design** - How it was measured (1-2 sentences)
5. **test_statistic_estimates**:
   - `inputs` - Extracted values with source references and verbatim text snippets
   - `derivation_code` - Python function deriving distribution with bootstrap/Monte Carlo
   - `median`, `iqr`, `ci95`, `units` - Statistical outputs (using outlier-robust statistics)
   - `key_assumptions` - 3-5 critical assumptions as list with number and text
6. **derivation_explanation** - Step-by-step explanation referencing assumptions
7. **key_study_limitations** - Critical limitations affecting reliability
8. **primary_data_sources** - Papers with data (real DOIs required)
9. **secondary_data_sources** - Reference values (doi_or_url field)
10. **validation_weights** - Quality scores for 7 dimensions (see rubrics below)

---

# Technical Specs

## Model Output Code

**Species Alignment Requirement:**

Before writing code, verify the mapping between:
- Literature measurement (e.g., "PD-1+ CD8+ cells")
- Model species (e.g., "V_T.CD8_exh")

Document the alignment in `test_statistic_definition`:
- If exact match: state "Direct measurement of model species"
- If proxy: state "Literature proxy (X) maps to model species (Y) because..."

Your code MUST use the exact species names provided in "Available model species" below.
If the literature measures something different, document the mapping assumption.

```python
import numpy as np

def compute_test_statistic(time, species_dict):
    """Compute test statistic from model simulation."""
    # Extract species, interpolate, compute metric
    return test_statistic_value  # float
```

## Derivation Code
```python
import numpy as np

def derive_distribution(inputs):
    """Derive expected distribution from literature data."""
    # Extract input values
    # Bootstrap/Monte Carlo for uncertainty
    # Propagate through computations
    # Use outlier-robust statistics (median/IQR instead of mean/variance)
    return {
        'median_stat': float,
        'iqr_stat': float,
        'ci95_stat': [lower, upper]
    }
```

## Self-Verification Requirement

Before submitting, mentally execute your derivation code:

1. Compute the median using input values directly (e.g., for ratio: numerator_median / denominator_median)
2. Compare to your reported `median` value
3. If they differ by >20%, explain why (e.g., Jensen's inequality for ratios of distributions)

**Include in derivation_explanation:**
"Point estimate check: [calculation] ≈ [reported median] [explanation if different]"

Example: "Point estimate check: 112.8 / 8500×0.183 ≈ 0.073, but Monte Carlo median is 0.0125 due to the heavy right tail of the denominator distribution (Jensen's inequality)."

## Inputs Structure
Each input needs:
- `name`, `value`, `units`, `description`
- `source_ref` - References a source below
- `value_table_or_section` - Where the value appears
- `value_snippet` - VERBATIM quote showing the value (see snippet rules below)
- `units_table_or_section` - Where units are stated
- `units_snippet` - VERBATIM quote showing units (see snippet rules below)

**Only include inputs that are USED in derivation_code.** Every input must flow into the statistical computation.

**Do NOT include:**
- Boolean/qualitative indicators (e.g., `value: 1.0, units: "boolean (1=yes)"`)
- Confirmatory flags that just document study conditions (e.g., treatment-naive status, gating definitions)
- Metadata that doesn't contribute numeric values to the derivation

These conditions belong in `study_design`, `key_assumptions`, or `key_study_limitations` - not as inputs.

## Text Snippets (CRITICAL for automated verification)

Text snippets are automatically verified against the full paper text. Follow these rules strictly:

1. **VERBATIM only**: Copy exact text from the paper. Never paraphrase, summarize, or reconstruct.
2. **No table reconstruction**: Do NOT create artificial table notation like `CD8^{+} | ... | 17 (9-30)`. Tables are flattened when we extract text, so this format won't match.
3. **Use continuous text spans**: Find a short, continuous phrase that contains the value. For table data, the snippet should be just the cell value and any immediately adjacent text, e.g., `"17 (9-30)"` not a reconstructed row.
4. **Include context when helpful**: A few surrounding words help locate the snippet, e.g., `"median survival of 18.2 months"` is better than just `"18.2"`.
5. **Avoid LaTeX formatting**: Write `CD8+` not `CD8^{+}`. Write subscripts inline: `CO2` not `CO_{2}`.
6. **Keep snippets short**: 5-50 words is ideal. Long snippets are harder to match exactly.
7. **For units**: Find where units are explicitly stated, e.g., `"expressed as cells per high-power field"` or `"measured in ng/mL"`.

**Good snippet examples:**
- `"median CD8+ density was 17 (IQR 9-30) cells/HPF"` ✓
- `"n = 137 patients"` ✓
- `"tumor volume measured in mm³"` ✓

**Bad snippet examples:**
- `"CD8^{+} | No neoadjuvant | 17 (9-30)"` ✗ (reconstructed table, LaTeX)
- `"The study found elevated levels"` ✗ (no actual value)
- `"approximately 17"` ✗ (paraphrased, paper says "17 (9-30)")

## Sources Structure

**Primary (real DOIs required):**
```json
{
  "source_tag": "SMITH2020",
  "title": "Full paper title matching DOI metadata",
  "first_author": "Smith",
  "year": 2020,
  "doi": "10.1234/journal.2020.12345"
}
```

**Secondary/Methodological (doi_or_url field):**
Same structure but use `doi_or_url` instead of `doi` (can be DOI, URL, or null)

---

# Validation Rubrics

Assign weights [0-1] with brief justification:

{{SOURCE_AND_VALIDATION_RUBRICS}}

---

# Context

**Model:** {{MODEL_CONTEXT}}

**Scenario:** {{SCENARIO_CONTEXT}}

**Available model species:**
{{REQUIRED_SPECIES_WITH_UNITS}}

**Test statistic description:**
{{DERIVED_SPECIES_DESCRIPTION}}

---

# Temporal Mapping Requirement

When the scenario specifies a measurement timepoint (e.g., "day 7", "week 12"):

- [ ] Acknowledge that surgical/biopsy data represents a snapshot, not a specific model day
- [ ] Justify why the clinical timepoint approximates the model timepoint
- [ ] Add assumption: "Surgical specimen immune composition approximates model steady-state at day X"

**For baseline (no treatment) scenarios:**
The model's early timepoints (e.g., "day 7") typically represent immune equilibration before treatment. Clinical data from treatment-naive resections is appropriate, but note in limitations:
- Patients may have had tumors for months/years before resection
- Immune composition at resection ≠ immune composition at diagnosis
- The "day 7" model state represents a quasi-steady-state, not literal 7 days of tumor growth

**For treatment scenarios:**
Match literature timepoints to model timepoints as closely as possible:
- If model measures at day 14 post-treatment, find clinical data at ~2 weeks
- Document any temporal mismatch in assumptions
- Larger temporal mismatches (>50% difference) should reduce `regimen_match`

---

Generate test statistic metadata following all requirements above.

**Key points:**
- Use `\n` for line breaks, `\n\n` for paragraphs in text fields
- Python code should be plain text (no markdown code fences within the code strings)
- Numbers as numbers not strings
- Text snippets must follow the rules in "Text Snippets" section above (verbatim, no table reconstruction, no LaTeX)
- Every DOI will be validated - use real DOIs only
