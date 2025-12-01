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

# Scientific Soundness Checklist

Before finalizing your extraction, verify:

**1. Biological Validity**
- Does the measured entity actually represent what the test statistic claims? (e.g., cDC1/cDC2 are lineage subsets, NOT maturation states)
- If using a proxy, is it well-established in the literature?
- Weak proxies should reduce overall_confidence to ≤0.7 and be documented in limitations

**2. Cross-Modality Harmonization**
- Avoid combining flow cytometry + IHC unless absolutely necessary
- If unavoidable: document ALL conversion factors in assumptions, set overall_confidence ≤0.6

**3. Cascading Assumptions**
- Count your inferred values and assumed conversion factors
- 2-3 assumptions: acceptable with documentation
- 4+ assumptions: high risk - consider finding better data
- Each assumption should reduce overall_confidence by ~0.05-0.1

**4. Plausibility Check**
- Does your final median value make biological sense?
- Does ratio of input medians ≈ output median?
- Cross-check against other studies if possible
- Red flags: extreme values, fractions >1, ratios that seem off by orders of magnitude

**5. Honest Confidence Scores**
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
10. **methodological_sources** - Formulas/methods (doi_or_url field)
11. **validation_weights** - Quality scores for 7 dimensions (see rubrics below)

---

# Technical Specs

## Model Output Code
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

## Inputs Structure
Each input needs:
- `name`, `value`, `units`, `description`
- `source_ref` - References a source below
- `value_table_or_section` - Where the value appears
- `value_snippet` - VERBATIM quote showing the value (see snippet rules below)
- `units_table_or_section` - Where units are stated
- `units_snippet` - VERBATIM quote showing units (see snippet rules below)

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

Generate test statistic metadata following all requirements above.

**Key points:**
- Use `\n` for line breaks, `\n\n` for paragraphs in text fields
- Python code should be plain text (no markdown code fences within the code strings)
- Numbers as numbers not strings
- Text snippets must follow the rules in "Text Snippets" section above (verbatim, no table reconstruction, no LaTeX)
- Every DOI will be validated - use real DOIs only
