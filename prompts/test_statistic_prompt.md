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
   - `key_assumptions` - 3-5 critical assumptions as enumerated dict
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
- `value_snippet` - VERBATIM quote showing the value
- `units_table_or_section` - Where units are stated
- `units_snippet` - VERBATIM quote showing units

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

**Template:** {{TEMPLATE}}

---

# Response Format

Return JSON with all 11 fields:

```json
{
  "model_output": {
    "code": "import numpy as np\\n\\ndef compute_test_statistic(time, species_dict):\\n    ..."
  },
  "test_statistic_definition": "Math definition...",
  "study_overview": "What and why (1-2 sentences)",
  "study_design": "How measured (1-2 sentences)",
  "test_statistic_estimates": {
    "inputs": [
      {
        "name": "baseline_volume",
        "value": 100.0,
        "units": "mm³",
        "description": "Mean baseline tumor volume",
        "source_ref": "JONES2021",
        "value_table_or_section": "Table 1",
        "value_snippet": "baseline tumor volumes (mean 100 mm³, SD 15 mm³)",
        "units_table_or_section": "Table 1",
        "units_snippet": "volumes measured in mm³"
      }
    ],
    "derivation_code": "import numpy as np\\n\\ndef derive_distribution(inputs):\\n    ...",
    "median": 0.123,
    "iqr": 0.045,
    "ci95": [0.1, 0.15],
    "units": "days",
    "key_assumptions": {
      "1": "Exponential growth model applies during measured timeframe",
      "2": "Bootstrap adequately captures patient variability"
    }
  },
  "derivation_explanation": "**Step 1:** Extract volumes... **Step 2:** Fit growth... ASSUMPTION 1: ...",
  "key_study_limitations": "- **Small sample:** n=30 limits CI precision\\n- **Single institution:** ...",
  "primary_data_sources": [
    {
      "source_tag": "JONES2021",
      "title": "Full exact title from paper",
      "first_author": "Jones",
      "year": 2021,
      "doi": "10.1234/journal.2021.56789"
    }
  ],
  "secondary_data_sources": [],
  "methodological_sources": [],
  "validation_weights": {
    "species_match": {"value": 1.0, "justification": "Human, rubric=1.0"},
    "system_match": {"value": 1.0, "justification": "In vivo, rubric=1.0"},
    "overall_confidence": {"value": 0.85, "justification": "Good design, rubric=0.85"},
    "indication_match": {"value": 1.0, "justification": "Exact match, rubric=1.0"},
    "regimen_match": {"value": 0.85, "justification": "Same drug minor diffs, rubric=0.85"},
    "biomarker_population_match": {"value": 0.65, "justification": "Mixed population, rubric=0.65"},
    "stage_burden_match": {"value": 1.0, "justification": "Same stage, rubric=1.0"}
  }
}
```

**Key points:**
- Wrap in ```json block
- Use `\n` for line breaks, `\n\n` for paragraphs
- Raw Python (no ```python wrappers)
- Numbers as numbers not strings
- Text snippets must be verbatim quotes
- Every DOI will be validated - use real DOIs only
