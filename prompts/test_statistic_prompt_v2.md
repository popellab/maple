# Goal

You are a research assistant helping to extract and formalize test statistics for quantitative systems pharmacology (QSP) model validation from scientific literature.
Your task is to create **comprehensive, reproducible test statistic definitions** that quantify expected distributions of model-derived quantities based on experimental literature.

{{EXISTING_TEST_STATISTICS}}

For this test statistic, you must:

---

## Test Statistic Definition & Quantification
1. **Mathematical formalization:** Create a precise mathematical definition of how the test statistic is computed
2. **Biological interpretation:** Explain what the test statistic measures biologically and why it's relevant for model validation
3. **Literature data extraction:** Identify and extract quantitative measurements from text/tables that can be transformed into the expected distribution

---

## Statistical Distribution Characterization

**IMPORTANT:** The `model_output` code has already been provided by humans and is NOT part of your task. You only need to generate the statistical distribution from literature.

4. **Structured inputs:** Define all input values in the `inputs` list with source references
5. **Function-based code:** Provide Python code as a `derive_distribution(inputs)` function
6. **Bootstrap preferred:** Use bootstrap resampling when raw data available
7. **Uncertainty propagation:** Incorporate ALL sources of uncertainty:
   - **Multiple measurements:** Use bootstrap resampling when combining multiple data points
   - **Composite test statistics:** When test statistic depends on multiple quantities, propagate uncertainty from each component
   - **Measurement error:** Include uncertainty from assay variability when applicable
   - **Model assumptions:** Account for parametric uncertainty when making distributional assumptions
8. **Standard units:** Use standard unit formats (e.g., "percent", "mm³", "cells/µL", "dimensionless")
9. **Required outputs:** Function must return dict with:
   - `mean_stat`: Mean of Monte Carlo draws
   - `variance_stat`: Variance of Monte Carlo draws
   - `ci95_stat`: 95% percentile confidence interval as [lower, upper]

---

## Experimental Documentation

10. **Study overview (1-2 sentences):** WHAT test statistic is being measured, WHY it's biologically relevant for validation, and the overall approach
11. **Study design (1-2 sentences):** HOW the measurement was performed (assay type, sample size, key methods)
12. **Key assumptions (enumerated dict):** 3-5 critical assumptions only (e.g., distributional assumptions, model choices, data quality). Use format: `1: "Assumption text"`, `2: "Assumption text"`. Do NOT include trivial assumptions.
13. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (3-6 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
14. **Key study limitations:** List critical limitations and their specific impact on reliability

---

## Source Separation and Provenance Tracking

**CRITICAL:** Separate sources into THREE categories:

**V2 SCHEMA REQUIREMENT: Text and table-based extraction ONLY.**
- Do NOT extract data from figures or graphs via digitization
- Use only numerical values explicitly stated in text or tables
- If critical data only appears in figures, note this in key_study_limitations

### Primary Data Sources
Original measurements from unique studies. These should NOT be reused across derivations.

Each source is a list entry with:
- `source_tag`: Short tag for referencing (e.g., "TOPALIAN2012", "EISENHAUER2009")
- `title`: Full article title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null if not available)

Location and text snippets are in `inputs` (not here).

### Secondary Data Sources
Reference values, established constants, guidelines. Reuse is acceptable.

Each source is a list entry with:
- `source_tag`: Short tag for referencing
- `title`: Reference title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null)

Location and text snippets are in `inputs` (not here).

### Methodological Sources
Formulas, conversion factors, analysis methods. Reuse is expected.

Each source is a list entry with:
- `source_tag`: Short tag for referencing
- `title`: Article/reference title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null if not available)
- `used_for`: What this method/formula was used for
- `method_description`: Brief description of the method or formula

**Important:**
- All VALUES, UNITS, LOCATIONS, and TEXT SNIPPETS appear in `test_statistic_estimates.inputs`
- Each input must have `value_snippet` (text showing the value) and `units_snippet` (text showing the units)
- Use `table_or_section` format like "Table 2" or "Methods" (no page numbers)
- Sources provide ONLY structured citations (title, first_author, year, doi)
- No duplication between inputs and sources sections
- **TEXT/TABLE ONLY**: No figure digitization in v2 schema

---

## Validation Weights

Assign a **fixed weight in [0,1]** for each dimension with 1-2 sentence justification.
Use the following rubrics (tables). Do not invent new scales.

### Species Weight
| Value | Definition |
|-------|------------|
| 1.00 | Human |
| 0.85 | Non-human primate |
| 0.65 | Mouse (syngeneic/GEMM) |
| 0.45 | Rat or other small mammal |
| 0.25 | Non-mammalian vertebrate surrogate |
| 0.10 | Non-vertebrate/irrelevant |

### System Weight
| Value | Definition |
|-------|------------|
| 1.00 | In vivo (intact immune system) |
| 0.85 | Ex vivo human tissue/primary cells |
| 0.65 | Organoid / 3D co-culture |
| 0.45 | 2D primary cell culture |
| 0.25 | Stable cell line |
| 0.10 | Biochemical/reductionist assay |

### Overall Confidence
| Value | Definition |
|-------|------------|
| 1.00 | Large N, rigorous controls, validated assay |
| 0.85 | Good design, minor caveats |
| 0.65 | Adequate, some limitations |
| 0.45 | Weak design, limited validation |
| 0.25 | Major concerns |
| 0.10 | Minimal documentation |

### Indication Match
| Value | Definition |
|-------|------------|
| 1.00 | Exact disease/subtype match |
| 0.85 | Closely related subtype |
| 0.65 | Adjacent solid tumor |
| 0.45 | Distant tumor, distinct biology |
| 0.25 | Non-tumor immune/inflammatory |
| 0.10 | Irrelevant context |

### Regimen Match
| Value | Definition |
|-------|------------|
| 1.00 | Exact drug, dose, schedule, route |
| 0.85 | Same drug, minor dosing/schedule diffs |
| 0.65 | Same MoA class, similar PK |
| 0.45 | Different regimen, partial relevance |
| 0.25 | MoA related, PK not comparable |
| 0.10 | Non-representative exposure |

### Biomarker / Population Match
| Value | Definition |
|-------|------------|
| 1.00 | Exact biomarker profile |
| 0.85 | Close match, 1 key biomarker differs |
| 0.65 | Mixed population with subset match |
| 0.45 | Mismatched biomarker context |
| 0.25 | Opposite biomarker/immune status |
| 0.10 | No relevant biomarker info |

### Stage / Burden Match
| Value | Definition |
|-------|------------|
| 1.00 | Same stage/burden |
| 0.85 | Adjacent stage, similar biology |
| 0.65 | Earlier stage with partial overlap |
| 0.45 | Very different stage/progression |
| 0.25 | Pre-malignant / non-cancer |
| 0.10 | Stage not reported/irrelevant |

---

## Data Quality & Validation

15. **Citation verification:** Verify all citations come from real, accessible publications
16. **Data location verification:** Cross-check table/text references contain the claimed data
17. **Biological plausibility:** Sanity-check test statistic values against known biological ranges
18. **Input-source matching:** Every input must have a source_ref (or null if standard conversion/seed)
19. **Code verification:** Ensure derivation_code uses exactly the inputs defined in inputs list

---

## Structure & Completeness

20. `test_statistic_definition`: Mathematical definition of the test statistic
21. `study_overview` (1-2 sentences): WHAT and WHY - high-level biological context
22. `study_design` (1-2 sentences): HOW - concrete experimental details
23. `key_assumptions` (enumerated dict): 3-5 critical assumptions only
24. `derivation_explanation` must provide clear step-by-step explanation with "ASSUMPTION N: ..." justifications embedded
25. `derivation_code` must be a function taking inputs dict, returning dict with mean_stat/variance_stat/ci95_stat
26. Validation weights must follow rubric tables exactly (0–1) with concise justifications
27. All sections must be complete with consistent source attribution
28. **Sources separated:** primary (original data) vs secondary (reference) vs methodological (methods/formulas)
29. **No duplication:** Values/units only in inputs, not in sources
30. **TEXT/TABLE ONLY:** Extract from text and tables only, no figure digitization

---

**Key Requirements**
- Python code must define `derive_distribution(inputs)` function returning required statistics
- **Bootstrap is the default** for uncertainty quantification
- All input values must reference a source (via source_ref field)
- Weights must follow rubric tables exactly
- **Separate primary/secondary data sources from methodological sources**
- **No value/unit duplication** between inputs and sources sections
- **model_output code is NOT your responsibility** - it's provided by humans

---

## Provided Context

### Model Information
{{MODEL_CONTEXT}}

### Scenario Context
{{SCENARIO_CONTEXT}}

### Required Species with Units
The human-provided `model_output` code computes the test statistic from these model species:

{{REQUIRED_SPECIES_WITH_UNITS}}

**Note:** You do not need to write code that uses these species - that code is already provided in `model_output`. Your task is to generate the expected distribution from literature.

### Derived Species Description
{{DERIVED_SPECIES_DESCRIPTION}}

### Template
{{TEMPLATE}}

### Examples
{{EXAMPLES}}

Fill out the test statistic template for this biological expectation and experimental context.

---

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON):

```json
{
  "test_statistic_definition": "Mathematical definition...",
  "study_overview": "Brief 1-2 sentence summary of WHAT and WHY...",
  "study_design": "Brief 1-2 sentence summary of HOW...",
  "test_statistic_estimates": {
    "inputs": [
      {
        "name": "response_rate",
        "value": 0.28,
        "units": "dimensionless",
        "description": "Objective response rate from clinical trial",
        "source_ref": "TOPALIAN2012",
        "value_table_or_section": "Table 2",
        "value_snippet": "Among patients with melanoma, the objective response rate was 28%...",
        "units_table_or_section": "Table 2",
        "units_snippet": "Objective response rate was 28% (95% CI: 18-40%)..."
      }
    ],
    "derivation_code": "import numpy as np\\n\\ndef derive_distribution(inputs):\\n    ...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "dimensionless",
    "key_assumptions": {
      "1": "Binomial sampling adequately models patient heterogeneity",
      "2": "Imaging measurement error is normally distributed with CV=8%",
      "3": "Single-arm response rate is comparable to control-adjusted response"
    }
  },
  "derivation_explanation": "**Step 1:** Extract data. ASSUMPTION 1: Binomial sampling...\\n\\n**Step 2:** Calculate rate. ASSUMPTION 2: ...",
  "key_study_limitations": "- **Sample size:** ...\\n- **Measurement issues:** ...",
  "primary_data_sources": [
    {
      "source_tag": "TOPALIAN2012",
      "title": "Full article title",
      "first_author": "Topalian",
      "year": 2012,
      "doi": "10.xxxx/xxxxx"
    }
  ],
  "secondary_data_sources": [],
  "methodological_sources": [],
  "validation_weights": {
    "species_match": {"value": 1.0, "justification": "Human study"},
    "system_match": {"value": 1.0, "justification": "In vivo"},
    "overall_confidence": {"value": 0.85, "justification": "Good design, minor caveats"},
    "indication_match": {"value": 1.0, "justification": "Exact melanoma match"},
    "regimen_match": {"value": 1.0, "justification": "Anti-PD-1 monotherapy"},
    "biomarker_population_match": {"value": 0.85, "justification": "Close match"},
    "stage_burden_match": {"value": 0.65, "justification": "Advanced disease"}
  }
}
```

Requirements for JSON response:
- Wrap entire response in ```json code block tags
- Use proper JSON syntax (all strings quoted, proper escaping)
- Use `\n` for line breaks in multi-line strings
- For `derivation_code`: provide ONLY the raw Python code without ```python wrapper tags
- Use `\n\n` (double newline) to separate paragraphs or list items
- Numeric values should be actual numbers, not strings
- `inputs` is an ARRAY of objects with: name, value, units, description, source_ref, value_table_or_section, value_snippet, units_table_or_section, units_snippet
- Ensure every input with a source_ref has a corresponding source in primary/secondary/methodological sources
- Do NOT include "Exact quote:" prefix in snippets - just the text itself
- Do NOT include page numbers in value_table_or_section or units_table_or_section - just "Table X" or "Section name"
- **Do NOT generate model_output code** - that is provided separately by humans
