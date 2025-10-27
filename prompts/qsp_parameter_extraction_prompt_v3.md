# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

{{EXISTING_STUDIES}}

For this parameter, you must:

---

## Monte Carlo Parameter Estimation

1. **Structured inputs:** Define all input values in the `inputs` list with source references
2. **Function-based code:** Provide Python code as a `derive_parameter(inputs)` function
3. **Bootstrap preferred:** Use bootstrap resampling when raw data available
4. **Uncertainty propagation:** Incorporate ALL sources of uncertainty:
   - **Multiple measurements:** Use bootstrap resampling when combining multiple data points
   - **Composite parameters:** When parameter depends on multiple quantities, propagate uncertainty from each component
   - **Unit conversions:** Include uncertainty from conversion factors when applicable
   - **Model assumptions:** Account for parametric uncertainty when making distributional assumptions
5. **Standard units:** Use standard unit formats (e.g., "1/day", "nM", "mg/L", "dimensionless" for counts/ratios)
6. **Required outputs:** Function must return dict with:
   - `mean_param`: Mean of Monte Carlo draws
   - `variance_param`: Variance of Monte Carlo draws
   - `ci95_param`: 95% percentile confidence interval as [lower, upper]

---

## Experimental Documentation

6. **Study overview (1-2 sentences):** WHAT parameter is being measured, WHY it's biologically relevant, and the overall approach
7. **Study design (1-2 sentences):** HOW the measurement was performed (assay type, sample size, key methods)
8. **Key assumptions (enumerated dict):** 3-5 critical assumptions only (e.g., distributional assumptions, model choices, data quality). Use format: `1: "Assumption text"`, `2: "Assumption text"`. Do NOT include trivial assumptions like "bootstrap samples are independent" or "conversion factors are standard".
9. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (3-6 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
10. **Key study limitations:** List critical limitations and their specific impact on reliability

---

## Source Separation and Provenance Tracking

**CRITICAL:** Separate sources into THREE categories:

**V3 SCHEMA REQUIREMENT: Text and table-based extraction ONLY.**
- Do NOT extract data from figures or graphs via digitization
- Use only numerical values explicitly stated in text or tables
- If critical data only appears in figures, note this in key_study_limitations

### Primary Data Sources
Original measurements from unique studies. These should NOT be reused across derivations.

Each source is a list entry with:
- `source_tag`: Short tag for referencing (e.g., "MARCHINGO2014", "SMITH2020")
- `title`: Full article title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null if not available)

Location and text snippets are in `inputs` (not here).

### Secondary Data Sources
Reference values, textbook data, established constants. Reuse is acceptable.

Each source is a list entry with:
- `source_tag`: Short tag for referencing (e.g., "ALBERTS2015")
- `title`: Reference title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null for textbooks)

Location and text snippets are in `inputs` (not here).

### Methodological Sources
Formulas, conversion factors, analysis methods. Reuse is expected.

Each source is a list entry with:
- `source_tag`: Short tag for referencing (e.g., "EFRON1993")
- `title`: Article/reference title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null if not available)
- `used_for`: What this method/formula was used for
- `method_description`: Brief description of the method or formula

**Important:**
- All VALUES, UNITS, LOCATIONS, and TEXT SNIPPETS appear in `parameter_estimates.inputs`
- Each input must have `value_snippet` (text showing the value) and `units_snippet` (text showing the units)
- Use `table_or_section` format like "Table 2" or "Methods" (no page numbers)
- Sources (primary/secondary) provide ONLY structured citations (title, first_author, year, doi)
- No duplication between inputs and sources sections
- **TEXT/TABLE ONLY**: No figure digitization in v3 schema

---

## Biological Relevance Weights

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

10. **Citation verification:** Verify all citations come from real, accessible publications
11. **Data location verification:** Cross-check table/text references contain the claimed data
12. **Biological plausibility:** Sanity-check parameter values against known biological ranges
13. **Input-source matching:** Every input must have a source_ref (or null if standard conversion/seed)
14. **Code verification:** Ensure derivation_code uses exactly the inputs defined in inputs list

---

## Structure & Completeness

16. `study_overview` (1-2 sentences): WHAT and WHY - high-level biological context
17. `study_design` (1-2 sentences): HOW - concrete experimental details
18. `key_assumptions` (enumerated dict): 3-5 critical assumptions only
19. `derivation_explanation` must provide clear step-by-step explanation with "ASSUMPTION N: ..." justifications embedded
20. `derivation_code` must be a function taking inputs dict, returning dict with mean_param/variance_param/ci95_param
21. Biological relevance weights must follow rubric tables exactly (0–1) with concise justifications
22. All sections must be complete with consistent source attribution
23. **Sources separated:** primary (original data) vs secondary (reference) vs methodological (methods/formulas)
24. **No duplication:** Values/units only in inputs, not in sources
25. **TEXT/TABLE ONLY:** Extract from text and tables only, no figure digitization

---

**Key Requirements**
- Python code must define `derive_parameter(inputs)` function returning required statistics
- **Bootstrap is the default** for uncertainty quantification
- All input values must reference a source (via source_ref field)
- Weights must follow rubric tables exactly
- **Separate primary/secondary data sources from methodological sources**
- **No value/unit duplication** between inputs and sources sections
- Metadata must be pooling-ready for inverse-variance weighted pooling

---

## Provided Template
{{TEMPLATE}}

## Example
{{EXAMPLES}}

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

Fill out the metadata template for this parameter.

**IMPORTANT: Return your response as JSON** (the template above is shown in YAML for readability, but respond with JSON):

```json
{
  "mathematical_role": "Describe the mathematical role...",
  "parameter_range": "positive_reals",
  "study_overview": "Brief 1-2 sentence summary of WHAT and WHY...",
  "study_design": "Brief 1-2 sentence summary of HOW...",
  "parameter_estimates": {
    "inputs": [
      {
        "name": "median_survival",
        "value": 0.45,
        "units": "years",
        "description": "Median survival from trial endpoint",
        "source_ref": "MARCHINGO2014",
        "value_table_or_section": "Table 2",
        "value_snippet": "Median survival was 0.45 years (95% CI: 0.3-0.6) in the treatment arm.",
        "units_table_or_section": "Table 2",
        "units_snippet": "Median survival was 0.45 years (95% CI: 0.3-0.6) in the treatment arm."
      },
      {
        "name": "plasma_clearance_reference",
        "value": 0.67,
        "units": "L/h",
        "description": "Reference plasma clearance rate for similar therapeutic class",
        "source_ref": "GOODMAN2018",
        "value_table_or_section": "Table 4.3",
        "value_snippet": "Typical plasma clearance for checkpoint inhibitors ranges from 0.5-0.8 L/h, with population mean of 0.67 L/h.",
        "units_table_or_section": "Table 4.3",
        "units_snippet": "Clearance values are reported in liters per hour (L/h) for consistency with standard pharmacokinetic conventions."
      },
      {
        "name": "conversion_factor",
        "value": 365.25,
        "units": "days/year",
        "description": "Days per year for unit conversion",
        "source_ref": null,
        "value_table_or_section": null,
        "value_snippet": null,
        "units_table_or_section": null,
        "units_snippet": null
      }
    ],
    "derivation_code": "import numpy as np\\n\\ndef derive_parameter(inputs):\\n    median_survival = inputs['median_survival']['value']\\n    ...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "1/day",
    "key_assumptions": {
      "1": "Exponential survival model (constant hazard rate)",
      "2": "Parameter uncertainty follows normal distribution (CLT with n=100)",
      "3": "Two-compartment model adequately describes drug distribution"
    }
  },
  "derivation_explanation": "**Step 1:** Extract data. ASSUMPTION: Exponential survival...\\n\\n**Step 2:** Calculate rate. ASSUMPTION: ...",
  "key_study_limitations": "- **Sample size:** ...\\n- **Measurement issues:** ...",
  "primary_data_sources": [
    {
      "source_tag": "MARCHINGO2014",
      "title": "Full article title",
      "first_author": "Marchingo",
      "year": 2014,
      "doi": "10.xxxx/xxxxx"
    }
  ],
  "secondary_data_sources": [
    {
      "source_tag": "GOODMAN2018",
      "title": "Goodman & Gilman's: The Pharmacological Basis of Therapeutics",
      "first_author": "Brunton",
      "year": 2018,
      "doi": null
    }
  ],
  "methodological_sources": [
    {
      "source_tag": "COMPARTMENTAL2008",
      "title": "Pharmacokinetic compartmental analysis using exponential fitting",
      "first_author": "Gibaldi",
      "year": 2008,
      "doi": "10.xxxx/xxxxx",
      "used_for": "Two-compartment clearance formula",
      "method_description": "Used Equation 3.14 for converting elimination rate"
    }
  ],
  "biological_relevance": {
    "species_match": {"value": 1.0, "justification": "Human study"},
    "system_match": {"value": 1.0, "justification": "In vivo"},
    "overall_confidence": {"value": 0.85, "justification": "Good design, minor caveats"},
    "indication_match": {"value": 1.0, "justification": "Exact PDAC match"},
    "regimen_match": {"value": 1.0, "justification": "Baseline untreated"},
    "biomarker_population_match": {"value": 0.85, "justification": "Close match"},
    "stage_burden_match": {"value": 0.65, "justification": "Earlier stage"}
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
