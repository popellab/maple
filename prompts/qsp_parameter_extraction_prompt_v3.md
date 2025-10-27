# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

{{EXISTING_STUDIES}}

For this parameter, you must:

---

## Monte Carlo Parameter Estimation

1. **Structured inputs:** Define all input values in the `inputs` list with source references
2. **Function-based code:** Provide Python code as a `derive_parameter(inputs)` function
3. **Bootstrap preferred:** Use bootstrap resampling when raw/digitizable data available
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
9. **Derivation explanation:** Step-by-step plain-language explanation of the Python code (5 steps recommended). Reference and justify assumptions using "ASSUMPTION N: ..." format where N matches the key from key_assumptions.
10. **Key study limitations:** List critical limitations and their specific impact on reliability

---

## Source Separation and Provenance Tracking

**CRITICAL:** Separate sources into THREE categories:

### Primary Data Sources
Original measurements from unique studies. These should NOT be reused across derivations.
- Citation, DOI
- `is_primary: true`
- Figure/table location
- Exact text snippet showing the measurement

### Secondary Data Sources
Reference values, textbook data, established constants. Reuse is acceptable.
- Citation, DOI
- `is_primary: false`
- Location in reference
- Relevant quote

### Methodological Sources
Formulas, conversion factors, analysis methods. Reuse is expected.
- Citation, DOI
- What it was used for
- Formula or method description
- Location and quote

**Important:**
- All actual VALUES and UNITS appear only in `parameter_estimates.inputs`
- Sources provide only citations, locations, and text evidence
- No duplication between inputs and sources sections

**Digitization metadata (when applicable):**
- If data was extracted from figures/graphs, include `digitized` section
- Provide tool name/version, raw data points, axis labels, and uncertainty estimate

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
11. **Data location verification:** Cross-check figure/table references contain the claimed data
12. **Digitized data quality:** For digitized data, re-extract values independently and flag discrepancies
13. **Biological plausibility:** Sanity-check parameter values against known biological ranges
14. **Input-source matching:** Every input must have a source_ref (or assumption if not measured)
15. **Code verification:** Ensure derivation_code uses exactly the inputs defined in inputs list

---

## Structure & Completeness

16. `study_overview` (1-2 sentences): WHAT and WHY - high-level biological context
17. `study_design` (1-2 sentences): HOW - concrete experimental details
18. `key_assumptions` (list): 3-5 concise bullet points of key assumptions
19. `derivation_explanation` must provide clear step-by-step explanation with "ASSUMPTION: ..." justifications embedded
20. `derivation_code` must be a function taking inputs dict, returning dict with mean_param/variance_param/ci95_param
21. Biological relevance weights must follow rubric tables exactly (0–1) with concise justifications
22. All sections must be complete with consistent source attribution
23. **Sources separated:** primary (original data) vs secondary (reference) vs methodological (methods/formulas)
24. **No duplication:** Values/units only in inputs, not in sources
25. **Digitization metadata:** Include when data extracted from figures/graphs

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
        "description": "Median survival from trial",
        "source_ref": "PRIMARY_STUDY",
        "assumption": null
      }
    ],
    "derivation_code": "import numpy as np\\n\\ndef derive_parameter(inputs):\\n    ...",
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "1/day",
    "key_assumptions": {
      "1": "Exponential survival model (constant hazard rate)",
      "2": "Parameter uncertainty follows normal distribution (CLT with n=100)",
      "3": "Digitization error approximately ±0.3 mg/L"
    }
  },
  "derivation_explanation": "**Step 1:** Extract data. ASSUMPTION: Exponential survival...\\n\\n**Step 2:** Calculate rate. ASSUMPTION: ...",
  "key_study_limitations": "- **Sample size:** ...\\n- **Measurement issues:** ...",
  "primary_data_sources": {
    "PRIMARY_STUDY": {
      "citation": "Author et al. Journal. Year;Vol:Pages.",
      "doi": "10.xxxx/xxxxx",
      "is_primary": true,
      "figure_or_table": "Figure 3B",
      "text_snippet": "Exact quote showing measurement",
      "digitized": {
        "tool": "WebPlotDigitizer v4.6",
        "points": [[0.5, 45.2], [1.0, 38.1], [2.0, 31.5]],
        "x_label": "Time (hours)",
        "y_label": "Concentration (mg/L)",
        "uncertainty": 0.3
      }
    }
  },
  "secondary_data_sources": {},
  "methodological_sources": {},
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
- `inputs` is an ARRAY of objects, each with name/value/units/description/source_ref/assumption
- Ensure every input has a corresponding source in primary/secondary/methodological sources