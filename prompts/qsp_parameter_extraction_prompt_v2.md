# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.

{{EXISTING_STUDIES}}

For this parameter, you must:

---

## Monte Carlo Parameter Estimation
1. **Generate MC samples:** Provide fully executable R code that generates a numeric vector called `mc_draws` with ≥2000 samples.
2. **Bootstrap preferred:** Use bootstrap resampling when raw/digitizable data available.

3. **Uncertainty propagation:** Incorporate ALL sources of uncertainty, especially for composite parameters:
   - **Multiple measurements:** Use bootstrap resampling when combining multiple data points
   - **Composite parameters:** When parameter depends on multiple quantities, propagate uncertainty from each component
   - **Unit conversions:** Include uncertainty from conversion factors when applicable
   - **Model assumptions:** Account for parametric uncertainty when making distributional assumptions
4. **Required summary statistics:** Calculate and populate these 4 required fields:
   - `mean`: Mean of mc_draws
   - `variance`: Variance of mc_draws
   - `ci95`: 95% percentile confidence interval as [lower, upper]
   - `units`: Units string for the parameter

---

## Experimental Documentation
5. **Study overview:** Provide a concise narrative explaining the measurement approach, biological rationale, and how the parameter was derived.
6. **Technical details:** Document essential assay and study design details (measurement method, instrumentation, sample size, replicates, controls, data source, processing, and key assumptions).
7. **Derivation explanation:** Provide a step-by-step plain-language explanation of the R derivation code (5 steps recommended).
8. **Streamlined format:** Focus only on details critical for parameter derivation and pooling decisions.

---

## Source Separation and Provenance Tracking

**CRITICAL:** You must separate sources into two categories and track contribution weights:

### Data Sources
These are studies that contributed **actual measurements** used in your derivation.

For EACH data source, you must provide:
- **Study identification:** Citation, DOI, figure/table location
- **Data extracted:** List ALL specific data points extracted from this study:
  - Description: What measurement was extracted
  - Value: The numerical value (or "qualitative" if not quantitative)
  - Units: Units of the measurement
  - Figure/table: Exact location
  - Text snippet: Quote showing where this came from
  - **weight_in_synthesis:** How much did this specific data point contribute to your final parameter estimate? (0-1 scale)
    - Example: If you averaged 3 studies equally → each gets weight 0.33
    - Example: If you did weighted average with 50%, 30%, 20% → weights are 0.5, 0.3, 0.2
    - **This is CRITICAL for overlap analysis** - allows us to determine correlation strength when derivations share sources

### Methodological Sources
These are studies that provided **formulas, conversion factors, or methods** but not actual data.

For EACH methodological source, you must provide:
- Citation and DOI
- What it was used for (e.g., "Molecular weight conversion", "Apoptotic index formula")
- The formula or method used
- Location and quote

**Why this matters:** When multiple derivations share data sources (not methodological sources), they become correlated. Tracking `weight_in_synthesis` helps quantify this correlation.

---

## Pooling Weights Assessment

Assign a **fixed weight in [0,1]** for each dimension, and provide a 1–2 sentence justification.
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
9. **Citation verification:** Verify all citations and snippets come from real, accessible publications.
10. **Data location verification:** Cross-check figure/table references contain the claimed data at specified locations.
11. **Digitized data quality:** For digitized data, re-extract values independently and flag any discrepancies or resolution issues.
12. **Biological plausibility:** Sanity-check parameter values against known biological ranges for the target system.
13. **Data completeness:** Assess missing data, exclusions, and potential selection bias in reported results.
14. **Limitation impact assessment:** Categorize how each limitation specifically affects parameter reliability and pooling weight.

---

## Source Attribution & Formatting
15. **Consistent source tagging:** Reference all sources by tag throughout ALL sections - no unsupported claims.
16. **DOI preference:** Provide `doi` (prefer DOI over URL when available).
17. **Precise locations:** Give specific figure/table locations and exact text snippets where values originated.
18. **Weight calculation:** Ensure `weight_in_synthesis` values for all data extractions sum to approximately 1.0 for the derivation.

---

## Structure & Completeness
19. `study_overview` must provide a clear, non-redundant narrative without repeating technical_details.
20. `derivation_explanation` must provide a clear step-by-step plain-language explanation of the R code.
21. `derivation_code_r` must generate `mc_draws` with proper uncertainty propagation.
22. Pooling weights must follow the rubric tables exactly (0–1) with concise justifications.
23. All sections (`technical_details`, `derivation_explanation`, `data_sources`, `methodological_sources`, `key_study_limitations`) must be complete with consistent source attribution.
24. **Sources must be separated:** `data_sources` (actual measurements) vs `methodological_sources` (formulas/methods)
25. **Weights required:** Every data extraction must have `weight_in_synthesis` field

---

**Key Requirements**
- The R code must define `mc_draws` (≥2000 samples).
- **Bootstrap is the default** for uncertainty quantification. If bootstrapping is not possible, explain why and state the alternative.
- Weights must follow the rubric tables exactly.
- **Separate data sources from methodological sources** - this is critical for overlap analysis
- **Provide weight_in_synthesis for each data extraction** - quantifies how much each source contributed
- Metadata must be pooling-ready: everything needed for inverse-variance weighted pooling across studies.

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
  "study_overview": "This study measures...",
  "technical_details": "**Measurement:** ...\\n**Study design:** ...",
  "parameter_estimates": {
    "mean": 0.123,
    "variance": 0.001,
    "ci95": [0.1, 0.15],
    "units": "1/day"
  },
  "derivation_explanation": "**Step 1:** ...\\n\\n**Step 2:** ...",
  "derivation_code_r": "set.seed(123)\\nB <- 5000\\nmc_draws <- rnorm(B, mean=0.5, sd=0.1)",
  "pooling_weights": {
    "species_match": {"value": 1.0, "justification": "Human study"},
    "system_match": {"value": 1.0, "justification": "In vivo"},
    "overall_confidence": {"value": 0.85, "justification": "Good design, minor caveats"},
    "indication_match": {"value": 1.0, "justification": "Exact PDAC match"},
    "regimen_match": {"value": 1.0, "justification": "Baseline untreated"},
    "biomarker_population_match": {"value": 0.85, "justification": "Close biomarker match"},
    "stage_burden_match": {"value": 0.65, "justification": "Earlier stage"}
  },
  "key_study_limitations": "- **Sample size:** ...\\n- **Measurement issues:** ...",
  "data_sources": {
    "MARCHINGO2014": {
      "citation": "Marchingo JM et al. Science. 2014;346(6213):1123-1127.",
      "doi": "10.1126/science.1260044",
      "data_extracted": [
        {
          "description": "CD8+ T cell proliferation rate measured by CFSE dilution",
          "value": 0.48,
          "units": "1/day",
          "figure_or_table": "Figure 3B",
          "text_snippet": "CD8+ T cells proliferated at a rate of 0.48 per day",
          "weight_in_synthesis": 0.6
        },
        {
          "description": "T cell doubling time",
          "value": 2.2,
          "units": "days",
          "figure_or_table": "Figure 3C",
          "text_snippet": "Doubling time was approximately 2.2 days",
          "weight_in_synthesis": 0.4
        }
      ]
    }
  },
  "methodological_sources": {
    "GENERAL_METHOD": {
      "citation": "Reference for CFSE analysis method",
      "doi": "10.xxxx/method",
      "used_for": "CFSE dilution analysis methodology",
      "formula_or_method": "Standard flow cytometry gating",
      "figure_or_table": "Methods",
      "text_snippet": "CFSE dilution was analyzed using standard gates"
    }
  }
}
```

Requirements for JSON response:
- Wrap your entire response in ```json code block tags
- Use proper JSON syntax (all strings quoted, proper escaping)
- Use `\\n` for line breaks in multi-line strings (NOT Markdown or other formatting)
- For `derivation_code_r`: provide ONLY the raw R code without ```r wrapper tags
- Use `\\n\\n` (double newline) to separate paragraphs or list items
- Numeric values should be actual numbers, not strings (except placeholders)
- `data_extracted` is an ARRAY of objects, each with required `weight_in_synthesis` field
- Ensure `weight_in_synthesis` values across all data extractions sum to ~1.0
