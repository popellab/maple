# Goal

You are a research assistant helping to extract and document parameters for a quantitative systems pharmacology (QSP) immune oncology model.  
Your task is to create **comprehensive, reproducible metadata** for a model parameter by carefully analyzing scientific literature and experimental data.  

For this parameter, you must:

---

## Monte Carlo Parameter Estimation
1. **Generate MC samples:** Provide fully executable R code that generates a numeric vector called `mc_draws_canonical` with ≥2000 samples.  
2. **Canonical scale:** Always generate draws on the canonical scale specified for this parameter to ensure comparability across studies.  

3. **Bootstrap preferred:** Use bootstrap resampling when raw/digitizable data available.

4. **Uncertainty propagation:** Incorporate ALL sources of uncertainty, especially for composite parameters:
   - **Multiple measurements:** Use bootstrap resampling when combining multiple data points
   - **Composite parameters:** When parameter depends on multiple quantities, propagate uncertainty from each component
   - **Unit conversions:** Include uncertainty from conversion factors when applicable
   - **Model assumptions:** Account for parametric uncertainty when making distributional assumptions
5. **Required summary statistics:** Calculate and populate these 4 required fields:
   - `mu`: Mean of mc_draws_canonical  
   - `s2`: Variance of mc_draws_canonical
   - `natural_scale_mean`: Mean on natural (untransformed) scale
   - `natural_scale_ci95`: BCa 95% confidence interval on natural scale as [lower, upper]  

---

## Experimental Documentation
6. **Study overview:** Provide a concise narrative explaining the measurement approach, biological rationale, and how the parameter was derived.  
7. **Technical details:** Document essential assay and study design details (measurement method, instrumentation, sample size, replicates, controls, data source, processing, and key assumptions).  
8. **Streamlined format:** Focus only on details critical for parameter derivation and pooling decisions.  

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
16. **DOI preference:** Provide `doi_or_url` (prefer DOI over URL when available).
17. **Precise locations:** Give specific figure/table locations and exact text snippets where values originated.  

---

## Structure & Completeness
18. `study_overview` must provide a clear, non-redundant narrative without repeating technical_details.
19. `parameter_estimates.derivation_code_r` must generate `mc_draws_canonical` with proper uncertainty propagation.
20. Pooling weights must follow the rubric tables exactly (0–1) with concise justifications.
21. All sections (`technical_details`, `sources`, `key_study_limitations`) must be complete with consistent source attribution.  

---

**Key Requirements**
- The R code must define `mc_draws_canonical` (≥2000 samples, canonical scale).  
- **Bootstrap is the default** for uncertainty quantification. If bootstrapping is not possible, explain why and state the alternative.  
- Weights must follow the rubric tables exactly.  
- Metadata must be pooling-ready: everything needed for inverse-variance weighted pooling across studies.  

---

## Provided Template
{{TEMPLATE}}

## Example
{{EXAMPLES}}

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## CANONICAL_SCALE:
{{CANONICAL_SCALE}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

Fill out the YAML metadata template for this parameter.
