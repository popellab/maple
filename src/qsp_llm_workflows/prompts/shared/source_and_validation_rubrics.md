## Source Guidelines

**Three source categories:**

**Primary:** Original experimental data. Use real DOIs. No reuse across derivations.
**Secondary:** Reference values, textbooks. Can reuse. Use `doi_or_url` field.

**Data extraction:**
- Prefer text and tables when available
- Values and snippets go in `inputs` section (not sources)
- Each input needs `value_snippet` and `units_snippet`
- **Only include inputs that are USED in derivation_code** - every input must flow into the computation
- Do NOT include boolean/qualitative indicators (e.g., `value: 1.0, units: "boolean"`) or confirmatory flags documenting study conditions (e.g., treatment-naive status, gating definitions). These belong in `study_design`, `key_assumptions`, or `key_study_limitations` instead

**For text/table values:**
- `value_snippet` and `units_snippet` must be VERBATIM quotes from the paper

**For figure-extracted values (when text/table unavailable):**
- IS allowed, but with documentation requirements
- `value_snippet`: Describe the figure and your reading, e.g., "Figure 2A bar height ~45, error bar to ~60 (estimated from axis)"
- `units_snippet`: Quote the axis label verbatim, e.g., "y-axis label: 'CD163+ cells/HPF'"
- Reduce `overall_confidence` by 0.1 for figure-extracted values
- Add to `key_assumptions`: "Value estimated from figure; exact numeric not reported in text/tables"

---

## Validation Weights

Assign [0-1] weights using these rubrics:

**Species:** 1.0=Human | 0.85=NHP | 0.65=Mouse | 0.45=Rat | 0.25=Non-mammal | 0.10=Irrelevant

**System:** 1.0=In vivo | 0.85=Ex vivo | 0.65=Organoid | 0.45=2D primary | 0.25=Cell line | 0.10=Biochemical

**Confidence:** 1.0=Rigorous, large N | 0.85=Good design | 0.65=Adequate | 0.45=Weak | 0.25=Major concerns | 0.10=Minimal

**Indication:** 1.0=Exact match | 0.85=Close subtype | 0.65=Adjacent tumor | 0.45=Distant biology | 0.25=Non-tumor | 0.10=Irrelevant

**Regimen:** 1.0=Exact match | 0.85=Same drug minor diffs | 0.65=Same MoA | 0.45=Partial relevance | 0.25=MoA related | 0.10=Non-representative

**Biomarker:** 1.0=Exact marker/phenotype match to model species | 0.85=Same marker, minor gating difference | 0.75=Well-validated single-marker proxy (e.g., PD-1 for exhaustion) | 0.65=Mixed population or multi-step proxy | 0.55=Weak proxy with known confounders (e.g., iNOS for M1) | 0.45=Mismatched marker | 0.25=Opposite phenotype | 0.10=No marker info

**Stage:** 1.0=Same stage | 0.85=Adjacent stage | 0.65=Partial overlap | 0.45=Very different | 0.25=Pre-malignant | 0.10=Not reported
