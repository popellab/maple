## Source Guidelines

**Three source categories:**

**Primary:** Original experimental data. Use real DOIs. No reuse across derivations.
**Secondary:** Reference values, textbooks. Can reuse. Use `doi_or_url` field.
**Methodological:** Formulas, methods. Can reuse. Use `doi_or_url` field + `used_for` + `method_description`.

**Data extraction:**
- Text and tables only (NO figure digitization)
- Values and snippets go in `inputs` section (not sources)
- Each input needs `value_snippet` and `units_snippet` (verbatim quotes)

---

## Validation Weights

Assign [0-1] weights using these rubrics:

**Species:** 1.0=Human | 0.85=NHP | 0.65=Mouse | 0.45=Rat | 0.25=Non-mammal | 0.10=Irrelevant

**System:** 1.0=In vivo | 0.85=Ex vivo | 0.65=Organoid | 0.45=2D primary | 0.25=Cell line | 0.10=Biochemical

**Confidence:** 1.0=Rigorous, large N | 0.85=Good design | 0.65=Adequate | 0.45=Weak | 0.25=Major concerns | 0.10=Minimal

**Indication:** 1.0=Exact match | 0.85=Close subtype | 0.65=Adjacent tumor | 0.45=Distant biology | 0.25=Non-tumor | 0.10=Irrelevant

**Regimen:** 1.0=Exact match | 0.85=Same drug minor diffs | 0.65=Same MoA | 0.45=Partial relevance | 0.25=MoA related | 0.10=Non-representative

**Biomarker:** 1.0=Exact profile | 0.85=1 key diff | 0.65=Mixed population | 0.45=Mismatched | 0.25=Opposite | 0.10=No info

**Stage:** 1.0=Same stage | 0.85=Adjacent stage | 0.65=Partial overlap | 0.45=Very different | 0.25=Pre-malignant | 0.10=Not reported
