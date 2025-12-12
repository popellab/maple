## Source Guidelines

**Three source categories:**

**Primary:** Original experimental data. Use real DOIs. No reuse across derivations.
**Secondary:** Reference values, textbooks. Can reuse. Use `doi_or_url` field.

**Data extraction:**
- Prefer text and tables when available
- Values and snippets go in `inputs` section (not sources)
- **Units must be Pint-parseable** (e.g., `pg/mL`, `cells/mm^2`, `dimensionless`, `percent`). Do NOT use descriptive units like `pg/mg protein` or `% of CD45+ cells` - simplify to Pint-compatible strings
- **Only include inputs that are USED in derivation_code** - every input must flow into the computation
- Do NOT include boolean/qualitative indicators (e.g., `value: 1.0, units: "boolean"`) or confirmatory flags documenting study conditions (e.g., treatment-naive status, gating definitions). These belong in `study_design`, `key_assumptions`, or `key_study_limitations` instead

**Text Snippets (CRITICAL for automated verification):**

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

**For figure-extracted values (when text/table unavailable):**
- IS allowed, but with documentation requirements
- `value_snippet`: Describe the figure and your reading, e.g., "Figure 2A bar height ~45, error bar to ~60 (estimated from axis)"
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
