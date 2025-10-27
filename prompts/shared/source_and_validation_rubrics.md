## Source Separation and Provenance Tracking

**CRITICAL:** Separate sources into THREE categories:

**TEXT/TABLE EXTRACTION ONLY:**
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
- `source_tag`: Short tag for referencing (e.g., "COMPARTMENTAL2008")
- `title`: Article/reference title
- `first_author`: First author last name
- `year`: Publication year
- `doi`: DOI (or null if not available)
- `used_for`: What this method/formula was used for
- `method_description`: Brief description of the method or formula

**Important:**
- All VALUES, UNITS, LOCATIONS, and TEXT SNIPPETS appear in inputs section
- Each input must have `value_snippet` (text showing the value) and `units_snippet` (text showing the units)
- Use `table_or_section` format like "Table 2" or "Methods" (no page numbers)
- Sources provide ONLY structured citations (title, first_author, year, doi)
- No duplication between inputs and sources sections

---

## Validation/Relevance Weight Rubrics

Assign a **fixed weight in [0,1]** for each dimension with 1-2 sentence justification.
Use the following rubrics. Do not invent new scales.

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
