# MAPLE

**Model-Aware Parameterization from Literature Evidence**

---

QSP models have many biological parameters, and most can't be measured directly in the clinical context being modeled. The relevant data is usually scattered across papers, often from different species or indications entirely. Turning that into informative priors is tedious, error-prone, and rarely done systematically.

MAPLE provides a structured pipeline for this. It uses LLMs to extract measurements from papers into validated YAML schemas (with anti-hallucination checks against source text), and scores how well each data source translates to the model context across eight axes (species, indication, TME compatibility, etc.). Each axis contributes a component to a translation sigma that widens the likelihood for that target during joint inference — so a mouse in vitro measurement constraining the same parameter as a human clinical measurement will naturally contribute less. The output is a set of marginal distributions plus a Gaussian copula that preserves posterior correlations for downstream SBI calibration.

---

## How it works

MAPLE fits into a two-stage calibration pipeline:

| Stage | Input | Method | Output |
|-------|-------|--------|--------|
| **1** (this repo) | Literature data + self-contained forward models | Joint MCMC (NumPyro/NUTS) | `submodel_priors.yaml` (marginals + copula) |
| **2** ([qsp-sbi](https://github.com/popellab/qsp-sbi)) | Copula priors + clinical data + full QSP simulator | SBI (SNPE-C) | Final posterior |

## Quick start

```bash
pip install maple-qsp[inference]   # includes NumPyro/JAX
```

MAPLE works with any AI tool that can access your files and run Python — coding agents (Claude Code, Codex, Cursor) via [MCP](#setup), or chat UIs with code execution (Claude Cowork, ChatGPT with Code Interpreter) via the [Python API](#python-api). From your model repo, ask the agent to extract a parameter:

> *"Use the MAPLE tool to help me extract the k_IL6_sec parameter"*

The agent loads the extraction guide, investigates the parameter in your model code (units, mechanistic role, Hill function inputs), searches literature for constraining data, verifies DOIs, fetches PDFs from Zotero, and then extracts the SubmodelTarget YAML with validation at each step. The agent drives the workflow and tells you what to do at each step (e.g., add a paper to Zotero, digitize a figure). Your job is to verify that extracted inputs match the paper, that the agent isn't making up assumptions, and that the forward and error models make sense for the parameter and data source.

Once you have targets, run joint inference:

```bash
maple-yaml-to-prior --priors pdac_priors.csv submodel_targets/ --output priors/ --plot
```

---

## What goes into a SubmodelTarget

Each YAML file is a structured extraction from one paper. It connects a literature measurement to one or more model parameters through a self-contained forward model:

<details>
<summary><b>Example: IL-2 degradation rate from half-life data</b></summary>

```yaml
target_id: k_IL2_deg_deriv001

inputs:
  - name: t_half_alpha
    value: 6.0
    units: minute
    source_ref: Lotze1985
    value_snippet: "a half-life of approximately 5 to 7 min"

calibration:
  parameters:
    - name: k_IL2_deg
      units: 1/minute
  forward_model:
    type: algebraic
    formula: "t_half = ln(2) / k"
    code: |
      def compute(params, inputs):
          import numpy as np
          return np.log(2) / params['k_IL2_deg']
  error_model:
    - name: halflife_obs
      units: minute
      uses_inputs: [t_half_alpha, t_half_beta]
      sample_size_input: n_patients
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            vals = [inputs['t_half_alpha'], inputs['t_half_beta']]
            mu, sigma = np.mean(np.log(vals)), np.std(np.log(vals), ddof=1)
            return rng.lognormal(mu, sigma, n_bootstrap)

source_relevance:
  indication_match: related
  species_source: human
  species_target: human
  source_quality: primary_human_clinical
  # ... (8-axis rubric → translation sigma applied in likelihood)
```

</details>

**Forward model types** include algebraic formulas, dose-response curves (`direct_fit`), power laws, and ODE systems — both structured types with analytical solutions (`exponential_growth`, `first_order_decay`, `logistic`, etc.) and arbitrary user-provided ODEs (`custom_ode`) integrated numerically via diffrax. The source relevance assessment maps to a translation sigma that inflates the likelihood during inference — so mouse data naturally gets less weight than human data constraining the same parameter.

**Nuisance parameters** can be marked `nuisance: true` when needed by the forward model but not part of the QSP model (e.g., a proliferation rate that helps constrain an activation rate). They carry their own inline prior, are sampled during MCMC, but are excluded from the output priors.

There's also a **CalibrationTarget** schema for clinical/in vivo observables (biopsies, blood draws) that require full model simulation — these feed into Stage 2.

---

## LLM-assisted extraction

MAPLE works with any AI tool that can run Python and access your files. There are two ways to connect it:

### MCP server (coding agents)

For Claude Code, Codex, Cursor, and other MCP-compatible agents. Add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "maple": {
      "command": "python",
      "args": ["-m", "maple.mcp_server"]
    }
  }
}
```

### Python API (chat UIs)

For Claude Cowork, ChatGPT with Code Interpreter, or any environment that can `pip install` and run Python. The same tools are available as plain functions:

```python
from maple.mcp_server import extract_target, validate_target, run_joint_inference

# Load the extraction guide
guide = extract_target("submodel_target")

# Validate a target YAML
report = validate_target("path/to/target.yaml", "pdac_priors.csv")

# Run joint inference
report = run_joint_inference("pdac_priors.csv", "submodel_targets/")
```

### Tools

| Tool | Purpose |
|------|---------|
| `extract_target(target_type)` | Load the full extraction guide (schema, workflow, enum values, hard rules) |
| `validate_target(yaml_path, priors_csv)` | Schema validation + NumPyro MCMC prior derivation + snippet verification |
| `run_joint_inference(priors_csv, submodel_dir)` | Joint MCMC across all targets with diagnostic report |
| `compare_inference(priors_csv, submodel_dir)` | Single-target vs joint inference comparison |
| `verify_dois(dois)` | Verify DOIs resolve via CrossRef, return metadata |
| `fetch_papers_from_zotero(dois)` | Copy PDFs from Zotero's local storage into paper directories |

### Typical workflow

> **Agent** and **You** labels indicate who drives each step.

1. **You** — Ask the agent to extract a parameter (e.g., *"use the MAPLE tool to extract k_IL6_sec"*)
2. **Agent** — Loads the extraction guide (`extract_target`)
3. **Agent** — Investigates the parameter in your model code: units, mechanistic role, Hill function inputs, interactions with other parameters
4. **Agent** — Searches literature for quantitative data that constrains the parameter; verifies DOIs via CrossRef
5. **You** — Add papers to Zotero; agent calls `fetch_papers_from_zotero` to pull PDFs
6. **Agent** — Reads the paper. If figures contain richer data (scatter plots, dose-response curves), asks you to digitize with WebPlotDigitizer
7. **Agent** — Builds the SubmodelTarget YAML incrementally, validating at each step
8. **You** — Review inputs, forward model, and error model. Check that values match the paper, assumptions are justified, and the model makes sense for the data
9. **Agent** — Runs `validate_target` — fixes schema, MCMC, or snippet errors
10. Iterate steps 7-9 until validation passes

For extracting many parameters at once, see the [batch extraction pipeline](#batch-extraction-pipeline) below.

### Built-in safeguards

Pydantic validators catch common extraction failures automatically:

- **Anti-hallucination** — extracted values must appear in `value_snippet`; snippets are verified against paper text via Europe PMC / Unpaywall
- **Unit validation** — all unit strings checked against Pint
- **Code validation** — forward model and observation code syntax and execution checks
- **DOI verification** — CrossRef resolution and metadata matching
- **Invisible characters** — catches zero-width spaces and other PDF copy-paste artifacts

---

## Batch extraction pipeline

For extracting many parameters at once, MAPLE supports a staged batch pipeline that automates the multi-step workflow across a set of targets. Each stage caches its results per-target, so you can rerun any stage for any subset without redoing work.

### Pipeline stages

```
Stage 1    Lit search          Web search for papers per target (parallel)
Stage 1b   PDF collection      Zotero DOI lookup + interactive fetch loop
Stage 2    Paper assessment     Read PDFs, assess data quality (parallel)
Stage 2b   Plan review          Single LLM call reviewing all plans together
           Digitization summary Prioritized list of figures to digitize
  --- human digitization step (WebPlotDigitizer) ---
Stage 3    Extract              Assemble SubmodelTarget YAMLs (parallel)
Stage 3b   Derivation review    Single LLM call checking scientific soundness
Stage 3c   Validate             MCMC prior derivation + unit checks + snippet matching
```

### How it works

**Input**: a CSV listing target parameters:
```csv
target_id,parameters,cancer_type,notes
k_IL2_sec,k_IL2_sec,PDAC,"Per-cell IL-2 secretion rate. Search for: ELISA, single-cell secretion rates."
k_vas_growth,k_vas_growth,PDAC,"Rate law: dK/dt = k_vas_growth * C_total * VEGF/(VEGF+VEGF_50). Search for: MVD growth kinetics."
```

The `notes` field guides the lit search agent. Include rate laws, search terms, and context about what kind of data would constrain the parameter. Richer notes produce better search results.

**Per-target caching**: each target gets a directory (`work/staged_extraction/{target_id}/`) with independently cached files:
- `lit_search_results.json` (stage 1)
- `assessment.json` (stage 2)
- `{target_id}_*_deriv001.yaml` (stage 3)

To rerun a specific stage for a specific target, delete its cache file. Other targets and stages are untouched.

### Lit search (stage 1)

An LLM agent with web search finds 3-5 papers per target with quantitative data matching the parameter's model role. Each candidate includes:
- DOI (validated against CrossRef)
- Relevance summary and mapping concerns
- Jointly constrainable parameters (other QSP params the paper could also constrain)

Notes in the targets CSV matter. A terse note like "angiogenesis rate" may return nothing, while a note including the rate law and specific search terms ("MVD growth kinetics, vascular doubling times") finds relevant papers.

### PDF collection (stage 1b)

PDFs are fetched from Zotero's local SQLite database by exact DOI lookup (case-insensitive). An interactive loop handles missing papers:
1. Auto-fetch from Zotero storage
2. Copy missing DOIs to clipboard for Zotero "Add by Identifier"
3. Press Enter to re-fetch, 'b' to open in browser for manual download, 's' to skip
4. Final summary of still-missing papers with clickable DOI links

### Paper assessment (stage 2)

Each paper is read (PDF attached to the LLM) and assessed for:
- Data availability and location (table, text, or figure)
- Mapping quality to the model parameter
- Digitization need and priority (`critical` / `helpful` / `optional` / `not_needed`)
- Paper role: `standalone`, `required_for_derivation`, `alternative`, or `validation_only`
- Forward model suggestion and jointly constrainable parameters

The output is an **extraction plan**: the minimal set of papers (and specific figures/tables) needed for one complete derivation, plus alternative plans.

### Plan review (stage 2b)

A single LLM call reviews all extraction plans together, checking for:
- Proxy measurements when direct data exists in an alternative
- Small sample sizes when larger datasets are available
- Excessive digitization burden when simpler alternatives exist
- Empty plans that need lit search reruns

Verdicts: `proceed`, `switch_to_alt` (swaps the plan in assessment.json), `rerun_lit_search` (deletes caches, appends search guidance to targets CSV), or `defer`.

### Digitization

After plan review, a prioritized digitization summary shows which figures need WebPlotDigitizer treatment. Items are ranked:
- **[REQUIRED]**: in the extraction plan and critical priority
- By priority: critical > helpful > optional
- Extraction plan items vs alternatives/validation

Place WPD CSV exports in `work/staged_extraction/{target_id}/digitized/{source_tag}/`. The pipeline reads these automatically during extraction.

### Extraction (stage 3)

The LLM assembles a SubmodelTarget YAML following the extraction plan. It sees:
- The extraction plan with explicit "FOLLOW THIS" instructions
- Plan review reasoning for why this plan was chosen
- Paper PDFs (filtered to plan papers only to avoid context overflow)
- Digitized data CSVs
- Parameter context from model_structure.json
- Prior sanity check (current median/sigma)

Output is validated against the SubmodelTarget schema before writing.

### Derivation review (stage 3b)

A single LLM call reviews all completed derivations for scientific soundness:
- Forward model appropriateness
- Input data fidelity and unit conversions
- Biological plausibility
- Derivation logic (circular reasoning, proxy assumptions)
- Cross-target consistency (contradictory assumptions, redundant constraints)

### Validation (stage 3c)

Mechanical validation per target:
- SubmodelTarget schema validation against model_structure.json
- MCMC prior derivation (NumPyro/NUTS)
- Snippet-in-paper verification
- Passing targets are copied to `calibration_targets/submodel_targets/`

### Example script

The pipeline is used via a thin config script in your model repo:

```python
# scripts/staged_extraction.py
from pathlib import Path
from functools import partial

from maple.extraction import (
    collect_missing_pdfs, make_agents, run_assess, run_complete,
    run_derivation_review, run_lit_search, run_plan_review,
    run_stage, run_validate, summarize_digitizations,
    write_assessment_report, write_dois_md,
)

# Config — edit these for your project
TARGETS_CSV = Path("notes/targets.csv")
MODEL_STRUCTURE = Path("model_structure.json")
MODEL_CONTEXT = Path("model_context.txt")
WORK_DIR = Path("work/staged_extraction")
ZOTERO_STORAGE = Path("~/Zotero/storage").expanduser()
PRIORS_CSV = Path("parameters/priors.csv")
MODEL = "gpt-5.1"
TARGET_RANGE = (0, 20)  # which rows of targets CSV to process

# Setup
model_context = MODEL_CONTEXT.read_text().strip()
lit_search_agent, assess_agent, plan_review_agent, complete_agent, derivation_review_agent = make_agents(MODEL, MAX_RETRIES=7)

# Run stages interactively (each stage caches per-target)
# Stage 1: Lit search → Stage 1b: PDF fetch → Stage 2: Assess
# → Stage 2b: Plan review → Digitization → Stage 3: Extract
# → Stage 3b: Derivation review → Stage 3c: Validate
```

See the [PDAC model repo](https://github.com/jeliason/pdac-build/blob/main/scripts/staged_extraction.py) for a complete working example.

---

## Docs

Schema details, validator reference, and inference pipeline internals are in [CLAUDE.md](CLAUDE.md).

## License

MIT
