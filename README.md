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

Add the MAPLE MCP server to your coding agent (Claude Code, Codex, Cursor, etc.) — see [setup details](#setup). Then, from your model repo, ask the agent to extract a parameter:

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

MAPLE includes an MCP server that exposes the extraction schemas, enum references, validation tools, and a step-by-step workflow guide. This is the preferred way to fill out target YAMLs — working interactively with a coding agent, reading a paper together, and building the YAML incrementally with validation feedback at each step.

### Setup

Add the MCP server to your coding agent's config. For Claude Code, add to `.claude/settings.json`:

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

There's also a batch extraction CLI (`qsp-extract`) that sends a full paper to an LLM in one shot, but the interactive approach produces better results — the forward model parameterization and error model design usually need iteration.

### Built-in safeguards

Pydantic validators catch common extraction failures automatically:

- **Anti-hallucination** — extracted values must appear in `value_snippet`; snippets are verified against paper text via Europe PMC / Unpaywall
- **Unit validation** — all unit strings checked against Pint
- **Code validation** — forward model and observation code syntax and execution checks
- **DOI verification** — CrossRef resolution and metadata matching
- **Invisible characters** — catches zero-width spaces and other PDF copy-paste artifacts

---

## Docs

Schema details, validator reference, and inference pipeline internals are in [CLAUDE.md](CLAUDE.md).

## License

MIT
