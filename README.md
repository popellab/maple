# MAPLE

**Model-Aware Parameterization from Literature Evidence**

QSP models have many biological parameters, and most can't be measured directly in the clinical context being modeled. The relevant data is usually scattered across papers, often from different species or indications entirely. Turning that into informative priors is tedious, error-prone, and rarely done systematically.

MAPLE provides a structured pipeline for this. It uses LLMs to extract measurements from papers into validated YAML schemas (with anti-hallucination checks against source text), and scores how well each data source translates to the model context across eight axes (species, indication, TME compatibility, etc.). Each axis contributes a component to a translation sigma that widens the likelihood for that target during joint inference — so a mouse in vitro measurement constraining the same parameter as a human clinical measurement will naturally contribute less. The output is a set of marginal distributions plus a Gaussian copula that preserves posterior correlations for downstream SBI calibration.

## How it works

MAPLE fits into a two-stage calibration pipeline:

**Stage 1** (this repo): Literature data with self-contained forward models → joint MCMC (NumPyro/NUTS) → `submodel_priors.yaml`

**Stage 2** ([qsp-sbi](https://github.com/popellab/qsp-sbi)): Copula priors + clinical data + full QSP simulator → SBI (SNPE-C) → final posterior

## Quick start

```bash
git clone https://github.com/popellab/maple.git
cd maple
pip install -e ".[inference]"   # includes NumPyro/JAX

# Run joint inference
maple-yaml-to-prior --priors pdac_priors.csv submodel_targets/ --output priors/ --plot
```

## What goes into a SubmodelTarget

Each YAML file is a structured extraction from one paper. It connects a literature measurement to one or more model parameters through a self-contained forward model:

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

Forward model types include algebraic formulas, dose-response curves (`direct_fit`), power laws, and several ODE types. The source relevance assessment maps to a translation sigma that inflates the likelihood during inference — so mouse data naturally gets less weight than human data constraining the same parameter.

There's also a **CalibrationTarget** schema for clinical/in vivo observables (biopsies, blood draws) that require full model simulation. These feed into Stage 2.

## LLM-assisted extraction

MAPLE includes an MCP server that exposes the extraction schemas, enum references, validation tools, and a step-by-step workflow guide. This is the preferred way to fill out target YAMLs — working interactively with Claude Code (or another MCP client), reading a paper together, and building the YAML incrementally with validation feedback at each step.

```bash
# Run the MCP server
python -m maple.mcp_server
```

There's also a batch extraction CLI (`qsp-extract`) that sends a full paper to an LLM in one shot, but the interactive approach tends to produce better results — it's easier to get the forward model right when you can iterate on it, and the source relevance assessment benefits from back-and-forth discussion about the experimental context.

Pydantic validators catch hallucinated values (snippet verification against paper text via Europe PMC), unit errors, invisible characters from PDF copy-paste, and other common extraction failures. These run both during interactive extraction and on batch output.

## Docs

Schema details, validator reference, and inference pipeline internals are in [CLAUDE.md](CLAUDE.md).

## License

MIT