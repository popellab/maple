# MAPLE — Model-Aware Parameterization from Literature Evidence

[![Tests](https://github.com/popellab/maple/actions/workflows/test.yml/badge.svg)](https://github.com/popellab/maple/actions/workflows/test.yml)

Extract calibration targets from scientific literature for quantitative systems pharmacology (QSP) model calibration. Uses structured YAML schemas with Pydantic validation and joint Bayesian inference (NumPyro) to produce informative priors for downstream SBI calibration.

## Installation

```bash
git clone https://github.com/popellab/maple.git
cd maple
pip install -e .

# For joint inference pipeline (NumPyro/JAX)
pip install -e ".[inference]"
```

## Two Schemas, One Goal

Maple provides two complementary schemas for extracting calibration targets from literature. Both produce validated YAML files that feed into Bayesian inference.

| | **SubmodelTarget** | **CalibrationTarget** |
|---|---|---|
| **Use case** | In vitro / preclinical data with isolated submodels | Clinical / in vivo data requiring full model context |
| **Forward model** | Self-contained ODE or algebraic model | Full QSP model simulation |
| **Key fields** | `inputs`, `calibration` (parameters, forward_model, error_model) | `observable`, `scenarios`, `empirical_data` (distribution_code) |
| **Inference** | Joint MCMC (NumPyro) → marginals + copula | SBI (SNPE-C) in qsp-sbi |
| **Validation script** | `scripts/validate_submodel_target.py` | `scripts/validate_calibration_target.py` |

## SubmodelTarget

For in vitro and preclinical data where a small submodel (ODE or algebraic) connects extracted literature values to model parameters.

**Structure:**
- `inputs` — Values extracted from papers with full provenance (snippets, source refs)
- `calibration.parameters` — Model parameter names and units (priors come from CSV)
- `calibration.forward_model` — Typed ODE or algebraic model (exponential growth, first-order decay, Michaelis-Menten, direct_fit dose-response, power_law scaling, etc.)
- `calibration.error_model` — Bootstrap observation code; likelihood family inferred automatically

```yaml
target_id: psc_proliferation_PDAC_deriv001

inputs:
  - name: fold_increase_mean
    value: 4.37
    units: dimensionless
    source_ref: schneider_2001
    value_snippet: "PDGF increased DNA synthesis 4.37 ± 0.89-fold"

calibration:
  parameters:
    - name: k_apsc_prolif
      units: 1/day
  forward_model:
    type: exponential_growth
    rate_constant: k_apsc_prolif
    state_variables:
      - name: N
        units: dimensionless
        initial_condition: {value: 1.0, rationale: "Normalized"}
    independent_variable: {name: time, units: day, span: [0, 3]}
  error_model:
    - name: fold_increase
      units: dimensionless
      uses_inputs: [fold_increase_mean, fold_increase_sd]
      sample_size_input: n_samples
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            mean = inputs['fold_increase_mean']
            sd = inputs['fold_increase_sd']
            return rng.lognormal(np.log(mean), sd / np.sqrt(sample_size), n_bootstrap)
```

### Validation & Extraction

```bash
# Validate
python scripts/validate_submodel_target.py \
  --model-structure model_structure.json target.yaml

# Extract from literature via LLM
qsp-extract targets.csv \
  --type submodel_target \
  --model-structure model_structure.json \
  --model-context model_context.txt \
  --output-dir metadata-storage
```

## CalibrationTarget

For clinical and in vivo observables (e.g., tumor cell densities, immune cell counts from patient biopsies) where the experimental context must be carefully documented because it may differ from the model context.

**Structure:**
- `observable` — What is being measured (species, units, compartment, support type)
- `scenarios` — Experimental conditions with intervention details
- `empirical_data` — Literature-extracted inputs and Monte Carlo `distribution_code` that derives median/CI95
- `source_relevance` — Formal assessment of indication match, species translation, TME compatibility
- `experimental_context` — Species, indication, stage, treatment history

```yaml
calibration_target_id: cd8_tumor_density_PDAC

observable:
  species: CD8_T
  units: cells/mm^2
  support: positive
  compartment: tumor.primary
  aggregation_type: spatial_density

scenarios:
  - name: baseline
    intervention: {type: none, description: "Treatment-naive"}

empirical_data:
  median: [42.0]
  ci95: [[12.0, 138.0]]
  units: cells/mm^2
  sample_size: 45
  inputs:
    - name: cd8_density_mean
      value: 42.0
      units: cells/mm^2
      value_snippet: "Mean CD8+ T cell density was 42 cells/mm²"
  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        rng = np.random.default_rng(42)
        mean = inputs['cd8_density_mean']
        sd = inputs['cd8_density_sd']
        samples = rng.normal(mean.magnitude, sd.magnitude, 10000) * mean.units
        return {
            'median_obs': np.median(samples),
            'ci95_lower': np.percentile(samples, 2.5),
            'ci95_upper': np.percentile(samples, 97.5),
        }
```

### Validation & Extraction

```bash
# Validate
python scripts/validate_calibration_target.py \
  --species-units species_units.json target.yaml

# Extract from literature via LLM
qsp-extract targets.csv \
  --type calibration_target \
  --output-dir metadata-storage
```

## Submodel Prior Inference

Run joint Bayesian inference across SubmodelTargets to produce informative priors:

```bash
# Joint inference across all submodel targets
maple-yaml-to-prior --priors pdac_priors.csv submodel_targets/ \
    --output priors/ --plot
```

The pipeline:
1. Loads broad starting priors from CSV
2. Builds a joint NumPyro model from all SubmodelTarget forward models
3. Applies translation sigma (from source relevance) inside the likelihood
4. Runs MCMC (NUTS) to produce joint posterior samples
5. Fits marginal distributions + Gaussian copula
6. Writes `submodel_priors.yaml` for downstream SBI consumption

## Project Structure

```
src/maple/
├── core/
│   ├── calibration/
│   │   ├── submodel_target.py         # SubmodelTarget schema
│   │   ├── calibration_target_models.py  # CalibrationTarget schema
│   │   ├── submodel_inference.py      # Joint NumPyro MCMC inference
│   │   ├── posterior_parameterizer.py # Marginals + Gaussian copula
│   │   ├── yaml_to_prior.py          # Orchestrator + CLI
│   │   └── ...
│   ├── tools/                         # LLM agent tools
│   └── workflow/                      # Workflow orchestration
├── cli/                               # CLI entry points
└── prompts/                           # LLM instruction prompts
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Developer guide and schema details
- [maple-paper](https://github.com/popellab/maple-paper) - Manuscript, PDAC calibration targets, and reproducibility scripts

## License

MIT
