# MAPLE

[![Tests](https://github.com/popellab/maple/actions/workflows/test.yml/badge.svg)](https://github.com/popellab/maple/actions/workflows/test.yml)

Extract calibration targets from scientific literature for quantitative systems pharmacology (QSP) model calibration. Uses structured YAML schemas with Pydantic validation, then translates to Julia/Turing.jl for Bayesian inference.

## Installation

```bash
git clone https://github.com/popellab/maple.git
cd maple
python -m venv venv
source venv/bin/activate
pip install -e .
```

## SubmodelTarget Schema

The **SubmodelTarget** schema is the primary format for calibration targets. It separates:
- **inputs**: Values extracted from literature with full provenance
- **calibration**: How those values constrain model parameters (priors, ODE model, measurements)

Example structure:
```yaml
target_id: psc_proliferation_PDAC_deriv001

inputs:
  - name: fold_increase_mean
    value: 4.37
    units: dimensionless
    role: target
    input_type: direct_measurement
    source_ref: schneider_2001
    value_snippet: "PDGF increased DNA synthesis 4.37 ± 0.89-fold"

calibration:
  parameters:
    - name: k_apsc_prolif
      units: 1/day
      prior:
        distribution: lognormal
        mu: 0.0
        sigma: 1.0
  forward_model:
    type: exponential_growth
    rate_constant: k_apsc_prolif
    state_variables:
      - name: N
        units: dimensionless
        initial_condition: {value: 1.0, rationale: "Normalized"}
    independent_variable:
      name: time
      units: day
      span: [0, 3]
  error_model:
    - name: fold_increase
      units: dimensionless
      uses_inputs: [fold_increase_mean, fold_increase_sd]
      evaluation_points: [3.0]
      observation_code: |
        def derive_observation(inputs, sample_size, ureg):
            return {
                'value': inputs['fold_increase_mean'],
                'sd': inputs['fold_increase_sd'].magnitude,
            }
      likelihood: {distribution: lognormal}
```

### Validation

```bash
python scripts/validate_submodel_target.py path/to/target.yaml
```

The schema validates:
- All references (`input_ref`, `uses_inputs`, `source_ref`) point to existing items
- Extracted values appear in `value_snippet` (anti-hallucination check)
- Snippets appear in actual paper text (via Europe PMC/Unpaywall)
- DOIs resolve and metadata matches
- Prior predictions are on the same scale as observations

### LLM Extraction

Extract SubmodelTarget YAMLs from scientific literature:

```bash
qsp-extract targets.csv \
  --type submodel_target \
  --model-structure model_structure.json \
  --model-context model_context.txt \
  --output-dir metadata-storage
```

## Julia Code Generation

Translate validated YAML targets to Julia/Turing.jl for Bayesian inference:

```bash
# Single target (--model-structure required)
python -m maple.core.calibration.julia_translator \
    --model-structure model_structure.json \
    target.yaml

# Joint inference (parameters with same name are shared)
python -m maple.core.calibration.julia_translator --joint \
    --model-structure model_structure.json \
    target1.yaml target2.yaml target3.yaml \
    --output joint_calibration.jl

# Use --fixed-sigma to treat all sigmas as fixed (faster sampling)
python -m maple.core.calibration.julia_translator --joint \
    --model-structure model_structure.json \
    --fixed-sigma \
    target1.yaml target2.yaml target3.yaml
```

The translator generates complete Julia scripts with:
- ODE functions (or algebraic compute functions)
- Turing `@model` blocks with priors and likelihoods (normal or lognormal)
- Optional priors on sigma when `sd_uncertain: true`
- NUTS sampling code with convergence diagnostics
- Posterior marginal plots with prior overlays

## CalibrationTarget Schema

The **CalibrationTarget** schema handles clinical/in vivo observables that require full model simulation context:

```bash
# Validate calibration target YAMLs
python scripts/validate_calibration_target.py \
  --species-units path/to/species_units.json \
  target.yaml

# Extract calibration targets from literature
qsp-extract targets.csv \
  --type calibration_target \
  --output-dir metadata-storage
```

## Project Structure

```
src/maple/
├── core/
│   ├── calibration/
│   │   ├── submodel_target.py         # SubmodelTarget schema
│   │   ├── calibration_target_models.py  # CalibrationTarget schema
│   │   ├── julia_translator.py        # YAML → Julia/Turing.jl
│   │   └── ...
│   ├── tools/                         # LLM agent tools
│   └── workflow/                      # Workflow orchestration
├── cli/                               # CLI entry points
└── prompts/                           # LLM instruction prompts
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Developer guide and schema details

## License

MIT
