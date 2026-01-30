# QSP LLM Workflows

[![Tests](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml/badge.svg)](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml)

Extract calibration targets from scientific literature for quantitative systems pharmacology (QSP) model calibration. Uses structured YAML schemas with Pydantic validation, then translates to Julia/Turing.jl for Bayesian inference.

## Installation

```bash
git clone https://github.com/popellab/qsp-llm-workflows.git
cd qsp-llm-workflows
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
python -m qsp_llm_workflows.core.calibration.julia_translator \
    --model-structure model_structure.json \
    target.yaml

# Joint inference (parameters with same name are shared)
python -m qsp_llm_workflows.core.calibration.julia_translator --joint \
    --model-structure model_structure.json \
    target1.yaml target2.yaml target3.yaml \
    --output joint_calibration.jl

# Use --fixed-sigma to treat all sigmas as fixed (faster sampling)
python -m qsp_llm_workflows.core.calibration.julia_translator --joint \
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

## Project Structure

```
src/qsp_llm_workflows/
├── core/
│   └── calibration/
│       ├── submodel_target.py      # SubmodelTarget schema (primary)
│       ├── julia_translator.py     # YAML → Julia/Turing.jl
│       └── ...
├── cli/                            # Command-line tools
└── prompts/                        # LLM instruction prompts
```

## Older Workflows

The following schemas are from earlier development and may be deprecated:

- **IsolatedSystemTarget**: Earlier schema for in vitro/preclinical data with Python submodel code
- **CalibrationTarget**: Base class for clinical/in vivo data requiring full model simulation
- **Parameter extraction**: Direct parameter value extraction (legacy)
- **Test statistics**: Validation constraints from experimental data (legacy)

See [docs/](docs/) for documentation on these older workflows.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Developer guide and schema details

## License

MIT
