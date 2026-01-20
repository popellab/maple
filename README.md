# QSP LLM Workflows

[![Tests](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml/badge.svg)](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml)

Extract calibration targets from scientific literature for quantitative systems pharmacology (QSP) model calibration. This package uses OpenAI's API to read papers, extract experimental data with uncertainty estimates, and generate Python code for Bayesian inference.

## Installation

```bash
git clone https://github.com/popellab/qsp-llm-workflows.git
cd qsp-llm-workflows
python -m venv venv
source venv/bin/activate
pip install -e .
```

Store your OpenAI API key:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Quick Start

### 1. Export model structure

```bash
qsp-export-model \
  --matlab-model ../your-model/model.m \
  --output jobs/input_data/model_definitions.json \
  --export-structure jobs/input_data/model_structure.json
```

### 2. Create input CSV

```csv
target_id,cancer_type,parameters,notes
cd8_proliferation,PDAC,"k_CD8_pro","CD8 T cell proliferation from in vitro assays"
spheroid_growth,PDAC,"k_C1_growth,C_max","Cancer growth from spheroid experiments"
```

### 3. Run extraction

```bash
qsp-extract \
  targets.csv \
  --type isolated_system_target \
  --output-dir metadata-storage \
  --model-structure jobs/input_data/model_structure.json
```

### 4. Review outputs

Results are unpacked to `metadata-storage/to-review/isolated_system_targets/` as YAML files containing:
- Study interpretation and key assumptions
- ODE submodel code sharing parameter names with your full model
- Empirical data with uncertainty (median, 95% CI, sample size)
- Python code deriving distributions from literature values
- Source tracking with DOIs and text snippets

### 5. Generate Julia inference code (optional)

Translate YAML targets to Julia/Turing.jl for Bayesian inference:

```bash
# Single target
python -m qsp_llm_workflows.core.calibration.julia_translator target.yaml

# Joint inference across multiple targets (parameters with same name are shared)
python -m qsp_llm_workflows.core.calibration.julia_translator --joint \
    psc_proliferation.yaml psc_death.yaml psc_recruitment.yaml \
    --output joint_calibration.jl
```

## What gets extracted

**IsolatedSystemTarget** (primary workflow): For in vitro, ex vivo, or preclinical data. Generates a simplified ODE submodel that captures experimental dynamics while sharing parameter names with your full QSP model. This enables joint Bayesian inference across multiple calibration targets.

Example output structure:
```yaml
study_interpretation: |
  CD8+ T cell proliferation measured via CFSE dilution in 7-day culture...

parameters: [k_CD8_pro]

submodel:
  code: |
    def submodel(t, y, params, inputs):
        N = y[0]
        k_pro = params['k_CD8_pro']
        return [k_pro * N]
  t_span: [0, 7]
  t_unit: day

empirical_data:
  median: [2.5]
  ci95: [[1.8, 3.2]]
  units: 1/day
  sample_size: 5
  distribution_code: |
    def derive_distribution(inputs, ureg):
        # Converts literature values to parameter distribution
        ...
```

**CalibrationTarget**: For clinical/in vivo data where the full model is needed. Uses an `observable` function to compute measurements from model species.

## Project structure

```
src/qsp_llm_workflows/
├── core/
│   └── calibration/              # Calibration target models
│       ├── calibration_target_models.py  # CalibrationTarget base
│       ├── isolated_system_target.py     # IsolatedSystemTarget
│       ├── submodel_target.py            # SubmodelTarget
│       ├── julia_translator.py           # YAML → Julia/Turing.jl
│       ├── observable.py                 # Submodel, Observable
│       ├── shared_models.py              # EstimateInput, Source
│       └── code_validator.py             # Code validation
├── cli/                          # qsp-extract, qsp-validate, etc.
├── prompts/                      # LLM instruction prompts
└── templates/                    # YAML output templates
```

## Key concepts

**Submodel**: A standalone ODE system that approximates full model dynamics for the isolated experimental system. Uses the same parameter names as your full model.

**Direct conversion mode**: For simple analytical relationships (k = ln(2) / t_half), omit the submodel and let `distribution_code` compute the parameter directly.

**Input types**: Literature values can be classified as `direct_parameter` (literal values), `proxy_measurement` (requires conversion), `experimental_condition` (protocol choices), or `inferred_estimate` (interpreted from qualitative text).

**Vector-valued data**: Time-course and dose-response data supported via `index_values`, `index_unit`, and `index_type` fields.

## Documentation

- [Calibration workflow guide](docs/calibration_workflow.md) - Complete guide to IsolatedSystemTarget extraction
- [Legacy workflows](docs/automated_workflow.md) - Parameter estimates and test statistics (legacy)
- [CLAUDE.md](CLAUDE.md) - Package internals for developers

## License

MIT
