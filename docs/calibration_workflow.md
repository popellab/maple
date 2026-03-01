# Calibration Target Workflow

This guide covers extracting calibration targets from scientific literature for QSP model calibration. Calibration targets are raw observables with uncertainty estimates that constrain model parameters via Bayesian inference.

## Overview

There are two types of calibration targets:

- **IsolatedSystemTarget**: For in vitro, ex vivo, or preclinical data where a simplified ODE submodel captures the experimental dynamics. The submodel shares parameter names with the full QSP model, enabling joint inference. This is the primary workflow.

- **CalibrationTarget**: For clinical/in vivo data where the full model is needed. Uses an `observable` to compute measurements from model species.

Both types include:
- Empirical data with uncertainty (median, 95% CI, sample size)
- Python code that derives distributions from literature values
- Source tracking with DOIs and text snippets
- Experimental context (species, indication, system type)

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key (see [JHU API Key Guide](https://support.cmts.jhu.edu/hc/en-us/articles/38383798293133-Guide-to-Managing-API-Keys-and-Usage-Limits-on-platform-openai-com))
- Model structure file exported from your QSP model

### Installation

```bash
cd ~/Projects
git clone git@github.com:popellab/maple.git
cd qsp-llm-workflows

python3 -m venv venv
source venv/bin/activate
pip install -e .

echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Export Model Structure

The extraction needs parameter context from your model. Export using `--export-structure`:

```bash
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/model.m \
  --output jobs/input_data/model_definitions.json \
  --export-structure jobs/input_data/model_structure.json
```

This creates:
- `model_definitions.json`: Flat parameter definitions for prompts
- `model_structure.json`: Structured model data for validation (parameters, species, reactions)
- `species_units.json`: Unit information for each species

Optionally create a `model_context.txt` file with a high-level description of your model (1-2 paragraphs describing the disease area, key mechanisms, and compartments).

## IsolatedSystemTarget Workflow

### Step 1: Prepare Input CSV

Create a CSV with targets to extract:

```csv
target_id,cancer_type,parameters,notes
cd8_proliferation,PDAC,"k_CD8_pro","CD8 T cell proliferation rate from in vitro expansion assays"
spheroid_growth,PDAC,"k_C1_growth,C_max","Cancer cell growth from spheroid assays"
```

Fields:
- `target_id`: Unique identifier for this calibration target
- `cancer_type`: Disease context (e.g., PDAC, NSCLC)
- `parameters`: Comma-separated parameter names from your model that this target constrains
- `notes`: Context to guide literature search

### Step 2: Run Extraction

```bash
qsp-extract \
  targets.csv \
  --type isolated_system_target \
  --output-dir ../your-model-repo/metadata-storage \
  --model-structure jobs/input_data/model_structure.json \
  --model-context jobs/input_data/model_context.txt
```

Options:
- `--preview-prompts`: Preview the generated prompt without calling the API
- `--reasoning-effort low|medium|high`: Control reasoning depth (default: low)

Results are unpacked to `metadata-storage/to-review/isolated_system_targets/`.

### Step 3: Review Output

Each extraction produces a YAML file with this structure:

```yaml
study_interpretation: |
  This study measured CD8+ T cell proliferation in vitro using CFSE dilution...

key_assumptions:
  - In vitro expansion rates assumed transferable to tumor microenvironment
  - Exponential growth phase used for rate calculation

key_study_limitations:
  - Single donor source limits generalizability
  - Culture conditions differ from in vivo

submodel:
  code: |
    def submodel(t, y, params, inputs):
        N = y[0]
        k_pro = params['k_CD8_pro']
        return [k_pro * N]
  state_variables:
    - name: cd8_cells
      units: cell
      initial_value: 10000.0
      source_ref: Smith2023
      value_location: "Methods, p.3"
      value_snippet: "10^4 cells seeded per well"
  parameters:
    - k_CD8_pro
  t_span: [0, 7]
  t_unit: day
  observable:
    units: cell
  rationale: |
    Simple exponential growth captures the expansion phase...

experimental_context:
  species: human
  indication: pdac
  compartment: tumor_draining_lymph_node
  system: in_vitro_primary_cells

empirical_data:
  median: [2.5]
  ci95: [[1.8, 3.2]]
  units: 1/day
  sample_size: 5
  sample_size_rationale: "n=5 donors stated in Methods"
  inputs:
    - name: doubling_time_mean
      value: 6.7
      units: hour
      source_ref: Smith2023
      value_location: "Results, Fig 2A"
      value_snippet: "Mean doubling time was 6.7 ± 1.2 hours"
  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        rng = np.random.default_rng(42)
        t_double = inputs['doubling_time_mean']
        # Convert doubling time to proliferation rate: k = ln(2) / t_double
        k_samples = np.log(2) / rng.normal(t_double.magnitude, 1.2, 10000) * (1 / t_double.units)
        k_samples = k_samples.to('1/day')
        return {
            'median_obs': np.median(k_samples),
            'ci95_lower': np.percentile(k_samples, 2.5),
            'ci95_upper': np.percentile(k_samples, 97.5),
        }

primary_data_source:
  source_tag: Smith2023
  title: "CD8+ T cell expansion kinetics in pancreatic cancer"
  doi: "10.1234/example.2023.001"
  year: 2023

tags:
  - ai-generated
```

### Step 4: Validate

Pydantic validators run automatically during extraction and catch most issues. For additional manual review:

1. **Check study_interpretation**: Does it accurately describe what was measured?
2. **Check submodel code**: Does the ODE capture the right dynamics?
3. **Check distribution_code**: Does it correctly convert literature values to parameter estimates?
4. **Check snippets**: Do value_snippets contain the claimed values?
5. **Check DOIs**: Do they resolve to the correct papers?

## Direct Conversion Mode

For simple analytical relationships (e.g., k = ln(2) / t_half), you can omit the submodel entirely:

```yaml
# No submodel needed - distribution_code computes parameter directly
submodel: null

empirical_data:
  median: [0.693]
  ci95: [[0.58, 0.83]]
  units: 1/day
  inputs:
    - name: half_life
      value: 24.0
      units: hour
      ...
  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        rng = np.random.default_rng(42)
        t_half = inputs['half_life']
        k = np.log(2) / rng.normal(t_half.magnitude, 2.0, 10000) * (1 / t_half.units)
        return {
            'median_obs': np.median(k.to('1/day')),
            'ci95_lower': np.percentile(k.to('1/day'), 2.5),
            'ci95_upper': np.percentile(k.to('1/day'), 97.5),
        }
```

Use direct conversion when:
- Simple algebraic formula relates literature value to parameter
- Examples: doubling time, half-life, Kd from binding assays

Use submodel when:
- Multiple interacting parameters need joint estimation
- Nonlinear dynamics (logistic growth, saturation kinetics)
- Time-course data requiring ODE fitting

## Input Types

Inputs can be classified by `input_type`:

- `direct_parameter`: Value reported literally in paper ("mean = 42.0")
- `proxy_measurement`: Requires conversion ("doubling time = 8h" → rate constant)
- `experimental_condition`: Protocol choice from paper (seeding density, E:T ratio)
- `inferred_estimate`: Value interpreted from qualitative text ("maintained viability" → 0.95)

Use `inferred_estimate` when the numeric value doesn't appear literally but is a reasonable interpretation. Snippet validation is skipped for this type.

## Figure-Extracted Data

For values read from figures:

```yaml
inputs:
  - name: tumor_volume_day14
    value: 150.0
    units: mm**3
    source_type: figure
    figure_id: "Figure 2A"
    extraction_method: manual
    extraction_notes: "Read from y-axis at day 14 timepoint"
    value_snippet: "Approximate value read from growth curve"
```

## Vector-Valued Data

For time-course or dose-response data:

```yaml
empirical_data:
  median: [10.0, 25.0, 80.0, 150.0]
  ci95: [[8.0, 12.0], [20.0, 30.0], [65.0, 95.0], [120.0, 180.0]]
  units: mm**3
  sample_size: [5, 5, 4, 4]  # Can vary per timepoint
  index_values: [0, 7, 14, 21]
  index_unit: day
  index_type: time
```

## Troubleshooting

**"OPENAI_API_KEY not found"**: Create `.env` file with your key.

**Validation errors during extraction**: Check the error message for guidance. Common issues:
- DOI doesn't resolve: Verify DOI at https://doi.org/
- Value not in snippet: Check for formatting differences or use `input_type: inferred_estimate`
- Unit mismatch: Ensure distribution_code returns Pint Quantities with correct units
- Hardcoded constants: Move numeric values with units to `inputs` or `assumptions`

**Submodel integration fails**: Check for:
- Division by zero
- Numerical instability (very large rates)
- Wrong return shape (must return list matching state_variables length)

## File Organization

```
your-model-repo/
└── metadata-storage/
    ├── to-review/
    │   └── isolated_system_targets/  # New extractions
    └── calibration_targets/          # Approved targets

qsp-llm-workflows/
├── jobs/
│   └── input_data/
│       ├── model_structure.json
│       ├── model_context.txt
│       └── targets.csv
└── .env
```
