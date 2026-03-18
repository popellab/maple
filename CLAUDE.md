# CLAUDE.md

Developer guide for Claude Code when working with this repository.

---

## Overview

MAPLE (Model-Aware Parameterization from Literature Evidence) provides tools for extracting QSP calibration targets from scientific literature and producing informative priors via joint Bayesian inference.

**Two extraction workflows:**
- **SubmodelTarget**: In vitro / preclinical data with self-contained forward models (ODE or algebraic) → joint MCMC inference (NumPyro) → marginals + Gaussian copula priors
- **CalibrationTarget**: Clinical / in vivo observables requiring full model simulation context, with Monte Carlo distribution derivation

**Two-stage calibration:**
- **Stage 1** (this repo): SubmodelTargets → joint MCMC (NumPyro/NUTS) → `submodel_priors.yaml` (marginals + copula)
- **Stage 2** (qsp-sbi): CalibrationTargets + copula priors → SBI (SNPE-C) with SimBiology → final posterior

## Installation

```bash
git clone https://github.com/popellab/maple.git
cd maple
pip install -e .

# For joint inference pipeline (NumPyro/JAX)
pip install -e ".[inference]"
```

## SubmodelTarget Schema

For in vitro and preclinical data with self-contained forward models. Located in `submodel_target.py`.

### Key Design

**Separation of concerns:**
- `inputs`: Raw values extracted from papers with full provenance
- `calibration`: Everything needed for inference (parameters, model, measurements)

**Input roles** (`InputRole` enum):
- `initial_condition`: Used as IC for ODE integration
- `target`: Used as calibration target (likelihood term)
- `fixed_parameter`: Fixed value in model (not estimated)
- `auxiliary`: Supporting data (e.g., SD values)

**Parameters** — names and units only. Priors come from `pdac_priors.csv`, not the YAML:
```yaml
parameters:
  - name: k_apsc_prolif
    units: 1/day
```

**Nuisance parameters** — needed by the forward model but not in the QSP model. Carry their own inline prior, sampled during MCMC but excluded from output:
```yaml
parameters:
  - name: k_activation
    units: 1/day
  - name: k_prolif
    units: 1/day
    nuisance: true
    prior:
      distribution: lognormal
      mu: -2.3
      sigma: 0.8
```
Validators enforce: `nuisance: true` requires `prior`; non-nuisance forbids `prior`.

**Forward model types** (discriminated union in `calibration.forward_model`):
- `exponential_growth`: dy/dt = k * y
- `first_order_decay`: dy/dt = -k * y
- `two_state`: A → B transition
- `saturation`: dy/dt = k * (1 - y)
- `logistic`: dy/dt = k * y * (1 - y/K)
- `michaelis_menten`: dy/dt = -Vmax * y / (Km + y)
- `direct_fit`: Dose-response curves (hill, linear, exponential) with auto-generated code
- `power_law`: Biophysical scaling: y = coefficient * (x / reference_x) ^ exponent
- `algebraic`: No ODE, forward model maps params → observable (e.g., `t_half = ln(2) / k`)
- `custom_ode`: User-provided ODE code

**Algebraic models** (params → observable):
```yaml
forward_model:
  type: algebraic
  formula: "t_half = ln(2) / k"
  code: |
    def compute(params, inputs):
        import numpy as np
        return np.log(2) / params['k']
```

Code must use only `np.*` functions (mapped to `jax.numpy` at inference time). No `ureg`, no `scipy`, no branching on parameter values.

**Custom ODE models:**
```yaml
forward_model:
  type: custom_ode
  code: |
    def ode(t, y, params, inputs):
        ...
```

**Error model** (`calibration.error_model`):
```yaml
error_model:
  - name: t_half_measurement
    units: day
    uses_inputs: [t_half_mean, t_half_sd]
    sample_size_input: n_samples
    observation_code: |
      def derive_observation(inputs, sample_size, rng, n_bootstrap):
          import numpy as np
          mean = inputs['t_half_mean']
          sd = inputs['t_half_sd']
          return rng.lognormal(np.log(mean), sd / np.sqrt(sample_size), n_bootstrap)
```

The likelihood family is inferred automatically from bootstrap samples via `fit_distributions()` — no manual `likelihood` field.

**`x_input` for dose-response / power-law models:** Each error model entry can specify `x_input` pointing to an input that provides the independent variable value (e.g., dose, concentration). This enables multi-point evaluation:
```yaml
error_model:
  - name: response_low_dose
    x_input: dose_low      # evaluate forward model at this x value
    uses_inputs: [obs_low_mean, obs_low_sd]
    ...
  - name: response_high_dose
    x_input: dose_high     # evaluate at a different x value
    uses_inputs: [obs_high_mean, obs_high_sd]
    ...
```

### Validation

```bash
# Validate a single file
python scripts/validate_submodel_target.py path/to/target.yaml

# Validate multiple files
python scripts/validate_submodel_target.py *.yaml
```

**Built-in validators** (run automatically on model validation):

| Validator | What it checks |
|-----------|----------------|
| `validate_input_refs` | All `uses_inputs`, `input_ref` references point to existing inputs |
| `validate_source_refs` | All `source_ref` match a `source_tag` in data sources |
| `validate_parameter_roles` | Model parameter strings match `calibration.parameters` |
| `validate_ode_model_requirements` | ODE models have `state_variables` and `independent_variable.span` |
| `validate_custom_code_syntax` | Python code has correct function signatures |
| `validate_observation_bootstrap_samples` | `observation_code` returns valid bootstrap array |
| `validate_observation_code_execution` | `observation_code` executes successfully |
| `validate_no_invisible_characters` | No invisible/control chars (zero-width spaces, soft hyphens, etc.) |
| `validate_span_ordering` | `span[0] < span[1]` and both non-negative |
| `validate_evaluation_points_within_span` | Evaluation points within independent_variable.span |
| `validate_input_values_in_snippets` | Extracted values appear in `value_snippet` (anti-hallucination) |
| `validate_snippets_in_source` | Verifies snippets appear in actual paper text (Europe PMC, Unpaywall) |
| `validate_all_parameters_used_in_forward_model` | All parameters in calibration.parameters are accessed in forward model code |
| `validate_doi_resolution_and_metadata` | DOIs resolve via CrossRef, metadata matches |
| `validate_units_are_valid_pint` | All unit strings are valid Pint units |
| `validate_no_hardcoded_values_in_observation_code` | All numeric values come through inputs |
| `validate_ode_requires_observable` | ODE models have observable in error_model |
| `validate_source_quality_peer_reviewed` | Warns if `source_quality` is `non_peer_reviewed` |
| `validate_secondary_sources_quality` | Warns if secondary sources have `non_peer_reviewed` quality |
| `validate_clipping_suggests_lognormal` | Warns if observation_code clips to avoid negatives |
| `validate_large_variance_documented` | Warns if high variance not discussed in identifiability_notes |
| `warn_multi_param_algebraic_identifiability` | Warns when AlgebraicModel has more params than measurements |
| `warn_observation_cv_unreasonable` | Warns if CV is unreasonably large or small |

**observation_code signature** (required format):
```python
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    import numpy as np
    # inputs: dict of {name: float} (magnitudes only, no Pint)
    # rng: numpy.random.Generator (seeded)
    # n_bootstrap: number of samples to generate
    mean = inputs['obs_mean']
    sd = inputs['obs_sd']
    return rng.lognormal(np.log(mean), sd / np.sqrt(sample_size), n_bootstrap)
    # Must return 1D numpy array of parametric bootstrap samples
```

**No invisible characters** - catches PDF copy-paste issues:
- Zero-width spaces, soft hyphens, byte order marks
- Control characters (except tab, newline, carriage return)
- Unicode letters (Greek, accents) and math symbols (±, ≥) are allowed

**Snippet-in-source verification** - catches hallucinated quotes:
- Fetches paper text from Europe PMC (abstracts + PMC full text)
- Falls back to Unpaywall for open access PDFs/HTML
- Uses fuzzy matching to find snippets in source text
- Set `source_access: restricted` on inputs to skip verification for paywalled papers

```yaml
inputs:
  - name: some_value
    value: 10.0
    source_access: restricted  # Skip auto-verification for this input
```

### Source Relevance Assessment

The `source_relevance` field captures how well the source data translates to the target model context. Required enums are in `enums.py`.

```yaml
source_relevance:
  indication_match: proxy  # exact, related, proxy, unrelated
  indication_match_justification: "Prostate stromal cells used for PDAC CCL2..."
  species_source: mouse
  species_target: human
  source_quality: primary_animal_in_vitro  # primary_human_clinical, ..., non_peer_reviewed
  perturbation_type: pharmacological  # physiological_baseline, pathological_state, etc.
  tme_compatibility: low  # high, moderate, low
  tme_compatibility_notes: "EG7 thymoma is T cell-permissive; PDAC is excluded..."
  measurement_directness: direct
  temporal_resolution: endpoint_pair
  experimental_system: in_vitro_primary
```

**Key fields:**

| Field | Purpose | When to flag |
|-------|---------|--------------|
| `indication_match` | Does disease match? | `proxy`/`unrelated` require justification |
| `source_quality` | Peer review status | `non_peer_reviewed` (Wikipedia, preprints) triggers warning |
| `tme_compatibility` | TME similarity | `low` for PDAC if source is T cell-permissive tumor |
| `measurement_directness` | How directly was the parameter measured? | `proxy_observable` adds most uncertainty |

Translation sigma is computed from all 8 axes (added in quadrature, floor of 0.15) and applied inside the likelihood during joint inference. See `yaml_to_prior.py:compute_translation_sigma()` for the full rubric.

### Interactive Extraction (MCP Server)

The preferred way to create SubmodelTarget YAMLs is interactively via the MCP server with Claude Code. This produces better results than batch extraction because the forward model and source relevance assessment benefit from iterative discussion.

**Setup:** Add to `.claude/settings.json`:
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

**Two tools are available:**

- `extract_target(target_type)` — Loads the full extraction guide: multi-step workflow, prompt template, valid enum values, and hard rules. Call this before starting any extraction session.
- `validate_target(yaml_path, priors_csv, papers_dir)` — Runs schema validation (Pydantic), prior derivation via NumPyro MCMC (bootstrap + forward model + distribution fitting + translation sigma), and snippet verification against source PDFs. `priors_csv` is required (e.g., `parameters/pdac_priors.csv`).

**Typical workflow:**

1. Call `extract_target` to load the guide
2. Investigate the parameter in the model code (units, mechanistic role, Hill function inputs)
3. Search literature for quantitative data that constrains the parameter
4. User obtains PDFs into `papers/<source_tag>/` directories
5. Read the paper and build the YAML incrementally — inputs first, then forward model, then error model, then source relevance
6. Call `validate_target` after each major section to catch errors early
7. Iterate until validation passes

### Batch Extraction (CLI)

For bulk extraction without interactive refinement:

```bash
# Extract submodel targets (requires model context files)
qsp-extract targets.csv \
  --type submodel_target \
  --model-structure model_structure.json \
  --model-context model_context.txt \
  --output-dir metadata-storage

# Preview prompts without API call
qsp-extract targets.csv \
  --type submodel_target \
  --model-structure model_structure.json \
  --model-context model_context.txt \
  --output-dir metadata-storage \
  --preview-prompts
```

**Input CSV format:**
```csv
target_id,parameters,notes
psc_proliferation,k_apsc_prolif,Focus on activated PSCs
psc_death,k_apsc_death,
```

## CalibrationTarget Schema

For clinical/in vivo observables (biopsies, blood draws, resections). Located in `calibration_target_models.py`.

### Key Design

**Experimental context vs model context:** Each observable has an experimental context (species, indication, compartment, treatment) that may differ from the QSP model context. Mismatches are documented and quantified via `source_relevance`.

**Core structure:**
- `observable` — What is being measured (species name from model, units, compartment, support type, aggregation)
- `scenarios` — Experimental conditions with interventions
- `empirical_data` — Literature values + Monte Carlo `distribution_code` that derives median/CI95
- `experimental_context` — Species, indication, stage, treatment history
- `source_relevance` — Indication match, species translation, TME compatibility

**Strict matching requirements** (enforced in extraction prompts):
- Species must match exactly (no cross-species substitution)
- Indication must match exactly (no related cancer substitution)
- Compartment must match exactly (no serum-for-tissue substitution)
- Source must be in vivo patient data (no cell culture/organoids)

**distribution_code** derives calibration statistics from extracted inputs:
```python
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

**Vector-valued targets** supported via `index_values`, `index_unit`, `index_type` for time-course or dose-response data.

### Validation

```bash
# Validate (requires species_units.json for unit checking)
python scripts/validate_calibration_target.py \
  --species-units path/to/species_units.json target.yaml

# Validate directory of targets
python scripts/validate_calibration_target.py \
  --species-units path/to/species_units.json calibration_targets/

# Skip DOI checks (faster, offline)
python scripts/validate_calibration_target.py --skip-doi \
  --species-units path/to/species_units.json target.yaml
```

**CalibrationTarget validators** (run on `model_validate`):
- DOI resolution and title matching (CrossRef API)
- `distribution_code` execution — computed median/CI95 must match reported values within 1%
- `measurement_code` unit dimensionality check (Pint)
- `value_snippet` contains declared numeric values (anti-hallucination)
- Source reference consistency (`source_ref` → `source_tag`)
- No invisible/control characters
- Species existence in model (via `species_units` context)

### LLM Extraction

```bash
qsp-extract targets.csv \
  --type calibration_target \
  --output-dir metadata-storage

# Preview prompts without API call
qsp-extract targets.csv \
  --type calibration_target \
  --output-dir metadata-storage \
  --preview-prompts
```

**Input CSV format:**
```csv
observable_name,description,species,compartment,notes
cd8_tumor_density,CD8+ T cell density in primary tumor,CD8_T,tumor.primary,Treatment-naive PDAC
```

## Submodel Prior Inference

Joint Bayesian inference across SubmodelTargets using NumPyro. Located in `submodel_inference.py` and `posterior_parameterizer.py`.

### Pipeline

```
pdac_priors.csv (broad starting priors)
    + SubmodelTarget YAMLs (data + forward models)
    → build joint NumPyro model:
        - priors from CSV
        - forward models: structured algebraic, exec(algebraic code), analytical ODE, or diffrax ODE
        - likelihoods with translation sigma in observation noise
        - NaN guard: solver failures → -inf log-prob → NUTS rejects sample
    → MCMC (NUTS) → joint posterior samples
    → fit marginal distributions per parameter
    → fit Gaussian copula (correlation matrix)
    → output: submodel_priors.yaml (marginals + copula for Stage 2)
```

### Usage (Python)

```python
from maple.core.calibration.yaml_to_prior import process_targets

result = process_targets(
    priors_csv=Path("pdac_priors.csv"),
    yaml_paths=[Path("target1.yaml"), Path("target2.yaml")],
    output_dir=Path("priors/"),
    plot=True,
)
```

### CLI Usage

```bash
# Joint inference across all submodel targets
maple-yaml-to-prior --priors pdac_priors.csv submodel_targets/ --output priors/ --plot

# With reference database and CSV export
maple-yaml-to-prior --priors pdac_priors.csv submodel_targets/ \
    --output priors/ --export-csv updated_priors.csv --reference-db reference_values.yaml
```

### Output Format

```yaml
# submodel_priors.yaml
metadata:
  n_targets: 10
  n_parameters: 13
  n_samples: 20000

parameters:
  - name: k_apsc_prolif
    marginal:
      distribution: lognormal
      mu: -0.31
      sigma: 0.85
      median: 0.733
      cv: 0.95
    source_targets: [psc_proliferation_PDAC_deriv001]

copula:
  type: gaussian
  parameters: [k_apsc_prolif, k_apsc_death]
  correlation:
    - [1.0, -0.42]
    - [-0.42, 1.0]
```

### Key Design

- **Translation sigma in likelihood**: Per-target source relevance maps to a translation σ applied inside the likelihood during MCMC (not post-hoc). MCMC naturally upweights more relevant sources.
- **Shared parameters**: Parameters with the same name across targets are sampled once and reused.
- **Likelihood family inferred**: Bootstrap samples are fit with lognormal/gamma/inv-gamma; best by AIC determines the likelihood type.
- **JAX-traceable**: All forward models must use `np.*` functions (mapped to `jax.numpy`). No `scipy`, no branching on parameter values.
- **ODE support**: Analytical closed-form solutions for `exponential_growth`, `first_order_decay`, `saturation`, `two_state`, `logistic`. Numerical integration via diffrax (`Tsit5`) for `michaelis_menten` and `custom_ode`. Custom observables (e.g., `A/(Q+A)`) are exec'd with `jnp`.
- **NaN guard**: Forward functions that return NaN (e.g., diffrax solver failure from extreme MCMC proposals) trigger a `-inf` log-probability via `numpyro.factor`, causing NUTS to reject the proposal rather than crashing.

## Package Structure

```
src/maple/
├── core/
│   ├── calibration/
│   │   ├── submodel_target.py         # SubmodelTarget schema
│   │   ├── calibration_target_models.py  # CalibrationTarget schema
│   │   ├── submodel_inference.py      # Joint NumPyro MCMC inference
│   │   ├── posterior_parameterizer.py # Marginal fitting + Gaussian copula
│   │   ├── yaml_to_prior.py          # Orchestrator + CLI
│   │   ├── julia_translator.py        # YAML → Julia/Turing.jl (legacy)
│   │   ├── code_validator.py          # Python code validation
│   │   ├── enums.py                   # Shared enums
│   │   ├── exceptions.py              # Validation error hierarchy
│   │   ├── experimental_context.py    # ExperimentalContext model
│   │   ├── observable.py              # Observable models
│   │   ├── scenario.py                # Scenario models
│   │   ├── shared_models.py           # Shared Pydantic models
│   │   ├── submodel_utils.py          # Submodel utilities
│   │   └── validators.py              # Validation functions
│   ├── tools/
│   │   └── view_figure.py             # Figure extraction tool
│   ├── workflow/
│   │   ├── context.py                 # Workflow context
│   │   ├── step.py                    # Workflow step ABC
│   │   └── steps.py                   # Concrete workflow steps
│   ├── config.py                      # WorkflowConfig
│   ├── exceptions.py                  # Workflow exceptions
│   ├── immediate_processor.py         # Pydantic AI request processor
│   ├── model_structure.py             # ModelStructure for validation
│   ├── model_structure_exporter.py    # SimBiology → JSON export
│   ├── output_directory.py            # Output directory management
│   ├── prompt_builder.py              # Prompt builder classes
│   ├── prompts.py                     # Prompt assembly functions
│   ├── resource_utils.py              # Package resource access
│   ├── unit_registry.py               # Shared Pint UnitRegistry
│   └── workflow_orchestrator.py       # Main orchestrator
├── cli/
│   ├── extract.py                     # qsp-extract entry point
│   ├── export_model.py                # qsp-export-model entry point
│   └── interactive.py                 # Interactive target selection
└── prompts/                           # LLM instruction prompts
```

## Shared Pint UnitRegistry

All code that handles units must use the shared Pint UnitRegistry:

```python
from maple.core.unit_registry import ureg

time = np.linspace(0, 14, 100) * ureg.day
concentration = 5.0 * ureg.nanomolarity
```

**Custom units defined:**
- `cell = [cell_count]` - for cell counts and densities
- `nanomolarity = nanomolar` - SimBiology convention alias

**IMPORTANT:** Never create a new `pint.UnitRegistry()`. Always import `ureg` from the shared module.

## Development

### Running Tests

```bash
pytest
```

### CLI Entry Points

- `qsp-extract` — Run extraction workflows (calibration_target or submodel_target)
- `qsp-export-model` — Export SimBiology model structure to JSON
- `maple-yaml-to-prior` — Convert SubmodelTarget YAMLs to priors via joint NumPyro MCMC (requires `--priors CSV`)
