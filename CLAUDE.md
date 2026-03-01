# CLAUDE.md

Developer guide for Claude Code when working with this repository.

---

## Overview

This repository provides tools for extracting QSP calibration targets from scientific literature and translating them to Julia/Turing.jl for Bayesian inference.

**Primary Workflow:**
- **SubmodelTarget** schema: Structured YAML format separating data extraction from model specification
- **Julia Translator**: Converts validated YAML to executable Turing.jl inference scripts

**Older Workflows (may be deprecated):**
- **IsolatedSystemTarget**: Earlier schema with Python submodel code
- **CalibrationTarget**: Base class for full-model observables
- **Parameter extraction / Test statistics**: Legacy LLM extraction workflows

## Installation

```bash
git clone https://github.com/popellab/maple.git
cd maple
pip install -e .
```

## SubmodelTarget Schema

The primary schema for calibration targets. Located in `submodel_target.py`.

### Key Design

**Separation of concerns:**
- `inputs`: Raw values extracted from papers with full provenance
- `calibration`: Everything needed for inference (parameters, model, measurements)

**Input roles** (`InputRole` enum):
- `initial_condition`: Used as IC for ODE integration
- `target`: Used as calibration target (likelihood term)
- `fixed_parameter`: Fixed value in model (not estimated)
- `auxiliary`: Supporting data (e.g., SD values)

**Parameter priors** (`Prior` model):
```yaml
parameters:
  - name: k_apsc_prolif
    units: 1/day
    prior:
      distribution: lognormal
      mu: 0.0  # log(1.0)
      sigma: 1.0
      rationale: "Wide prior centered at 1/day..."
```

Supported distributions: `lognormal`, `normal`, `uniform`, `half_normal`

**Forward model types** (discriminated union in `calibration.forward_model`):
- `exponential_growth`: dy/dt = k * y
- `first_order_decay`: dy/dt = -k * y
- `two_state`: A → B transition
- `saturation`: dy/dt = k * (1 - y)
- `logistic`: dy/dt = k * y * (1 - y/K)
- `michaelis_menten`: dy/dt = -Vmax * y / (Km + y)
- `algebraic`: No ODE, forward model maps params → observable (e.g., `t_half = ln(2) / k`)
- `custom_ode`: User-provided ODE code (requires `code_julia`)

**Algebraic models** (params → observable):
```yaml
forward_model:
  type: algebraic
  formula: "t_half = ln(2) / k"
  code: |
    def compute(params, inputs, ureg):
        import numpy as np
        k = params['k']
        return np.log(2) / k * ureg('day')
  code_julia: |
    function compute(params, inputs)
        return log(2) / params["k"]
    end
```

**Custom ODE models:**
```yaml
forward_model:
  type: custom_ode
  code: |
    def ode(t, y, params, inputs):
        ...
  code_julia: |
    function my_ode!(du, u, p, t)
        ...
    end
```

**Error model** (`calibration.error_model`):
```yaml
error_model:
  - name: t_half_measurement
    units: day
    uses_inputs: [t_half_mean, t_half_sd]
    observation_code: |
      def derive_observation(inputs, sample_size, ureg):
          return {
              'value': inputs['t_half_mean'],
              'sd': inputs['t_half_sd'].magnitude,
              'sd_uncertain': False,
          }
    likelihood:
      distribution: lognormal
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
| `validate_observation_code_return_signature` | `observation_code` returns `{value, sd}` |
| `validate_observation_code_execution` | `observation_code` executes and returns valid dict |
| `validate_observation_sd_units_for_likelihood` | SD units match likelihood type (dimensionless for lognormal) |
| `validate_no_invisible_characters` | No invisible/control chars (zero-width spaces, soft hyphens, etc.) |
| `validate_span_ordering` | `span[0] < span[1]` and both non-negative |
| `validate_evaluation_points_within_span` | Evaluation points within independent_variable.span |
| `validate_input_values_in_snippets` | Extracted values appear in `value_snippet` (anti-hallucination) |
| `validate_snippets_in_source` | Verifies snippets appear in actual paper text (Europe PMC, Unpaywall) |
| `validate_all_parameters_used_in_forward_model` | All parameters in calibration.parameters are accessed in forward model code |
| `validate_doi_resolution_and_metadata` | DOIs resolve via CrossRef, metadata matches |
| `validate_units_are_valid_pint` | All unit strings are valid Pint units |
| `validate_algebraic_model_output_units` | AlgebraicModel.code returns correct units |
| `validate_no_hardcoded_values_in_observation_code` | All numeric values come through inputs |
| `validate_prior_predictive_scale` | Prior prediction matches observation scale (catches unit errors) |
| `validate_algebraic_prior_predictive` | AlgebraicModel forward prediction matches data scale |
| `validate_ode_requires_observable` | ODE models have observable in error_model |
| `validate_source_quality_peer_reviewed` | Warns if `source_quality` is `non_peer_reviewed` |
| `validate_secondary_sources_quality` | Warns if secondary sources have `non_peer_reviewed` quality |
| `validate_prior_reflects_translation_uncertainty` | Warns if prior σ is too narrow for `estimated_translation_uncertainty_fold` |
| `warn_multi_param_algebraic_identifiability` | Warns when AlgebraicModel has more params than measurements |
| `warn_observation_sd_unreasonable` | Warns if SD is >100x or <0.001x observed value |

**observation_code return signature** (required format):
```python
def derive_observation(inputs, sample_size, ureg):
    # ... computation ...
    return {
        'value': mean_value * ureg('pg/mL'),  # Pint Quantity, required
        'sd': sd_value,                        # float or Quantity, required
        'sd_uncertain': False,                 # bool, optional (True → prior on sigma)
        'n': sample_size,                      # int, optional
    }
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
  estimated_translation_uncertainty_fold: 10.0
```

**Key fields:**

| Field | Purpose | When to flag |
|-------|---------|--------------|
| `indication_match` | Does disease match? | `proxy`/`unrelated` require justification |
| `source_quality` | Peer review status | `non_peer_reviewed` (Wikipedia, preprints) triggers warning |
| `tme_compatibility` | TME similarity | `low` for PDAC if source is T cell-permissive tumor |
| `estimated_translation_uncertainty_fold` | Expected translation error | Should match prior σ: `σ ≈ ln(fold)` |

**Translation uncertainty → prior width:**
```
fold = 3.0  →  σ = ln(3.0) ≈ 1.1
fold = 10.0 →  σ = ln(10.0) ≈ 2.3
```

The `validate_prior_reflects_translation_uncertainty` validator warns if prior σ is less than 70% of `ln(estimated_translation_uncertainty_fold)`.

### LLM Extraction

Extract SubmodelTarget YAMLs from scientific literature using the CLI:

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

## Julia Translator

Translates SubmodelTarget YAMLs to Julia Turing.jl inference scripts. Located in `julia_translator.py`.

### Single Target

```python
from maple.core.calibration.julia_translator import JuliaTranslator

translator = JuliaTranslator()
julia_code = translator.generate_script("target.yaml")
```

### Joint Inference (Multiple Targets)

```python
from maple.core.calibration.julia_translator import JointInferenceBuilder

builder = JointInferenceBuilder()
julia_code = builder.build_from_files([
    "psc_proliferation.yaml",
    "psc_death.yaml",
    "psc_recruitment.yaml",  # Shares parameters with above
])
```

**Automatic parameter sharing**: Parameters with the same name across targets are automatically identified and shared in the joint Turing model.

### CLI Usage

```bash
# Single target (--model-structure required)
python -m maple.core.calibration.julia_translator \
    --model-structure model_structure.json \
    target.yaml

# Joint inference
python -m maple.core.calibration.julia_translator --joint \
    --model-structure model_structure.json \
    target1.yaml target2.yaml target3.yaml \
    --output joint_calibration.jl

# Joint inference with fixed sigmas (faster sampling)
python -m maple.core.calibration.julia_translator --joint \
    --model-structure model_structure.json \
    --fixed-sigma \
    target1.yaml target2.yaml target3.yaml
```

### What Gets Generated

The translator produces complete Julia scripts with:
- Observed data constants (value, sigma from observation_code)
- ODE functions for each model type (or compute functions for algebraic)
- Simulate wrapper functions
- Turing `@model` with priors and likelihoods (normal or lognormal)
- Priors on sigma when `sd_uncertain: true` in observation_code
- NUTS sampling code with convergence diagnostics
- Posterior marginal plots with prior overlays

## Package Structure

```
src/maple/
├── core/
│   ├── calibration/
│   │   ├── submodel_target.py      # SubmodelTarget schema (primary)
│   │   ├── julia_translator.py     # YAML → Julia/Turing.jl
│   │   ├── isolated_system_target.py  # Older schema
│   │   ├── calibration_target_models.py  # Base classes
│   │   └── ...
│   └── unit_registry.py            # Shared Pint UnitRegistry
├── cli/                            # Command-line tools
└── prompts/                        # LLM instruction prompts
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

### Git Commands

```bash
git status
git diff
git add .
```

### Running Tests

```bash
python -c "from maple import PromptAssembler; print('Import works')"
```

---

## Older Workflows (Experimental/Deprecated)

The sections below document older schemas that may be deprecated in favor of SubmodelTarget.

### IsolatedSystemTarget Schema

Earlier schema for in vitro/preclinical data. Uses Python submodel code instead of the typed model discriminated union.

**Key differences from SubmodelTarget:**
- Submodel code is Python (not typed model types)
- More complex nested structure
- No built-in Julia translation

**Structure:**
```
IsolatedSystemTarget(CalibrationTarget)
├── study_interpretation: str
├── key_assumptions: List[str]
├── parameters: List[str]             # Full model parameter names
├── submodel
│   ├── code: str                    # Python ODE code
│   ├── inputs: List[SubmodelInput]
│   ├── state_variables: List[SubmodelStateVariable]
│   ├── t_span: [t_start, t_end]
│   └── observable: SubmodelObservable
├── experimental_context
├── empirical_data
│   ├── median, ci95, units
│   ├── inputs: List[EstimateInput]
│   └── distribution_code: str       # Python distribution derivation
└── primary_data_source, secondary_data_sources
```

### Legacy CLI Commands

```bash
# Parameter extraction (legacy)
qsp-extract input.csv --type parameter --output-dir metadata-storage

# Test statistics (legacy)
qsp-extract test_stats.csv --type test_statistic --output-dir metadata-storage

# IsolatedSystemTarget extraction (older)
qsp-extract targets.csv --type isolated_system_target --output-dir metadata-storage
```

See [docs/](docs/) for additional documentation on these older workflows.
