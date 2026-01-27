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
git clone https://github.com/popellab/qsp-llm-workflows.git
cd qsp-llm-workflows
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

**Model types** (discriminated union):
- `exponential_growth`: dy/dt = k * y
- `first_order_decay`: dy/dt = -k * y
- `two_state`: A → B transition
- `saturation`: dy/dt = k * (1 - y)
- `logistic`: dy/dt = k * y * (1 - y/K)
- `michaelis_menten`: dy/dt = -Vmax * y / (Km + y)
- `direct_conversion`: No ODE, analytical formula
- `custom`: User-provided ODE code (requires `code_julia`)

**Custom ODE models:**
```yaml
model:
  type: custom
  code: |
    def ode(t, y, params, inputs):
        ...
  code_julia: |
    function my_ode!(du, u, p, t)
        ...
    end
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
| `validate_distribution_code_return_signature` | `distribution_code` returns `{median, ci95_lower, ci95_upper}` |
| `validate_no_invisible_characters` | No invisible/control chars (zero-width spaces, soft hyphens, etc.) |
| `validate_span_ordering` | `span[0] < span[1]` and both non-negative |
| `validate_input_values_in_snippets` | Extracted values appear in `value_snippet` (anti-hallucination) |
| `validate_doi_resolution_and_metadata` | DOIs resolve via CrossRef, metadata matches |
| `validate_units_are_valid_pint` | All unit strings are valid Pint units |
| `validate_distribution_code_required_with_formula` | `direct_conversion` models have `distribution_code` |
| `validate_prior_predictive_scale` | Prior prediction matches observation scale (catches unit errors) |
| `validate_source_quality_peer_reviewed` | Warns if `source_quality` is `non_peer_reviewed` |
| `validate_secondary_sources_quality` | Warns if secondary sources have `non_peer_reviewed` quality |
| `validate_prior_reflects_translation_uncertainty` | Warns if prior σ is too narrow for `estimated_translation_uncertainty_fold` |

**distribution_code return signature** (required format):
```python
def derive_distribution(inputs, ureg):
    # ... computation ...
    return {
        'median': float(np.median(samples)),
        'ci95_lower': float(np.percentile(samples, 2.5)),
        'ci95_upper': float(np.percentile(samples, 97.5)),
    }
```

**No invisible characters** - catches PDF copy-paste issues:
- Zero-width spaces, soft hyphens, byte order marks
- Control characters (except tab, newline, carriage return)
- Unicode letters (Greek, accents) and math symbols (±, ≥) are allowed

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
from qsp_llm_workflows.core.calibration.julia_translator import JuliaTranslator

translator = JuliaTranslator()
julia_code = translator.generate_script("target.yaml")
```

### Joint Inference (Multiple Targets)

```python
from qsp_llm_workflows.core.calibration.julia_translator import JointInferenceBuilder

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
# Single target
python -m qsp_llm_workflows.core.calibration.julia_translator target.yaml

# Joint inference
python -m qsp_llm_workflows.core.calibration.julia_translator --joint \
    target1.yaml target2.yaml target3.yaml \
    --output joint_calibration.jl
```

### What Gets Generated

The translator produces complete Julia scripts with:
- Observed data constants (median, sigma from CI95)
- ODE functions for each model type
- Simulate wrapper functions
- Turing `@model` with priors and likelihoods
- NUTS sampling code with convergence diagnostics

## Package Structure

```
src/qsp_llm_workflows/
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
from qsp_llm_workflows.core.unit_registry import ureg

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
python -c "from qsp_llm_workflows import PromptAssembler; print('Import works')"
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
