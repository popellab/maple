# CLAUDE.md

Developer guide for Claude Code when working with this repository.

---

## Overview

MAPLE (Model-Aware Parameterization from Literature Evidence) provides schemas, validation, and extraction tooling for QSP calibration targets from scientific literature.

**Two target schemas:**
- **SubmodelTarget**: In vitro / preclinical data with self-contained forward models (ODE or algebraic)
- **CalibrationTarget**: Clinical / in vivo observables requiring full model simulation context, with Monte Carlo distribution derivation

**Inference lives in [qsp-inference](https://github.com/jeliason/qsp-inference)** — maple owns the data infrastructure (schemas, validators, extraction pipeline), while qsp-inference owns the statistical interpretation (MCMC, SBI, parameter audit).

## Installation

```bash
git clone https://github.com/popellab/maple.git
cd maple
pip install -e .
```

## SubmodelTarget Schema

For in vitro and preclinical data with self-contained forward models. Located in `submodel_target.py`.

### Key Design

**Separation of concerns:**
- `inputs`: Raw values extracted from papers with full provenance
- `calibration`: Everything needed for inference (parameters, model, measurements)

**Input types** (`InputType` enum) — strongly prefer `direct_measurement`:
- `direct_measurement`: Value traceable to paper text (requires snippet/table_excerpt/figure_excerpt). Preferred for all extracted values.
- `unit_conversion`: Dimensionless conversion factor (e.g., IQR-to-SD, pM-per-nM). Use sparingly — only for genuine unit conversions.
- `reference_value`: Normalization/reference constant (e.g., V_T_ref, tumor_cell_density). Use sparingly — only for genuine physical constants.
- `derived_arithmetic`: Deterministic derivation from other inputs via a formula (e.g., `E = 3*G'`)

**Derived arithmetic inputs** — for values calculated from other extracted inputs:
```yaml
- name: E_stiff_kPa
  value: 51.0
  units: kilopascal
  input_type: derived_arithmetic
  source_inputs: [Gprime_stiff_kPa]
  formula: "3 * Gprime_stiff_kPa"
  rationale: "E = 2*(1+nu)*G' with nu=0.5 for incompressible hydrogels"
  source_ref: Smith2020
  source_location: "Table 1"
```
Validators enforce: `derived_arithmetic` requires `formula` + `source_inputs`; formula is evaluated and checked against `value` (1% tolerance). Non-derived inputs must not have `formula`/`source_inputs`.

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
| `validate_derived_arithmetic_inputs` | Formula evaluates correctly, source_inputs exist, fields only on derived_arithmetic |
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

Translation sigma is computed from all 8 axes (added in quadrature, floor of 0.15) and applied inside the likelihood during joint inference. See `qsp_inference.submodel.prior:compute_translation_sigma()` in [qsp-inference](https://github.com/jeliason/qsp-inference) for the full rubric.

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

**MCP tools:**

- `extract_target(target_type)` — Loads the full extraction guide: multi-step workflow, prompt template, valid enum values, and hard rules. Call this before starting any extraction session.
- `validate_target(yaml_path, priors_csv, papers_dir)` — Runs schema validation (Pydantic) and snippet verification against source PDFs. `priors_csv` is required (e.g., `parameters/pdac_priors.csv`).
- `verify_dois(dois)` — Verifies DOIs resolve via CrossRef and returns metadata.
- `fetch_papers_from_zotero(dois)` — Fetches PDFs from Zotero for given DOIs into the papers directory.

> **Note:** Inference tools (`run_joint_inference`, `compare_inference`) have moved to [qsp-inference](https://github.com/jeliason/qsp-inference).

**Typical workflow:**

1. Call `extract_target` to load the guide
2. Investigate the parameter in the model code (units, mechanistic role, Hill function inputs)
3. Search literature for quantitative data that constrains the parameter
4. User obtains PDFs into `papers/<source_tag>/` directories
5. Read the paper — check figures for richer data (scatter plots, dose-response curves with error bars). Prefer digitizing figures via WebPlotDigitizer over text-reported summary statistics when figures contain more information.
6. Build the YAML incrementally — inputs first, then forward model, then error model, then source relevance
7. Call `validate_target` after each major section to catch errors early
8. Iterate until validation passes

### Batch Extraction (`maple.extraction`)

For extracting many parameters at once, use the staged pipeline in `maple.extraction`. This replaces the old `qsp-extract` CLI with a more capable multi-stage workflow. See the README for the full pipeline description.

```python
from maple.extraction import (
    make_agents, run_lit_search, run_assess, run_plan_review,
    run_complete, run_derivation_review, run_validate, run_stage,
    collect_missing_pdfs, summarize_digitizations,
    write_assessment_report, write_dois_md,
)
```

**Key modules:**
- `maple.extraction.pipeline` — schemas, prompts, agents, stage functions
- `maple.extraction` — re-exports everything

**Input CSV format:**
```csv
target_id,parameters,cancer_type,notes
k_IL2_sec,k_IL2_sec,PDAC,"Per-cell IL-2 secretion rate. Search for: ELISA, single-cell secretion rates."
k_vas_growth,k_vas_growth,PDAC,"Rate law: dK/dt = k_vas_growth * C_total * VEGF/(VEGF+VEGF_50). Search for: MVD growth kinetics."
```

**Stages:** lit search → PDF collection → paper assessment → plan review → digitization → extraction → derivation review → validation. Each stage caches per-target; delete a cache file to rerun that stage for that target.

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

CalibrationTarget extraction uses the interactive MCP workflow (see README). The staged batch pipeline (`maple.extraction`) currently supports SubmodelTarget extraction only.

**Input CSV format:**
```csv
observable_name,description,species,compartment,notes
cd8_tumor_density,CD8+ T cell density in primary tumor,CD8_T,tumor.primary,Treatment-naive PDAC
```

## Submodel Prior Inference

**Moved to [qsp-inference](https://github.com/jeliason/qsp-inference).** Joint Bayesian inference (NumPyro MCMC, NPE, parameter audit) now lives in the `qsp_inference.submodel` package. Maple provides the schemas (`SubmodelTarget`, `SourceRelevanceAssessment`) that qsp-inference consumes.

## Package Structure

```
src/maple/
├── core/
│   ├── calibration/
│   │   ├── submodel_target.py         # SubmodelTarget schema
│   │   ├── calibration_target_models.py  # CalibrationTarget schema
│   │   ├── code_validator.py          # Python code validation
│   │   ├── enums.py                   # Shared enums
│   │   ├── exceptions.py              # Validation error hierarchy
│   │   ├── experimental_context.py    # ExperimentalContext model
│   │   ├── observable.py              # Observable models
│   │   ├── scenario.py                # Scenario models
│   │   ├── shared_models.py           # Shared Pydantic models (SourceRelevanceAssessment, etc.)
│   │   ├── snippet_validator.py       # Snippet-in-paper verification
│   │   └── validators.py              # DOI, fuzzy matching, value validation
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
├── extraction/
│   ├── __init__.py                    # Re-exports pipeline components
│   └── pipeline.py                    # Staged batch extraction pipeline
├── cli/
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

- `qsp-export-model` — Export SimBiology model structure to JSON

For batch extraction, use the staged pipeline via `maple.extraction` (see README).
