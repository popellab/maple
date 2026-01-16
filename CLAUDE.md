# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note for end users:** If you're looking for setup and usage instructions, see [docs/calibration_workflow.md](docs/calibration_workflow.md) for the main guide. This file is for developers and contributors.

---

## Overview

This repository contains LLM workflow automation tools for extracting calibration targets from scientific literature for QSP model calibration. It uses Pydantic AI and OpenAI's API to read papers, extract experimental data with uncertainty, and generate Python code for Bayesian inference.

**Primary Workflow:**
- **IsolatedSystemTarget**: Extract calibration targets from in vitro, ex vivo, or preclinical data. Generates ODE submodels that share parameter names with the full QSP model for joint Bayesian inference.

**Secondary Workflows:**
- **CalibrationTarget**: For clinical/in vivo data where the full model is needed
- **Parameter extraction** (legacy): Direct parameter value extraction
- **Test statistics** (legacy): Validation constraints from experimental data

All extracted metadata is stored in a user-specified output directory (e.g., `metadata-storage/`) as YAML files.

## Repository Organization

**This repository (`qsp-llm-workflows`):**
- Installable Python package for QSP metadata extraction workflows
- Reusable across any QSP model or disease area
- Focus: Core extraction, validation, and storage workflows

**Paper repository (`qsp-llm-workflows-paper`, to be created):**
- Paper-specific code, validation analyses, and manuscript figures
- Validation study comparing LLM extraction to legacy parameter database

## Installation

```bash
git clone https://github.com/popellab/qsp-llm-workflows.git
cd qsp-llm-workflows
python -m venv venv
source venv/bin/activate
pip install -e .

echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

After installation, CLI commands (`qsp-extract`, `qsp-validate`, etc.) are available system-wide.

## Key Commands

### IsolatedSystemTarget Workflow (Primary)

**Step 1: Export model structure**
```bash
qsp-export-model \
  --matlab-model ../your-model/model.m \
  --output jobs/input_data/model_definitions.json \
  --export-structure jobs/input_data/model_structure.json
```

This creates:
- `model_definitions.json`: Flat parameter definitions for prompts
- `model_structure.json`: Structured model data for validation (parameters, species, reactions)
- `species_units.json`: Unit information for each species

**Step 2: Create input CSV**
```csv
target_id,cancer_type,parameters,notes
cd8_proliferation,PDAC,"k_CD8_pro","CD8 T cell proliferation from in vitro assays"
spheroid_growth,PDAC,"k_C1_growth,C_max","Cancer growth from spheroid experiments"
```

**Step 3: Run extraction**
```bash
qsp-extract \
  targets.csv \
  --type isolated_system_target \
  --output-dir metadata-storage \
  --model-structure jobs/input_data/model_structure.json \
  --model-context jobs/input_data/model_context.txt
```

Options:
- `--preview-prompts`: Preview prompts without calling API
- `--reasoning-effort low|medium|high`: Control reasoning depth (default: low)

Results are unpacked to `metadata-storage/to-review/isolated_system_targets/`.

### Legacy Workflows

```bash
# Parameter extraction
qsp-extract input.csv --type parameter --output-dir metadata-storage

# Test statistics
qsp-extract test_stats.csv --type test_statistic --output-dir metadata-storage

# Validation (for legacy workflows)
qsp-validate parameter_estimates --dir metadata-storage/to-review/parameter_estimates
qsp-validate test_statistics --dir metadata-storage/to-review/test_statistics \
  --species-units-file jobs/input_data/species_units.json
```

See [docs/automated_workflow.md](docs/automated_workflow.md) for legacy workflow documentation.

## Package Structure

```
qsp-llm-workflows/
├── src/
│   └── qsp_llm_workflows/           # Main package
│       ├── __init__.py              # Package version, public API
│       │
│       ├── core/                     # Core libraries
│       │   ├── prompt_builder.py    # Prompt generation
│       │   ├── workflow_orchestrator.py  # Workflow automation
│       │   ├── unit_registry.py     # Shared Pint UnitRegistry
│       │   │
│       │   └── calibration/         # Calibration target models
│       │       ├── __init__.py      # Re-exports all calibration classes
│       │       ├── calibration_target_models.py  # CalibrationTarget base class
│       │       ├── isolated_system_target.py  # IsolatedSystemTarget
│       │       ├── observable.py        # Observable, Submodel, SubmodelObservable
│       │       ├── shared_models.py     # EstimateInput, SubmodelInput, Source
│       │       ├── enums.py             # Species, Indication, Compartment, System
│       │       ├── scenario.py          # Intervention, Scenario
│       │       ├── experimental_context.py  # ExperimentalContext
│       │       ├── code_validator.py    # Unified code validation
│       │       └── exceptions.py        # Calibration validation exceptions
│       │
│       ├── cli/                      # CLI entry points
│       │   ├── extract.py           # qsp-extract
│       │   ├── validate.py          # qsp-validate
│       │   └── export_model.py      # qsp-export-model
│       │
│       ├── process/                  # Result processing
│       │   └── unpack_results.py
│       │
│       └── prompts/                  # LLM instruction prompts
│           ├── isolated_system_target_prompt.md
│           ├── calibration_target_prompt.md
│           └── ...
│
├── pyproject.toml                    # Package metadata & dependencies
├── README.md
├── CLAUDE.md
└── .env                              # API keys (gitignored)
```

## Architecture

### Calibration Target Model Architecture

The calibration target models use a modular, inheritance-based architecture with **co-located inputs** - each code block has its own inputs for clarity.

**Key Models:**
- `IsolatedSystemTarget`: Primary model for in vitro/preclinical data. Uses `submodel` with nested ODE code that shares parameter names with the full model for joint inference.
- `CalibrationTarget`: Base class for clinical/in vivo data. Uses `observable` to compute measurements from full model species.
- `Submodel`: ODE code, inputs, state variables, parameters, and a nested `SubmodelObservable`
- `SubmodelStateVariable`: Self-contained with initial value + provenance (no reference indirection)

**Input Architecture:**
```
Inputs are co-located with the code blocks that use them:

- submodel.inputs (SubmodelInput)      # Experimental conditions for ODE code
- submodel.state_variables             # Self-contained with initial values + provenance
- observable.inputs (SubmodelInput)    # Literature inputs for observable code
- observable.constants                 # Geometric/modeling constants
- empirical_data.inputs                # EstimateInput for distribution_code
- empirical_data.assumptions           # ModelingAssumption for derivation
```

**Input Classes:**
- `SubmodelInput`: Experimental conditions for submodel/observable code (E:T ratio, dose, etc.)
- `EstimateInput`: Literature values for distribution_code derivation (mean, SD, etc.)
- `ModelingAssumption`: Computational assumptions with rationale (n_mc_samples, etc.)
- `ObservableConstant`: Geometric/modeling constants with biological_basis

**Input Type Classification (`EstimateInput.input_type`):**
- `direct_parameter`: Value reported directly in paper (e.g., "mean = 42.0")
- `proxy_measurement`: Requires conversion (e.g., "doubling time = 8h" → rate constant)
- `experimental_condition`: Protocol choice from paper (e.g., seeding density, E:T ratio)
- `inferred_estimate`: Value interpreted from qualitative text (skips snippet validation)

**Figure Source Support:**
Input classes support figure-extracted data via:
- `source_type`: `text` (default), `table`, or `figure`
- `figure_id`: Figure identifier (required when source_type is `figure`)
- `extraction_method`: `manual`, `digitizer`, `webplotdigitizer`, or `other`

**Model Structure:**
```
IsolatedSystemTarget(CalibrationTarget)
├── study_interpretation: str         # Scientific interpretation
├── key_assumptions: List[str]        # Required (min 1)
├── key_study_limitations: List[str]
├── submodel                          # ODE submodel (replaces observable)
│   ├── code: str                    # submodel(t, y, params, inputs) -> [dydt]
│   ├── inputs: List[SubmodelInput]  # Experimental conditions
│   ├── state_variables: List[SubmodelStateVariable]
│   │   └── name, units, initial_value, source_ref, value_location, value_snippet
│   ├── parameters: List[str]        # Full model parameter names
│   ├── t_span: [t_start, t_end]
│   ├── t_unit: str                  # e.g., "day", "hour"
│   ├── observable: SubmodelObservable
│   │   ├── code: Optional[str]      # compute_observable(t, y, constants, ureg)
│   │   ├── units: str               # Required
│   │   └── constants: List[ObservableConstant]
│   └── rationale: str
├── experimental_context
│   └── species, indication, compartment, system, etc.
├── scenario (optional)
├── empirical_data
│   ├── median: List[float]          # Length-1 for scalar, longer for vector
│   ├── ci95: List[List[float]]
│   ├── units: str
│   ├── sample_size: Union[int, List[int]]
│   ├── sample_size_rationale: str
│   ├── index_values, index_unit, index_type  # For vector data
│   ├── inputs: List[EstimateInput]
│   ├── assumptions: List[ModelingAssumption]
│   └── distribution_code: str
└── primary_data_source, secondary_data_sources
```

**Direct Conversion Mode:**
For simple analytical relationships (k = ln(2) / t_half), omit the submodel:
```yaml
submodel: null  # distribution_code computes parameter directly
```

**Submodel Observable Simplification:**
If observable is just a state variable (no transformation), omit `observable.code`:
```yaml
observable:
  units: cell  # Defaults to return y[0] * ureg(units)
```

**Import Pattern:**
```python
from qsp_llm_workflows.core.calibration import (
    CalibrationTarget, IsolatedSystemTarget, IndexType, InputType,
    Observable, Submodel, SubmodelStateVariable,
    EstimateInput, SubmodelInput, ModelingAssumption,
    CodeType, CodeValidator, validate_code_block, find_hardcoded_constants,
)
```

### Parameter Context (IsolatedSystemTarget)

When extracting IsolatedSystemTarget data, all parameter context is injected into the prompt at build time from `model_structure.json`:
- Parameter units and descriptions
- Reactions using each parameter with rate laws
- Related species and other parameters in the same reactions
- Broader reaction network context

The `model_structure.json` file (generated with `--export-structure` flag) is the single source of truth for validation context.

### Shared Pint UnitRegistry

All code that handles units must use the shared Pint UnitRegistry:

```python
from qsp_llm_workflows.core.unit_registry import ureg

# Creating quantities
time = np.linspace(0, 14, 100) * ureg.day
concentration = 5.0 * ureg.nanomolarity

# Unit conversions
density.to('cell / mm**3')
```

**Custom units defined:**
- `cell = [cell_count]` - for cell counts and densities
- `nanomolarity = nanomolar` - SimBiology convention alias

**IMPORTANT:** Never create a new `pint.UnitRegistry()`. Always import `ureg` from the shared module.

### Unified Code Validation

All code blocks are validated using `CodeValidator` from `code_validator.py`:

- `submodel.code` - ODE function `submodel(t, y, params, inputs)`
- `submodel.observable.code` - Observable `compute_observable(t, y, constants, ureg)`
- `observable.code` - Full model observable `compute_observable(time, species_dict, constants, ureg)`
- `distribution_code` - Distribution derivation `derive_distribution(inputs, ureg)`

**Hardcoded Constant Detection:**
The validator uses AST analysis to find numeric literals multiplied by ureg units. All numbers with units must come from `inputs`, `assumptions`, or `constants` dicts.

### Data Flow (IsolatedSystemTarget)

1. **Model Export**: Export `model_structure.json` from MATLAB
2. **Input CSV**: User creates CSV with target_id, cancer_type, parameters, notes
3. **Prompt Generation**: System injects parameter context from model_structure.json
4. **LLM Processing**: Pydantic AI extracts data and generates submodel code
5. **Unpacking**: Results unpacked to `to-review/isolated_system_targets/`
6. **Validation**: Pydantic validators check code execution, units, snippets

### YAML Field Ordering

Output YAML files use a consistent field order defined in `unpack_results.py`:
1. study_interpretation, key_assumptions, key_study_limitations (narrative)
2. submodel/observable (model structure)
3. experimental_context, scenario (context)
4. empirical_data (data)
5. primary_data_source, secondary_data_sources (sources)
6. tags, metadata (footers)

## Key Design Principles

**Package Architecture:**
- **Installable**: `pip install -e .` for development
- **CLI-first**: Commands available system-wide after install
- **Resource Management**: Uses `importlib.resources` for template/prompt access
- **No sys.path manipulation**: Clean imports throughout
- **Class-based**: Prompt builders inherit from `PromptBuilder` base class

**Code Standards:**
- **No backward compatibility**: Use clean, modern interfaces
- **Class-focused architecture**: Prefer class-based designs
- **Explicit interfaces**: Require all necessary arguments
- **Shared UnitRegistry**: Always use `ureg` from unit_registry.py

## Development

### Git Commands

Use plain `git` commands (not `git -C`):

```bash
git status
git diff
git add .
```

### Running Tests

```bash
python -c "from qsp_llm_workflows import PromptAssembler; print('✓ Import works')"
qsp-extract --help
```

### Adding New Validators

Validators are classes with a `.validate()` method that returns a ValidationReport:

```python
from qsp_llm_workflows.core.validation_utils import ValidationReport

class MyValidator:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def validate(self) -> dict:
        report = ValidationReport("My Validation")
        report.add_pass(filename, message)
        return report.to_dict()
```

Then import and call in `run_all_validations.py`.
