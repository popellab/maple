# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note for end users:** If you're looking for setup and usage instructions, see [docs/automated_workflow.md](docs/automated_workflow.md) for a beginner-friendly guide. This file is for developers and contributors.

---

## Overview

This repository contains LLM workflow automation tools for extracting and validating quantitative systems pharmacology (QSP) metadata from scientific literature using Pydantic AI and OpenAI's API.

**Supported Workflows:**
- **Parameter extraction**: Extract parameter values, ranges, and statistical distributions with detailed literature tracking
- **Test statistics**: Create validation constraints from experimental data with uncertainty quantification
- **Quick estimates**: Rapid batch estimation for calibration targets (CSV in → CSV out, single LLM request)

All extracted metadata is stored in a user-specified output directory (e.g., `metadata-storage/`) with flat file structures for easy access.

## Repository Organization

**This repository (`qsp-llm-workflows`):**
- Installable Python package for QSP metadata extraction workflows
- Reusable across any QSP model or disease area
- Focus: Core extraction, validation, and storage workflows

**Paper repository (`qsp-llm-workflows-paper`, to be created):**
- Paper-specific code, validation analyses, and manuscript figures
- Validation study comparing LLM extraction to legacy parameter database
- Reproducible research for publication

**Manuscript documentation (`docs-manuscript/`):**
- Paper collaboration materials (gitignored, shared via email)
- Includes onboarding guide, presentation, and paper outline
- Not checked into repository to keep codebase focused on reusable tools

## Installation

### For Development (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourorg/qsp-llm-workflows.git
cd qsp-llm-workflows

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### For End Users

```bash
# Install directly from GitHub
pip install git+https://github.com/yourorg/qsp-llm-workflows.git

# Or install from PyPI (if published)
pip install qsp-llm-workflows
```

After installation, CLI commands (`qsp-extract`, `qsp-validate`, etc.) are available system-wide.

## Key Commands

### CSV Preparation (Step 1)

**Before running the extraction workflow**, you must prepare an enriched input CSV with model definitions.

#### Parameter Extraction CSV

Simple input (parameter names only):
```csv
parameter_name
k_C_growth
k_C_death
```

**Export model definitions** (from MATLAB script or SimBiology project):
```bash
# Option 1: Export from MATLAB script
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/your_model_file.m \
  --output jobs/input_data/model_definitions.json

# Option 2: Export from SimBiology project file (faster if already compiled)
qsp-export-model \
  --simbiology-project ../your-model-repo/models/your_model.sbproj \
  --output jobs/input_data/model_definitions.json
```

**Enrich with model definitions**:
```bash
# Enrich simple CSV with model definitions
qsp-enrich-csv parameter \
  simple_input.csv \
  jobs/input_data/model_definitions.json \
  YOUR_CANCER_TYPE \
  -o jobs/input_data/enriched_extraction_input.csv
```

This creates an enriched CSV with:
- `cancer_type`, `parameter_name`
- `parameter_units`, `parameter_description`
- `model_context` (JSON with reactions, rules, related parameters)

#### Test Statistic CSV

**New workflow:** For test statistics, the user provides the `compute_test_statistic` function code directly. This ensures the computation is exactly what you intend. The LLM then focuses on extracting literature values that match your test statistic definition.

Pre-enriched input (user provides the code):
```csv
test_statistic_id,output_unit,model_output_code
tgfb_concentration_baseline,nanomolarity,"def compute_test_statistic(time, species_dict, ureg):
    return species_dict['V_T.TGFb'][0]"
tumor_volume_day14,millimeter ** 3,"def compute_test_statistic(time, species_dict, ureg):
    import numpy as np
    cells = species_dict['V_T.C1']
    day_14_idx = np.argmin(np.abs(time.magnitude - 14))
    return cells[day_14_idx] * (1e-6 * ureg.millimeter**3 / ureg.cell)"
```

**Key points:**
- `output_unit`: Pint-parseable unit string (e.g., `nanomolarity`, `millimeter ** 3`, `1 / day`, `dimensionless`)
- `model_output_code`: Python function with signature `(time, species_dict, ureg)` where:
  - `time`: numpy array with Pint day units
  - `species_dict`: dict mapping species names to Pint quantities (e.g., `species_dict['V_T.TGFb']` returns nanomolar values)
  - `ureg`: Pint UnitRegistry for unit conversions
- Function must return a Pint Quantity with the declared `output_unit`

**Export model and species units first:**
```bash
# Export model definitions AND species_units.json
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/your_model_file.m \
  --output jobs/input_data/model_definitions.json
# This also creates jobs/input_data/species_units.json containing:
# - Species units (e.g., V_T.CD8: cell, V_T.TGFb: nanomolarity)
# - Parameter units (e.g., initial_tumour_diameter: centimeter)
# - Compartment volumes (e.g., V_T: milliliter, V_C: liter)
```

**Enrich with scenario context and validate units:**
```bash
# Enrich and validate (validates Pint units during enrichment)
qsp-enrich-csv test_statistic \
  test_stats_input.csv \
  scenario.yaml \
  jobs/input_data/species_units.json \
  -o jobs/input_data/test_statistic_input.csv
```

**Validation during enrichment:**
- Parses `model_output_code` and checks function signature
- Extracts species accessed from code (via AST)
- Validates all accessed species exist in `species_units.json`
- Executes code with Pint-wrapped mock data
- Verifies output has correct unit dimensionality

This creates an enriched CSV with:
- `test_statistic_id`, `output_unit`, `model_output_code`
- `scenario_context`

**Note:** Model definitions and species units are exported from MATLAB model files. Scenario context files are stored in model-specific repositories (e.g., `your-model-repo/scenarios/`).

### Automated Workflow (Step 2)

**Single-command automated extraction** - Handles prompt generation, processing, and unpacking:

```bash
# Parameter extraction (complete pipeline)
qsp-extract input.csv --type parameter --output-dir metadata-storage

# Test statistics
qsp-extract test_stats.csv --type test_statistic --output-dir metadata-storage

# With custom timeout (default: 3600s)
qsp-extract input.csv --type parameter --output-dir metadata-storage --timeout 7200

# Use immediate mode for faster processing (via Responses API)
qsp-extract input.csv --type parameter --output-dir metadata-storage --immediate
```

**What the automated workflow does:**
1. Generates prompts from input CSV
2. Processes requests via Pydantic AI (shows progress)
3. Unpacks results to `<output-dir>/to-review/`
4. Prints summary with next steps

**After workflow completes, manually run validation:**
```bash
# For parameter estimates
qsp-validate parameter_estimates --dir metadata-storage/to-review/parameter_estimates

# For test statistics (requires species_units.json for unit validation)
qsp-validate test_statistics \
  --dir metadata-storage/to-review/test_statistics \
  --species-units-file jobs/input_data/species_units.json
```

See `docs/automated_workflow.md` for complete documentation.

### Validation Suite

The automated validation suite includes 9 validators:

1. **Schema Compliance** - YAML structure matches template
2. **Code Execution** - R/Python derivation_code runs without errors
3. **Model Output Code** - Test statistic `compute_test_statistic` function validates (correct signature, returns Pint Quantity with correct units)
4. **Text Snippets** - Snippets contain declared values
5. **Source References** - All source_refs point to defined sources
6. **DOI Validity** - DOIs resolve and metadata matches
7. **Value Consistency** - Values consistent across related extractions
8. **Duplicate Primary Sources** - Primary data sources not already used in accepted extractions
9. **Automated Snippet Source Verification** - Verifies snippets via Europe PMC API

**Manual Snippet Source Verification**:
- Generates report with DOI links and snippets grouped by source
- Prints report to console during validation
- Waits for user to manually verify snippets in papers (y/n prompt)
- User clicks DOI links and uses Ctrl+F/Cmd+F to search for snippets
- Writes validation results to `snippet_sources.json`
- If verified, adds `manual_snippet_source_verification` tag to all YAML files

**Validation Tagging**:
After all validation checks complete, the suite automatically:
- Tags all YAML files with passed validation checks
- Appends `validation` section to end of each file (preserves formatting)
- Includes list of passed checks and timestamp
- Example tag: `template_compliance_validation`, `code_execution_testing`

```yaml
# Validation metadata (added to end of each file)
validation:
  tags:
    - template_compliance_validation
    - code_execution_testing
    - manual_snippet_source_verification
  validated_at: '2025-11-03T10:30:00'
```

### Quick Estimate Workflow

**Simple batch workflow for rapid calibration target estimates** - CSV in, single LLM request, CSV out.

**Input CSV structure:**
```csv
calibration_target_id,cancer_type,observable_description,model_species,model_indication,model_compartment,model_system,model_treatment_history,model_stage_burden,relevant_compartments
cd8_tumor_density_baseline,PDAC,"CD8+ T cell density in untreated PDAC tumors at baseline",human,PDAC,tumor.primary,clinical.resection,treatment_naive,resectable,V_T
```

**Run quick estimate:**
```bash
qsp-quick-estimate input.csv -o output.csv
```

**Output CSV contains:**
- `calibration_target_id`: Target ID from input
- `estimate`: Numeric value
- `units`: Pint-parseable units (e.g., `cell / millimeter**2`, `nanomolarity`)
- `uncertainty`: Uncertainty value if available
- `uncertainty_type`: Type of uncertainty (`se`, `sd`, `ci95`, `range`, `iqr`, `other`)
- `value_snippet`: Exact text from paper
- `paper_name`: Full paper title
- `doi`: Paper DOI
- `threshold_description`: Human-readable context description

**Key features:**
- **Single LLM request** for all targets (not one per row)
- **Strict model context matching** enforced in prompt (species, indication, system)
- **Fast turnaround** for initial scoping and ballpark estimates
- No validation suite or YAML unpacking (quick and simple)

### Other CLI Commands

```bash
# Export model definitions (from MATLAB script or SimBiology project)
qsp-export-model --matlab-model model.m --output defs.json
qsp-export-model --simbiology-project model.sbproj --output defs.json
```

## Package Structure

This repository is organized as an installable Python package:

```
qsp-llm-workflows/
├── src/
│   └── qsp_llm_workflows/           # Main package
│       ├── __init__.py              # Package version, public API
│       │
│       ├── core/                     # Core libraries
│       │   ├── prompt_builder.py    # Prompt generation
│       │   ├── prompt_assembly.py   # Modular prompt assembly engine
│       │   ├── workflow_orchestrator.py  # Workflow automation
│       │   ├── parameter_utils.py   # Parameter processing
│       │   ├── model_definition_exporter.py
│       │   ├── validation_utils.py  # Validation utilities
│       │   ├── resource_utils.py    # Package resource access
│       │   ├── header_utils.py      # Header field management
│       │   ├── schema_version_detector.py
│       │   ├── quick_estimate_models.py  # Quick estimate Pydantic models
│       │   │
│       │   │   # Calibration target models (modular architecture)
│       │   ├── calibration_target_models.py  # CalibrationTarget base class
│       │   ├── isolated_system_target.py  # IsolatedSystemTarget + Cut models
│       │   ├── shared_models.py     # Input, Source, Snippet, TrajectoryData, etc.
│       │   ├── enums.py             # Species, Indication, Compartment, System enums
│       │   ├── scenario.py          # Intervention, Measurement, Scenario
│       │   ├── experimental_context.py  # Stage, TreatmentContext, ExperimentalContext
│       │   ├── validators.py        # Validation helper functions
│       │   ├── exceptions.py        # Custom exception classes
│       │   └── unit_registry.py     # Shared Pint UnitRegistry
│       │
│       ├── prepare/                  # Prompt generation
│       │   ├── create_parameter_prompts.py
│       │   ├── create_test_statistic_prompts.py
│       │   ├── enrich_parameter_csv.py
│       │   └── enrich_test_statistic_csv.py
│       │
│       ├── run/                      # Request processing
│       │   └── upload_immediate.py
│       │
│       ├── process/                  # Result processing
│       │   └── unpack_results.py
│       │
│       ├── validate/                 # Validation checks
│       │   ├── run_all_validations.py
│       │   ├── check_schema_compliance.py
│       │   ├── check_text_snippets.py
│       │   ├── test_code_execution.py
│       │   ├── check_doi_validity.py
│       │   ├── check_source_references.py
│       │   ├── check_value_consistency.py
│       │   ├── check_duplicate_primary_sources.py
│       │   └── check_snippet_sources_manual_verify.py
│       │
│       ├── cli/                      # CLI entry points
│       │   ├── extract.py           # qsp-extract
│       │   ├── validate.py          # qsp-validate
│       │   ├── enrich.py            # qsp-enrich-csv
│       │   ├── export_model.py      # qsp-export-model
│       │   └── quick_estimate.py    # qsp-quick-estimate
│       │
│       ├── templates/                # YAML templates (package data)
│       │   ├── parameter_metadata_template.yaml
│       │   ├── test_statistic_template.yaml
│       │   ├── configs/
│       │   │   └── prompt_assembly.yaml
│       │   └── examples/
│       │
│       └── prompts/                  # Prompt files (package data)
│           ├── parameter_prompt.md
│           └── test_statistic_prompt.md
│
├── pyproject.toml                    # Package metadata & dependencies
├── README.md
├── CLAUDE.md
└── .env                              # API keys (gitignored)
```

## Architecture

### Modular Prompt Assembly System

The package uses a generalized prompt assembly system that builds prompts from modular components:

- **Prompts** (`prompts/`): Base prompt files with placeholders
- **Templates** (`templates/`): YAML templates and examples
- **Configs** (`templates/configs/`): Configuration for prompt assembly
- **Core** (`core/`): Core libraries (prompt generation, workflow orchestration)

### Data Flow

**Parameter Extraction Workflow:**
1. **CSV Enrichment**: Simple CSV (parameter names) + model definitions JSON → enriched CSV
2. **Prompt Generation**: System generates prompts with model context
3. **LLM Processing**: Pydantic AI processes requests and creates structured YAML outputs
4. **Unpacking**: Results unpacked to `<output-dir>/to-review/parameter_estimates/`
5. **Validation**: Automated validation suite checks quality and completeness

**Test Statistics Workflow:**
1. **Model Export**: Export model definitions + `species_units.json` from MATLAB
2. **CSV Enrichment**: User-provided CSV (with `model_output_code`) + scenario YAML + species_units.json → enriched CSV
   - **Pint unit validation**: During enrichment, each `compute_test_statistic` function is:
     - Parsed (AST checks signature is `(time, species_dict, ureg)`)
     - Validated (all accessed species exist in model)
     - Executed with Pint-wrapped mock data
     - Verified to return correct unit dimensionality
3. **Prompt Generation**: System generates prompts with full context
4. **LLM Processing**: LLM extracts literature values matching the user-defined test statistic
5. **Unpacking**: Results unpacked to `<output-dir>/to-review/test_statistics/`
6. **Aggregation**: Distributions pooled using inverse-variance weighting

### Shared Pint UnitRegistry

All code that handles units must use the shared Pint UnitRegistry from `unit_registry.py`:

```python
from qsp_llm_workflows.core.unit_registry import ureg
```

**Why a shared registry?**
- Pint quantities from different registries cannot be compared or combined
- Custom units (`cell`, `nanomolarity`) are defined once in the shared registry
- All validators, code execution, and enrichment use the same registry

**Custom units defined:**
- `cell = [cell_count]` - for cell counts and densities
- `nanomolarity = nanomolar` - SimBiology convention alias
- `micromolarity`, `millimolarity`, `molarity` - additional SimBiology aliases

**Usage in code:**
```python
from qsp_llm_workflows.core.unit_registry import ureg

# Creating quantities
time = np.linspace(0, 14, 100) * ureg.day
concentration = 5.0 * ureg.nanomolarity

# Unit conversions
density.to('cell / mm**3')

# Dimensionless ratios
ratio.to(ureg.dimensionless)
```

**IMPORTANT:** Never create a new `pint.UnitRegistry()` in validation or processing code. Always import `ureg` from the shared module.

### Calibration Target Model Architecture

The calibration target models use a modular, inheritance-based architecture:

```
calibration_target_models.py:
  CalibrationTarget (base class)
  ├── calibration_target_estimates (median, iqr, ci95, distribution_code)
  ├── scenario (interventions + measurements)
  ├── experimental_context (species, compartment, system + optional clinical/in vitro fields)
  ├── study_overview, study_design, key_assumptions, key_study_limitations
  └── primary_data_source, secondary_data_sources

isolated_system_target.py:
  IsolatedSystemTarget(CalibrationTarget)
  └── cuts: List[Cut]  # Species, compartment, or reaction cuts for reduced models

  Cut models:
  ├── SpeciesCut (clamped, excluded, zero_flux, prescribed)
  ├── CompartmentCut (excluded)
  └── ReactionCut (disabled)
```

**Key models:**
- `CalibrationTarget` (in `calibration_target_models.py`): For clinical/in vivo data where full model is assumed
- `IsolatedSystemTarget` (in `isolated_system_target.py`): Extends CalibrationTarget with `cuts` for in vitro experiments
- `ExperimentalContext`: Unified context supporting both clinical (indication, treatment, stage) and in vitro (cell_lines, culture_conditions) fields
- `TrajectoryData` / `DoseResponseData`: Optional structured multi-point data in shared_models.py

**Cut types for isolated systems (in `isolated_system_target.py`):**
- `SpeciesCut`: Clamp, exclude, zero_flux, or prescribe a species
- `CompartmentCut`: Exclude entire compartment
- `ReactionCut`: Disable a reaction

### Key Design Principles

**Package Architecture:**
- **Installable**: `pip install -e .` for development, `pip install` for distribution
- **CLI-first**: Commands like `qsp-extract` available system-wide after install
- **Resource Management**: Uses `importlib.resources` for robust template/prompt access
- **No sys.path manipulation**: Clean imports throughout
- **Library code is library code**: Validation scripts are imported, not called via subprocess
- **Class-based**: Prompt builders inherit from `PromptBuilder` base class

**Code Standards:**
- **No backward compatibility**: Use clean, modern interfaces without legacy support
- **Class-focused architecture**: Prefer class-based designs over functional approaches
- **No main runners in libraries**: Only CLI scripts have `if __name__ == "__main__"` blocks
- **Explicit interfaces**: Require all necessary arguments, avoid complex default logic
- **Shared UnitRegistry**: Always use `get_unit_registry()`, never create new registries

### Integration with Metadata Storage

This package integrates with a user-specified metadata storage directory:
- Reads API key from `.env` file (current directory)
- Writes extracted metadata to `<output-dir>/` directories (specified via `--output-dir`):
  - **Parameter estimates**: `to-review/parameter_estimates/{param_name}_{cancer_type}_deriv{num}.yaml`
  - **Test statistics**: `to-review/test_statistics/{test_stat_id}_{cancer_type}_deriv{num}.yaml`
- Output directory must exist (typically within your project repository, e.g., `metadata-storage/`)
- Sequential derivation numbering (deriv001, deriv002, etc.) enables multiple extractions per parameter

## Development

### Git Commands

**Important:** When running git commands in this repository, use plain `git` commands (not `git -C`):

```bash
# Correct
git status
git diff
git add .

# Incorrect (don't use -C flag)
git -C /path/to/repo status
```

### Running Tests

```bash
# Test package imports
python -c "from qsp_llm_workflows import PromptAssembler; print('✓ Import works')"

# Test CLI commands
qsp-extract --help
qsp-validate --help
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
        # ... perform validation ...
        report.add_pass(filename, message)
        return report.to_dict()
```

Then import and call in `run_all_validations.py`.

### Distribution

To publish the package to PyPI:

```bash
# Build package
python -m build

# Upload to PyPI
twine upload dist/*
```

Users can then install with:
```bash
pip install qsp-llm-workflows
```
