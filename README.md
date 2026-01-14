# QSP LLM Workflows

[![Tests](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml/badge.svg)](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml)

Extract parameter values and test statistics from scientific literature for quantitative systems pharmacology (QSP) models. This package uses OpenAI's API to read papers, pull out numeric values with uncertainty estimates, and generate reproducible Python code that derives distributions suitable for Bayesian model calibration.

## Installation

```bash
git clone https://github.com/popellab/qsp-llm-workflows.git
cd qsp-llm-workflows
python -m venv venv
source venv/bin/activate
pip install -e .
```

This gives you several CLI commands: `qsp-extract`, `qsp-validate`, `qsp-enrich-csv`, and `qsp-export-model`.

## Usage

The typical workflow is: prepare an input CSV describing what you need, run extraction, and validate the results.

```bash
# Extract parameter estimates from literature
qsp-extract params.csv --type parameter --output-dir metadata-storage

# Validate the outputs
qsp-validate parameter_estimates --dir metadata-storage/to-review/parameter_estimates
```

Extraction uses Pydantic AI to process requests via OpenAI's API.

For detailed setup instructions including API key configuration and input CSV format, see [docs/automated_workflow.md](docs/automated_workflow.md).

## What gets extracted

**Parameter estimates** include median values, interquartile ranges, and 95% confidence intervals derived from literature data. Each extraction includes the source papers, verbatim text snippets showing where values came from, and Python code that reproduces the statistical derivation.

**Test statistics** define validation targets for model outputs—things like "tumor volume at day 14 should be X ± Y mm³ based on clinical trial data." These are used to check whether model simulations match observed biology.

Both types go through a 9-validator suite that checks schema compliance, verifies DOIs resolve correctly, tests that derivation code runs, and confirms that text snippets actually contain the claimed values.

## Project structure

```
src/qsp_llm_workflows/
├── core/       # Prompt generation, workflow orchestration, validation utilities
│   └── calibration/  # Calibration target models
│       ├── calibration_target_models.py  # CalibrationTarget, CalibrationTargetEstimates
│       ├── isolated_system_target.py     # IsolatedSystemTarget for in vitro data
│       ├── observable.py                 # Observable, Submodel, SubmodelObservable
│       ├── shared_models.py              # EstimateInput, SubmodelInput, Source, Snippet
│       ├── enums.py                      # Species, Indication, Compartment, System, SourceType, ExtractionMethod
│       ├── scenario.py                   # Intervention, Scenario
│       └── code_validator.py             # Unified code validation (CodeValidator)
├── prepare/    # CSV enrichment and prompt generation
├── run/        # API request processing
├── process/    # Result unpacking
├── validate/   # The 9 validators
├── cli/        # Command-line entry points
├── templates/  # YAML output templates
└── prompts/    # LLM instruction prompts
```

**Calibration target types:**
- `CalibrationTarget`: For clinical/in vivo data. Uses `observable` to compute measurements from full model species.
- `IsolatedSystemTarget`: For in vitro/preclinical data. Uses `submodel` to define a standalone ODE that shares parameter names with the full model for joint inference. State variables are self-contained with initial values and provenance. Observable code is optional—if omitted, defaults to the first state variable with units.

**Input architecture:** Inputs are co-located with the code blocks that use them. `submodel.inputs` holds experimental conditions for ODE code, `observable.inputs` holds values for observable code, and `calibration_target_estimates.inputs` holds values for distribution derivation. State variables include their initial values directly (no reference indirection). Inputs support figure-extracted data via `source_type` field (`text`, `table`, or `figure`) with required `figure_id` and `extraction_method` for figure sources.

**Vector-valued data:** Calibration targets support both scalar and time-course/dose-response data through a unified pathway. Scalar data uses length-1 lists; vector data uses `index_values` to specify the indexing dimension (time points, doses, etc.).

## Documentation

The [automated workflow guide](docs/automated_workflow.md) walks through first-time setup and basic usage. For package internals—how prompts are assembled, how validation works, how to add new validators—see [CLAUDE.md](CLAUDE.md).

## License

MIT
