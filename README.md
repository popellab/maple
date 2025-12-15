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

This gives you several CLI commands: `qsp-extract`, `qsp-validate`, `qsp-fix`, `qsp-enrich-csv`, `qsp-export-model`, and `qsp-batch-monitor`.

## Usage

The typical workflow is: prepare an input CSV describing what you need, run extraction, validate the results, and fix any issues.

```bash
# Extract parameter estimates from literature
qsp-extract params.csv --type parameter --output-dir metadata-storage --immediate

# Validate the outputs
qsp-validate parameter_estimates --dir metadata-storage/to-review/parameter_estimates

# Fix any validation failures
qsp-fix parameter_estimates --dir metadata-storage/to-review/parameter_estimates --immediate
```

The `--immediate` flag processes requests via the Responses API for faster turnaround. Without it, requests go through OpenAI's Batch API (cheaper but can take up to 24 hours).

For detailed setup instructions including API key configuration and input CSV format, see [docs/automated_workflow.md](docs/automated_workflow.md).

## What gets extracted

**Parameter estimates** include median values, interquartile ranges, and 95% confidence intervals derived from literature data. Each extraction includes the source papers, verbatim text snippets showing where values came from, and Python code that reproduces the statistical derivation.

**Test statistics** define validation targets for model outputs—things like "tumor volume at day 14 should be X ± Y mm³ based on clinical trial data." These are used to check whether model simulations match observed biology.

Both types go through a 9-validator suite that checks schema compliance, verifies DOIs resolve correctly, tests that derivation code runs, and confirms that text snippets actually contain the claimed values.

## Project structure

```
src/qsp_llm_workflows/
├── core/       # Prompt assembly, batch creation, validation utilities
├── prepare/    # CSV enrichment and batch request generation
├── run/        # API upload and monitoring
├── process/    # Result unpacking
├── validate/   # The 9 validators
├── cli/        # Command-line entry points
├── templates/  # YAML output templates
└── prompts/    # LLM instruction prompts
```

## Documentation

The [automated workflow guide](docs/automated_workflow.md) walks through first-time setup and basic usage. For package internals—how prompts are assembled, how validation works, how to add new validators—see [CLAUDE.md](CLAUDE.md).

## License

MIT
