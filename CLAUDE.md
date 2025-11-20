# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note for end users:** If you're looking for setup and usage instructions, see [docs/automated_workflow.md](docs/automated_workflow.md) for a beginner-friendly guide. This file is for developers and contributors.

---

## Overview

This repository contains LLM workflow automation tools for extracting and validating quantitative systems pharmacology (QSP) metadata from scientific literature using OpenAI's batch API.

**Supported Workflows:**
- **Parameter extraction**: Extract parameter values, ranges, and statistical distributions with detailed literature tracking
- **Test statistics**: Create validation constraints from experimental data with uncertainty quantification

All extracted metadata is stored in the central `qsp-metadata-storage` repository with flat file structures for easy access.

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

**Export model definitions** (from model MATLAB file):
```bash
# Export from MATLAB model to JSON
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/your_model_file.m \
  --output batch_jobs/input_data/model_definitions.json
```

**Enrich with model definitions**:
```bash
# Enrich simple CSV with model definitions
qsp-enrich-csv parameter \
  simple_input.csv \
  batch_jobs/input_data/model_definitions.json \
  YOUR_CANCER_TYPE \
  -o batch_jobs/input_data/enriched_extraction_input.csv
```

This creates an enriched CSV with:
- `cancer_type`, `parameter_name`, `definition_hash`
- `parameter_units`, `parameter_description`
- `model_context` (JSON with reactions, rules, related parameters)

#### Test Statistic CSV

Partial input (test statistics without context):
```csv
test_statistic_id,required_species,derived_species_description
tumor_volume_day14,V_T.C,Tumor volume in mm³ at 14 days
```

Enrich with model and scenario context:
```bash
# Enrich partial CSV with model and scenario context
qsp-enrich-csv test_statistic \
  partial_test_stats.csv \
  model_context.txt \
  baseline_no_treatment.yaml \
  -o batch_jobs/input_data/test_statistic_input.csv
```

This creates an enriched CSV with:
- `test_statistic_id`, `cancer_type`, `context_hash`
- `model_context`, `scenario_context`
- `required_species`, `derived_species_description`

**Note:** Model definitions are exported from MATLAB model files. Scenario context files are stored in model-specific repositories (e.g., `your-model-repo/scenarios/`).

### Automated Workflow (Step 2)

**Single-command automated extraction** - Handles batch creation, upload, monitoring, unpacking, and git operations:

```bash
# Parameter extraction (complete pipeline)
qsp-extract input.csv --type parameter

# Test statistics
qsp-extract test_stats.csv --type test_statistic

# With custom timeout (default: 3600s)
qsp-extract input.csv --type parameter --timeout 7200

# Use immediate mode for faster processing (via Responses API)
qsp-extract input.csv --type parameter --immediate

# Create branch locally without pushing
qsp-extract input.csv --type parameter --no-push
```

**What the automated workflow does:**
1. Creates batch requests
2. Uploads to OpenAI API
3. Polls until completion (shows progress)
4. Unpacks results to `../qsp-metadata-storage/to-review/`
5. Creates review branch and pushes to remote
6. Prints summary with next steps

**After workflow completes, manually run validation:**
```bash
cd ../qsp-metadata-storage
git checkout review/batch-parameter-2025-10-27-abc123
qsp-validate parameter_estimates
```

See `docs/automated_workflow.md` for complete documentation.

### Validation Fix Workflow

**Automated validation fixing** - Sends failed YAMLs back to OpenAI for correction:

```bash
# Run validation first
qsp-validate test_statistics

# If failures detected, run fix workflow
qsp-fix test_statistics --immediate

# For parameter estimates
qsp-validate parameter_estimates
qsp-fix parameter_estimates --immediate

# With custom timeout for Batch API (default: 3600s)
qsp-fix test_statistics --timeout 7200
```

**What the validation fix workflow does:**
1. Loads validation JSON reports and aggregates errors by file
2. Creates fix batch requests with YAMLs + error lists + template
3. Uploads to OpenAI API
4. Monitors until completion
5. Unpacks fixed YAMLs (overwrites originals in to-review/)
6. Prompts to re-run validation

**Important notes:**
- Original files are backed up in git history before overwriting
- Single fix attempt per run (prevents wasting API calls on unfixable errors)
- After fixes complete, re-run `qsp-validate` to verify
- Manual review recommended for persistent failures
- **Use `--immediate` flag for faster processing** (good for testing, batch API can take up to 24 hours)

**Validation types that can be fixed:**
- Schema compliance (missing fields, wrong types)
- Source references (missing source_ref fields)
- Code execution (debugging R/Python code)
- Text snippets (verifying value_snippet contains values)
- DOI resolution (fixing malformed DOIs)

### Validation Suite

The automated validation suite includes 8 validators:

1. **Schema Compliance** - YAML structure matches template
2. **Code Execution** - R/Python code runs without errors
3. **Text Snippets** - Snippets contain declared values
4. **Source References** - All source_refs point to defined sources
5. **DOI Validity** - DOIs resolve and metadata matches
6. **Value Consistency** - Values consistent across related extractions
7. **Duplicate Primary Sources** - Primary data sources not already used in accepted extractions
8. **Manual Snippet Source Verification** - Interactive verification of snippets in papers

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

### Other CLI Commands

```bash
# Monitor batch job progress
qsp-batch-monitor batch_abc123

# Export model definitions
qsp-export-model --matlab-model model.m --output defs.json
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
│       │   ├── batch_creator.py     # Base classes for batch creation
│       │   ├── prompt_assembly.py   # Modular prompt assembly engine
│       │   ├── workflow_orchestrator.py  # Workflow automation
│       │   ├── parameter_utils.py   # Parameter processing
│       │   ├── model_definition_exporter.py
│       │   ├── validation_utils.py  # Validation utilities
│       │   ├── resource_utils.py    # Package resource access
│       │   ├── header_utils.py      # Header field management
│       │   ├── hash_utils.py        # Hashing utilities
│       │   └── schema_version_detector.py
│       │
│       ├── prepare/                  # Batch preparation
│       │   ├── create_parameter_batch.py
│       │   ├── create_test_statistic_batch.py
│       │   ├── enrich_parameter_csv.py
│       │   └── enrich_test_statistic_csv.py
│       │
│       ├── run/                      # Batch execution
│       │   ├── upload_batch.py
│       │   ├── upload_immediate.py
│       │   └── batch_monitor.py
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
│       │   ├── fix.py               # qsp-fix
│       │   ├── enrich.py            # qsp-enrich-csv
│       │   ├── export_model.py      # qsp-export-model
│       │   └── monitor.py           # qsp-batch-monitor
│       │
│       ├── templates/                # YAML templates (package data)
│       │   ├── parameter_metadata_template.yaml
│       │   ├── test_statistic_template.yaml
│       │   ├── configs/
│       │   │   ├── prompt_assembly.yaml
│       │   │   └── header_fields.yaml
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
- **Core** (`core/`): Core libraries (batch creation, prompt assembly, workflow orchestration)

### Data Flow

**Parameter Extraction Workflow:**
1. **CSV Enrichment**: Simple CSV (parameter names) + model definitions JSON → enriched CSV
2. **Batch Creation**: System generates batch requests with prompts and context
3. **LLM Processing**: OpenAI API processes requests and creates structured YAML outputs
4. **Unpacking**: Results unpacked to `../qsp-metadata-storage/parameter_estimates/`
5. **Validation**: Automated validation suite checks quality and completeness

**Test Statistics Workflow:**
1. **CSV Enrichment**: Partial CSV + model context + scenario YAML → enriched CSV
2. **Batch Creation**: System generates prompts with full context
3. **LLM Processing**: LLM creates test statistic definitions with R bootstrap code
4. **Unpacking**: Results unpacked to `../qsp-metadata-storage/test_statistics/`
5. **Aggregation**: Distributions pooled using inverse-variance weighting

### Key Design Principles

**Package Architecture:**
- **Installable**: `pip install -e .` for development, `pip install` for distribution
- **CLI-first**: Commands like `qsp-extract` available system-wide after install
- **Resource Management**: Uses `importlib.resources` for robust template/prompt access
- **No sys.path manipulation**: Clean imports throughout
- **Library code is library code**: Validation scripts are imported, not called via subprocess
- **Class-based**: Batch creators inherit from `BatchCreator` base class

**Code Standards:**
- **No backward compatibility**: Use clean, modern interfaces without legacy support
- **Class-focused architecture**: Prefer class-based designs over functional approaches
- **No main runners in libraries**: Only CLI scripts have `if __name__ == "__main__"` blocks
- **Explicit interfaces**: Require all necessary arguments, avoid complex default logic

### Integration with Metadata Storage

This package integrates with the central metadata storage system:
- Reads API key from `.env` file (current directory)
- Writes extracted metadata to `../qsp-metadata-storage/` directories:
  - **Parameter estimates**: `parameter_estimates/{param_name}_{author_year}_{cancer_type}_{hash}.yaml`
  - **Test statistics**: `test_statistics/{test_stat_id}_{cancer_type}_{hash}.yaml`
- Assumes `qsp-metadata-storage` repository exists as sibling directory
- Hash-based filenames enable multiple extractions per parameter

## Development

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
