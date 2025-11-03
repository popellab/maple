# QSP LLM Workflows

This repository contains scripts and workflows for automated parameter extraction from scientific literature using Large Language Models.

## Overview

This toolkit automates the extraction and validation of quantitative systems pharmacology (QSP) metadata from scientific literature using OpenAI's batch API. It supports multiple workflow types:

- **Parameter extraction**: Extract parameter values, ranges, and statistical distributions
- **Quick estimates**: Generate rapid parameter estimates for model initialization
- **Test statistics**: Create validation constraints from experimental data
- **Pooling metadata**: Add statistical pooling information to existing extractions

All extracted metadata is stored in the central `qsp-metadata-storage` repository.

## Repository Organization

**This repository (`qsp-llm-workflows`):**
- General-purpose LLM workflow tools for parameter extraction
- Reusable across any QSP model or disease area
- Focus: Core extraction, validation, and storage workflows

**Paper repository (`qsp-llm-workflows-paper`, to be created):**
- Paper-specific code, validation analyses, and manuscript figures
- Validation study comparing LLM extraction to legacy parameter database
- Reproducible research for publication

**Manuscript documentation:**
- Collaboration materials stored in `docs-manuscript/` (gitignored)
- Shared via email, not checked into repository
- Includes onboarding guide, presentation, and paper outline

## Directory Structure

```
├── scripts/           # Core automation scripts
│   ├── prepare/              # Batch creation scripts
│   │   ├── create_parameter_batch.py
│   │   ├── create_quick_estimate_batch.py
│   │   ├── create_test_statistic_batch.py
│   │   ├── create_checklist_batch.py
│   │   ├── create_pooling_metadata_batch.py
│   │   └── create_schema_conversion_batch.py
│   ├── run/                  # Batch execution scripts
│   │   ├── upload_batch.py
│   │   ├── upload_immediate.py
│   │   └── batch_monitor.py
│   ├── process/              # Results processing scripts
│   │   ├── unpack_results.py
│   │   └── unpack_single_json.py
│   ├── lib/                  # Core libraries
│   │   ├── batch_creator.py
│   │   ├── parameter_utils.py
│   │   └── prompt_assembly.py
│   ├── debug/                # Debug and inspection tools
│   │   ├── inspect_jsonl.py
│   │   ├── extract_prompt.py
│   │   └── pretty_print_csv.py
│   └── batch_workflow_commands.sh
├── prompts/          # Base prompt files with placeholders
│   ├── parameter_prompt.md
│   ├── quick_estimate_prompt.md
│   ├── test_statistic_prompt.md
│   └── suggest_test_statistics_prompt.md
├── templates/        # YAML templates and examples
│   ├── configs/              # Prompt assembly configuration
│   ├── parameter_metadata_template.yaml (v1 & v2)
│   ├── quick_estimate_template.yaml
│   ├── test_statistic_template.yaml
│   ├── prior_metadata_template.yaml
│   └── examples/             # Example filled templates
├── data/            # Reference data and examples
├── examples/        # Example workflows and outputs
├── batch_jobs/      # Batch processing files (gitignored)
└── docs-manuscript/ # Paper collaboration materials (gitignored)
```

## Workflow

1. **Create batch requests** from parameter CSV files
2. **Upload** to OpenAI batch API for processing
3. **Monitor** batch completion status
4. **Extract results** to YAML parameter files
5. **Review** extracted parameters in the central parameter storage

## Usage

The toolkit uses a **modular prompt assembly system** that builds prompts from reusable components:

### Parameter Extraction Workflow
```bash
# Create parameter extraction batch requests
python scripts/prepare/create_parameter_batch.py input.csv

# Upload and process
python scripts/run/upload_batch.py batch_jobs/parameter_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>

# Extract results to metadata storage
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/parameter_estimates input.csv "" templates/parameter_metadata_template.yaml
```

### Quick Estimates Workflow
```bash
# Create quick estimate batch requests
python scripts/prepare/create_quick_estimate_batch.py input.csv

# Upload and process (same as above)
python scripts/run/upload_batch.py batch_jobs/quick_estimate_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>

# Extract results
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/quick_estimates input.csv

# Aggregate estimates (in qspio-pdac)
python ../qspio-pdac/metadata/aggregate_quick_estimates.py input.csv \
  ../qsp-metadata-storage/quick_estimates output/
```

### Test Statistics Workflow
```bash
# Create test statistic batch requests
python scripts/prepare/create_test_statistic_batch.py input.csv

# Upload and process (same as above)
python scripts/run/upload_batch.py batch_jobs/test_statistic_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>

# Extract results
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/test_statistics input.csv "" templates/test_statistic_template.yaml

# Aggregate distributions (in qspio-pdac)
python ../qspio-pdac/metadata/aggregate_test_statistics.py input.csv \
  ../qsp-metadata-storage/test_statistics ../qsp-metadata-storage/scratch/
```

The complete workflow is documented in `scripts/batch_workflow_commands.sh`.

## Integration

This repository is designed to work with the central `qsp-metadata-storage` repository. Different workflow types write to different directories:

- **Parameter estimates**: `../qsp-metadata-storage/parameter_estimates/{param_name}_{author_year}_{cancer_type}_{hash}.yaml`
- **Quick estimates**: `../qsp-metadata-storage/quick_estimates/{param_name}_{cancer_type}_{hash}_deriv{N}.yaml`
- **Test statistics**: `../qsp-metadata-storage/test_statistics/{test_stat_id}_{cancer_type}_{hash}.yaml`

Aggregation scripts in `qspio-pdac/metadata/` pool results from multiple sources for model initialization and validation.

## Modular Prompt System

The toolkit features a **generalized prompt assembly system** that:
- **Separates concerns**: Base prompts, templates, examples, and runtime data are modular
- **Enables reusability**: Templates and examples can be shared across different prompt types
- **Simplifies maintenance**: Changes to templates only need to be made once
- **Supports extensibility**: New prompt types can be added through configuration

### Prompt Components
- **Base prompts** (`prompts/`): Core instructions with placeholder markers
- **Templates** (`templates/`): YAML templates and configuration files
- **Examples** (`templates/examples/`): Example filled templates
- **Assembly engine** (`scripts/lib/prompt_assembly.py`): Combines components into final prompts

## Architecture

### Class-based Batch Creation
The system uses a modular class-based architecture:

- **`BatchCreator`** (base class): Common functionality for all batch types
- **`ParameterBatchCreator`**: Handles parameter extraction from literature
- **`PoolingMetadataBatchCreator`**: Adds statistical metadata to existing studies
- **CLI scripts**: Simple interfaces to the batch creator classes

### Parameter Processing
- **`scripts/lib/parameter_utils.py`**: Utilities for loading and processing parameter data
- **`scripts/lib/prompt_assembly.py`**: Modular system for building prompts from components
- **Templates and examples**: Reusable prompt components in `templates/`

### Integration with Parameter Storage
The workflow tools target the central parameter storage repository:

```bash
# Extract parameters to central storage
python scripts/process/unpack_results.py batch_results.jsonl ../qsp-metadata-storage/parameter_estimates input.csv

# Process pooling metadata from central storage
python scripts/prepare/create_pooling_metadata_batch.py ../qsp-metadata-storage/parameter_estimates
```

This supports the three-tier QSP architecture where individual projects reference parameters from the central storage rather than storing duplicate copies.

## Validation Suite

The repository includes a comprehensive validation suite (`scripts/validate/run_all_validations.py`) that ensures metadata quality through 7 automated and interactive checks:

### Validation Checks

1. **Schema Compliance** - Verifies YAML structure matches template specification
2. **Code Execution** - Tests that R/Python code runs without errors and produces expected outputs
3. **Text Snippets** - Validates that value_snippet and units_snippet contain declared values
4. **Source References** - Ensures all source_ref fields point to defined sources
5. **DOI Validity** - Verifies DOIs resolve correctly and metadata matches
6. **Value Consistency** - Checks values are consistent across related extractions
7. **Manual Snippet Source Verification** - Interactive verification that snippets appear in papers

### Running Validation

```bash
# Run complete validation suite
python scripts/validate/run_all_validations.py test_statistics
python scripts/validate/run_all_validations.py parameter_estimates

# Results written to output/validation_results/
# - Individual JSON reports for each validator
# - master_validation_summary.json with aggregate results
# - snippet_sources.json with verification status
```

### Manual Snippet Verification

During validation, the suite generates a report listing all sources with:
- DOI links (clickable in terminal)
- Unique text snippets to verify
- Input names where snippets are used

**Workflow:**
1. Report prints to console during validation
2. User clicks DOI links to open papers
3. User searches for snippets in papers using Ctrl+F/Cmd+F
4. User confirms verification (y/n) when prompted
5. Suite tags files if verification passes

### Validation Tagging

After all checks complete, the suite automatically tags YAML files with validation results:

```yaml
# Validation metadata (appended to end of each file)
validation:
  tags:
    - template_compliance_validation
    - code_execution_testing
    - manual_snippet_source_verification
  validated_at: '2025-11-03T10:30:00'
```

Features:
- Preserves original file formatting and comments
- Only tags files for passed validation checks
- Enables tracking of validation status over time
- Supports re-validation (replaces existing tags)

### Validation Fix Workflow

Failed validations can be automatically corrected by re-submitting to OpenAI:

```bash
# Integrated into validation suite (prompts after failures)
python scripts/run_validation_fix.py test_statistics --immediate

# Workflow:
# 1. Loads validation errors from JSON reports
# 2. Creates fix batch requests (YAML + errors + template)
# 3. Submits to OpenAI API for correction
# 4. Unpacks fixed YAMLs (overwrites originals)
# 5. Prompts to re-run validation
```

### Optional: Automated Snippet Verification

For fully automated snippet verification (instead of manual):

```bash
# Configure environment variables in .env
VALIDATION_EMAIL=your.email@jhu.edu

# Optional: Institutional proxy for paywalled papers
HOPKINS_PROXY_URL=https://proxy1.library.jhu.edu/login?url=
HOPKINS_PROXY_COOKIES={"session": "xxx", "auth_token": "yyy"}

# Run standalone automated verifier
python scripts/validate/check_snippet_sources.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/snippet_sources.json \
  --email your.email@jhu.edu
```

The automated verifier fetches full text from Unpaywall API, Europe PMC, or institutional proxy and uses fuzzy matching to verify snippets. Coverage: ~50-60% of papers.
