# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note for end users:** If you're looking for setup and usage instructions, see [docs/automated_workflow.md](docs/automated_workflow.md) for a beginner-friendly guide. This file is for developers and contributors.

---

## Overview

This repository contains LLM workflow automation tools for extracting and validating quantitative systems pharmacology (QSP) metadata from scientific literature using OpenAI's batch API.

**Supported Workflows:**
- **Parameter extraction**: Extract parameter values, ranges, and statistical distributions with detailed literature tracking
- **Quick estimates**: Generate rapid parameter estimates for model initialization
- **Test statistics**: Create validation constraints from experimental data with uncertainty quantification
- **Pooling metadata**: Add statistical pooling information to existing extractions

All extracted metadata is stored in the central `qsp-metadata-storage` repository with flat file structures for easy access.

## Repository Organization

**This repository (`qsp-llm-workflows`):**
- General-purpose LLM workflow tools for parameter extraction
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

## Key Commands

### Python Environment Setup
**IMPORTANT:** Always activate the virtual environment before running Python scripts:

```bash
source venv/bin/activate
```

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
python scripts/export_model_definitions.py \
  --matlab-model ../qspio-pdac/immune_oncology_model_PDAC.m \
  --output batch_jobs/input_data/model_definitions.json
```

**Enrich with model definitions**:
```bash
# Enrich simple CSV with model definitions
python scripts/prepare/enrich_parameter_csv.py \
  simple_input.csv \
  batch_jobs/input_data/model_definitions.json \
  PDAC \
  -o batch_jobs/input_data/pdac_extraction_input.csv
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
python scripts/prepare/enrich_test_statistic_csv.py \
  partial_test_stats.csv \
  model_context.txt \
  baseline_no_treatment.yaml \
  -o batch_jobs/input_data/test_statistic_input.csv
```

This creates an enriched CSV with:
- `test_statistic_id`, `cancer_type`, `context_hash`
- `model_context`, `scenario_context`
- `required_species`, `derived_species_description`

**Note:** Model definitions are exported from MATLAB model files using `scripts/export_model_definitions.py`. Scenario context files are stored in model-specific repositories (e.g., `qspio-pdac/scenarios/`).

### Automated Workflow (Step 2)

**Single-command automated extraction** - Handles batch creation, upload, monitoring, unpacking, and git operations:

```bash
# Parameter extraction (complete pipeline)
python scripts/run_extraction_workflow.py input.csv --type parameter

# Test statistics
python scripts/run_extraction_workflow.py test_stats.csv --type test_statistic

# Quick estimates
python scripts/run_extraction_workflow.py quick.csv --type quick_estimate

# With custom timeout (default: 3600s)
python scripts/run_extraction_workflow.py input.csv --type parameter --timeout 7200

# Use immediate mode for faster processing (via Responses API)
python scripts/run_extraction_workflow.py input.csv --type parameter --immediate

# Create branch locally without pushing
python scripts/run_extraction_workflow.py input.csv --type parameter --no-push
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
python ../qsp-llm-workflows/scripts/validate/run_all_validations.py parameter_estimates
```

See `docs/automated_workflow.md` for complete documentation.

### Schema Conversion Workflow

**Automated schema conversion** - Scans for outdated schemas and converts them automatically:

```bash
# Scan for outdated files (dry run)
python scripts/run_schema_conversion.py --dry-run

# Convert all outdated files
python scripts/run_schema_conversion.py

# Convert only parameters
python scripts/run_schema_conversion.py --only parameter

# Convert only test statistics
python scripts/run_schema_conversion.py --only test_statistic
```

**Latest schema versions:**
- Parameters: v3 (templates/parameter_metadata_template.yaml)
- Test Statistics: v2 (templates/test_statistic_template.yaml)
- Quick Estimates: v1 (templates/quick_estimate_template.yaml)

**Note:** Only the latest templates are tracked in git. Old templates are retrieved from git history when needed for schema conversion.

The workflow automatically:
1. Scans metadata directories for files with outdated schema_version fields
2. Groups files by schema version transition (e.g., v1 → v3)
3. Creates batch requests for schema conversion
4. Monitors conversion progress
5. Unpacks converted files to to-review/ for verification
6. Creates review branch with converted files

### Validation Fix Workflow

**Automated validation fixing** - Sends failed YAMLs back to OpenAI for correction:

```bash
# Run validation first
python scripts/validate/run_all_validations.py test_statistics

# If failures detected, run fix workflow (automatic prompt, or run manually)
python scripts/run_validation_fix.py test_statistics --immediate

# For parameter estimates
python scripts/validate/run_all_validations.py parameter_estimates
python scripts/run_validation_fix.py parameter_estimates --immediate

# With custom timeout for Batch API (default: 3600s)
python scripts/run_validation_fix.py test_statistics --timeout 7200
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
- After fixes complete, re-run `run_all_validations.py` to verify
- Manual review recommended for persistent failures
- **Use `--immediate` flag for faster processing** (good for testing, batch API can take up to 24 hours)

**Validation types that can be fixed:**
- Schema compliance (missing fields, wrong types)
- Source references (missing source_ref fields)
- Code execution (debugging R/Python code)
- Text snippets (verifying value_snippet contains values)
- DOI resolution (fixing malformed DOIs)

### Validation Suite

The automated validation suite (`run_all_validations.py`) includes 8 validators:

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

### Manual Workflow (For Reference)

For fine-grained control, you can run individual steps:

**Parameter Extraction:**
```bash
python scripts/prepare/create_parameter_batch.py input.csv
python scripts/run/upload_batch.py batch_jobs/parameter_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/parameter_estimates input.csv "" templates/parameter_metadata_template_v3.yaml
```

**Quick Estimates:**
```bash
python scripts/prepare/create_quick_estimate_batch.py input.csv
python scripts/run/upload_batch.py batch_jobs/quick_estimate_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/quick_estimates input.csv
# Aggregate results
python ../qspio-pdac/metadata/aggregate_quick_estimates.py input.csv \
  ../qsp-metadata-storage/quick_estimates output/
```

**Test Statistics:**
```bash
python scripts/prepare/create_test_statistic_batch.py input.csv
python scripts/run/upload_batch.py batch_jobs/test_statistic_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/test_statistics input.csv "" templates/test_statistic_template_v2.yaml
# Aggregate distributions
python ../qspio-pdac/metadata/aggregate_test_statistics.py input.csv \
  ../qsp-metadata-storage/test_statistics ../qsp-metadata-storage/scratch/
```

### Script Organization

Scripts are organized by workflow stage:

**Prepare** (`scripts/prepare/`): CSV enrichment and batch request creation
- **CSV Enrichment (Step 1):**
  - `enrich_parameter_csv.py`: Enrich simple parameter CSV with model definitions
  - `enrich_test_statistic_csv.py`: Enrich partial test statistic CSV with context
- **Batch Creation (Step 2):**
  - `create_parameter_batch.py`: Parameter extraction batch requests
  - `create_quick_estimate_batch.py`: Quick estimate batch requests
  - `create_test_statistic_batch.py`: Test statistic batch requests
  - `create_pooling_metadata_batch.py`: Pooling metadata batch requests
  - `create_checklist_batch.py`, `create_schema_conversion_batch.py`: Other batch types

**Run** (`scripts/run/`): Execute batches
- `upload_batch.py`: Upload to OpenAI batch API (slower, handles large volumes)
- `upload_immediate.py`: Process via Responses API (faster feedback, testing)
- `batch_monitor.py`: Monitor batch progress and download results

**Automated Workflows**:
- `run_extraction_workflow.py`: Complete automated extraction pipeline (create → upload → monitor → validate → unpack → git commit/push)
- `run_schema_conversion.py`: Automated schema conversion for outdated metadata files

**Process** (`scripts/process/`): Extract results
- `unpack_results.py`: Extract JSON from batch results, convert to YAML
- `unpack_single_json.py`: Process individual JSON responses

**Lib** (`scripts/lib/`): Core libraries
- `batch_creator.py`: Base classes for batch creation
- `parameter_utils.py`: Parameter processing utilities
- `workflow_orchestrator.py`: Automated workflow orchestration
- `prompt_assembly.py`: Modular prompt assembly engine
- `schema_version_detector.py`: Schema version detection and file scanning

**Debug** (`scripts/debug/`): Debug and inspection tools
- `inspect_jsonl.py`: Examine batch request/response files
- `extract_prompt.py`: Extract prompts from batch requests
- `pretty_print_csv.py`: Format CSV output

**Manuscript** (`docs-manuscript/`): Paper collaboration materials (gitignored)
- `COLLABORATOR_ONBOARDING.md`: Comprehensive onboarding guide for paper collaborators
- `presentation.tex`: Beamer presentation introducing the project
- `paper_outline_standardization.md`: Complete paper outline
- Note: These materials are shared via email, not checked into repository

## Architecture

### Modular Prompt Assembly System
This repository uses a generalized prompt assembly system that builds prompts from modular components:

```
prompts/                         # Base prompt files with placeholders
├── parameter_prompt.md
├── quick_estimate_prompt.md
├── test_statistic_prompt.md
└── suggest_test_statistics_prompt.md
templates/                       # YAML templates and examples
├── configs/prompt_assembly.yaml # Configuration for prompt assembly
├── parameter_metadata_template.yaml (v1 & v2)
├── quick_estimate_template.yaml
├── test_statistic_template.yaml
├── prior_metadata_template.yaml
└── examples/                    # Example filled templates
scripts/
├── lib/                         # Core libraries
│   ├── prompt_assembly.py      # Prompt assembly engine
│   ├── batch_creator.py        # Base classes for batch creation
│   └── parameter_utils.py      # Parameter processing utilities
├── prepare/                     # Batch creation scripts
├── run/                         # Batch execution scripts
└── process/                     # Result processing scripts
```

### Data Flow

**Parameter Extraction Workflow:**
1. **CSV Enrichment** (Step 1): Simple CSV (parameter names) + model definitions JSON → enriched CSV with units, descriptions, model context
2. **Batch Creation**: Enriched CSV with all required fields (cancer_type, parameter_name, parameter_units, parameter_description, model_context, definition_hash)
3. **Prompt Assembly**: System combines base prompts + templates + examples + parameter context data
4. **LLM Processing**: Batch processing via OpenAI API creates structured YAML outputs
5. **Unpacking**: Results unpacked to `../qsp-metadata-storage/parameter_estimates/` with format: `{param_name}_{author_year}_{cancer_type}_{hash}.yaml`

**Quick Estimate Workflow:**
1. **CSV Enrichment** (Step 1): Simple CSV (parameter names) + model definitions JSON → enriched CSV
2. **Batch Creation**: Scripts generate quick estimate prompts for rapid parameter initialization
3. **LLM Processing**: LLM generates estimates with ranges based on literature knowledge
4. **Unpacking**: Results unpacked to `../qsp-metadata-storage/quick_estimates/` with format: `{param_name}_{cancer_type}_{hash}_deriv{N}.yaml`
5. **Aggregation**: Script pools estimates using lognormal statistics for positive-only parameters

**Test Statistics Workflow:**
1. **CSV Enrichment** (Step 1): Partial CSV (test_statistic_id, required_species, derived_species_description) + model_context.txt + scenario YAML → enriched CSV
2. **Batch Creation**: Scripts generate prompts with model context and scenario information
3. **LLM Processing**: LLM creates test statistic definitions with uncertainty quantification (R bootstrap code)
4. **Unpacking**: Results unpacked to `../qsp-metadata-storage/test_statistics/` with format: `{test_stat_id}_{cancer_type}_{hash}.yaml`
5. **Aggregation**: Script pools distributions using inverse-variance weighting

### Key Files and Directories

**Templates and Configuration:**
- `templates/configs/prompt_assembly.yaml`: Configuration controlling how prompts are assembled
- `templates/parameter_metadata_template.yaml`: YAML template for parameter metadata
- `templates/test_statistic_template.yaml`: YAML template for test statistics
- `templates/quick_estimate_template.yaml`: YAML template for quick estimates
- `templates/examples/`: Example filled templates for different parameters

**Prompts:**
- `prompts/parameter_prompt.md`: Base prompt for parameter extraction
- `prompts/test_statistic_prompt.md`: Base prompt for test statistics
- `prompts/quick_estimate_prompt.md`: Base prompt for quick estimates

**CSV Enrichment Scripts:**
- `scripts/prepare/enrich_parameter_csv.py`: Enrich simple parameter CSV with model definitions
- `scripts/prepare/enrich_test_statistic_csv.py`: Enrich partial test statistic CSV with context

**Note:** Model definitions and context files are exported from model-specific repositories (e.g., `qspio-pdac`). This repository provides general-purpose workflow tools that work with any model system.

### Class-based Batch Creation Architecture
Batch creation uses a modular class-based system:

- `scripts/lib/batch_creator.py`: Base `BatchCreator` class with common functionality
- `ParameterBatchCreator`: For parameter extraction requests (uses prompt assembly system)
- `PoolingMetadataBatchCreator`: For adding statistical metadata to existing YAMLs
- CLI scripts in `scripts/prepare/` provide simple interfaces to batch creators

### Script Dependencies
- `scripts/lib/parameter_utils.py`: Utilities for parameter processing (CSV loading, model context generation)
- `scripts/lib/prompt_assembly.py`: Modular prompt assembly from templates and examples
- `scripts/lib/batch_creator.py`: Class-based batch creation with shared functionality
- All API scripts expect `OPENAI_API_KEY` in `.env` file (current directory)
- `scripts/process/unpack_results.py` writes directly to `../qsp-metadata-storage/` directories with flat structure
- `scripts/prepare/create_pooling_metadata_batch.py` reads from `../qsp-metadata-storage/parameter_estimates/`

### Batch Processing Model
- Uses OpenAI's batch API with GPT-5 model and high reasoning effort
- Custom IDs follow format: `{cancer_type}_{parameter_name}_{index}`
- Results are saved to `batch_jobs/` directory (gitignored)
- Batch IDs are tracked in `.batch_id` files alongside JSONL files

### Architecture Benefits
- **Modular Prompts**: Templates and examples are reusable across different prompt types
- **Class-based Batching**: Common batch functionality shared via inheritance
- **Maintainable**: Changes to templates or batch logic only need to be made once
- **Extensible**: New prompt types and batch creators can be added easily
- **Flexible**: Components can be mixed and matched for different use cases

## Integration Points

## Integration with Parameter Storage

This repository integrates with the central metadata storage system:
- Reads API key from `.env` file (current directory)
- Writes extracted metadata to different directories based on workflow type:
  - **Parameter estimates**: `../qsp-metadata-storage/parameter_estimates/{param_name}_{author_year}_{cancer_type}_{hash}.yaml`
    - Hash computed from study context to enable multiple extractions per parameter
  - **Quick estimates**: `../qsp-metadata-storage/quick_estimates/{param_name}_{cancer_type}_{hash}_deriv{N}.yaml`
  - **Test statistics**: `../qsp-metadata-storage/test_statistics/{test_stat_id}_{cancer_type}_{hash}.yaml`
    - Hash computed from scenario context
- Assumes `qsp-metadata-storage` repository exists as sibling directory
- Aggregation scripts in `qspio-pdac/metadata/` pool results from multiple sources

## Standard Usage

- `scripts/process/unpack_results.py`: Extracts directly to `../qsp-metadata-storage/` directories with flat structure
- `scripts/prepare/create_pooling_metadata_batch.py`: Reads from `../qsp-metadata-storage/parameter_estimates/`

Example usage:
```bash
python scripts/process/unpack_results.py batch_results.jsonl ../qsp-metadata-storage/parameter_estimates input.csv
python scripts/prepare/create_pooling_metadata_batch.py ../qsp-metadata-storage/parameter_estimates
```

# Important Instructions

## Code Standards
- **No backward compatibility**: Use clean, modern interfaces without legacy support
- **Class-focused architecture**: Prefer class-based designs over functional approaches  
- **No main runners in libraries**: Only CLI scripts should have `if __name__ == "__main__":` blocks. Never add them to class files, utility modules, or library code
- **Explicit interfaces**: Require all necessary arguments, avoid complex default logic
