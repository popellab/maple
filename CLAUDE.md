# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### Workflow Examples

**Parameter Extraction:**
```bash
python scripts/prepare/create_parameter_batch.py input.csv
python scripts/run/upload_batch.py batch_jobs/parameter_requests.jsonl
python scripts/run/batch_monitor.py batch_<id>
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl \
  ../qsp-metadata-storage/parameter_estimates input.csv "" templates/parameter_metadata_template.yaml
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
  ../qsp-metadata-storage/test_statistics input.csv "" templates/test_statistic_template.yaml
# Aggregate distributions
python ../qspio-pdac/metadata/aggregate_test_statistics.py input.csv \
  ../qsp-metadata-storage/test_statistics ../qsp-metadata-storage/scratch/
```

### Script Organization

Scripts are organized by workflow stage:

**Prepare** (`scripts/prepare/`): Create batch requests
- `create_parameter_batch.py`: Parameter extraction batch requests
- `create_quick_estimate_batch.py`: Quick estimate batch requests
- `create_test_statistic_batch.py`: Test statistic batch requests
- `create_pooling_metadata_batch.py`: Pooling metadata batch requests
- `create_checklist_batch.py`, `create_schema_conversion_batch.py`: Other batch types

**Run** (`scripts/run/`): Execute batches
- `upload_batch.py`: Upload to OpenAI batch API (slower, handles large volumes)
- `upload_immediate.py`: Process via Responses API (faster feedback, testing)
- `batch_monitor.py`: Monitor batch progress and download results

**Process** (`scripts/process/`): Extract results
- `unpack_results.py`: Extract JSON from batch results, convert to YAML
- `unpack_single_json.py`: Process individual JSON responses

**Lib** (`scripts/lib/`): Core libraries
- `batch_creator.py`: Base classes for batch creation
- `parameter_utils.py`: Parameter processing utilities
- `prompt_assembly.py`: Modular prompt assembly engine

**Debug** (`scripts/debug/`): Debug and inspection tools
- `inspect_jsonl.py`: Examine batch request/response files
- `extract_prompt.py`: Extract prompts from batch requests
- `pretty_print_csv.py`: Format CSV output

**MATLAB** (`scripts/matlab/`): MATLAB integration
- `compute_test_statistic_from_yaml.m`: Test statistic computation
- `generate_calibration_target_from_yaml.m`: Calibration target generation
- `simple_test_harness.m`: Simple test harness

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
1. Input CSV with cancer_type and parameter_name columns
2. **Prompt assembly system** combines base prompts + templates + examples + parameter context data
3. Batch processing via OpenAI API creates structured YAML outputs
4. Results are unpacked directly to `../qsp-metadata-storage/parameter_estimates/` with filename format: `{param_name}_{author_year}_{cancer_type}_{hash}.yaml`

**Quick Estimate Workflow:**
1. Input CSV with cancer_type and parameter_name columns
2. Scripts generate quick estimate prompts for rapid parameter initialization
3. LLM generates estimates with ranges based on literature knowledge
4. Results are unpacked to `../qsp-metadata-storage/quick_estimates/` with format: `{param_name}_{cancer_type}_{hash}_deriv{N}.yaml`
5. Aggregation script pools estimates using lognormal statistics for positive-only parameters

**Test Statistics Workflow:**
1. Input CSV with test_statistic_id, scenario_context, required_species, and derived_species_description
2. Scripts generate prompts with model context and scenario information
3. LLM creates test statistic definitions with uncertainty quantification (R bootstrap code)
4. Results are unpacked to `../qsp-metadata-storage/test_statistics/` with format: `{test_stat_id}_{cancer_type}_{hash}.yaml`
5. Aggregation script pools distributions using inverse-variance weighting

### Key Data Files
- `data/simbio_parameters.csv`: Parameter definitions with Name, Units, Definition, References columns
- `data/model_context.csv`: Reaction context with Parameter, Reaction, ReactionRate, OtherParameters, OtherSpeciesWithNotes columns
- `templates/configs/prompt_assembly.yaml`: Configuration controlling how prompts are assembled
- `templates/parameter_metadata_template.yaml`: YAML template for parameter metadata
- `templates/prior_metadata_template.yaml`: YAML template for prior metadata generation
- `templates/examples/`: Example filled templates for different parameters

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
