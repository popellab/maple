# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains LLM workflow automation tools for extracting quantitative systems pharmacology (QSP) parameters from scientific literature using OpenAI's batch API. The tools are designed to extract parameters to the central `qsp-metadata-storage` repository.

## Key Commands

### Python Environment Setup
**IMPORTANT:** Always activate the virtual environment before running Python scripts:

```bash
source venv/bin/activate
```

### Core Batch Workflow
The main workflow commands are documented in `scripts/batch_workflow_commands.sh`:

```bash
# Create parameter extraction batch requests from CSV
python scripts/create_parameter_batch.py input.csv params.csv reactions.csv

# Upload to OpenAI batch API
python scripts/upload_batch.py batch_jobs/batch_requests.jsonl

# Monitor batch progress and download when complete
python scripts/batch_monitor.py batch_<id>

# Unpack results to YAML files in central metadata storage
python scripts/unpack_results.py batch_jobs/batch_<id>_results.jsonl ../qsp-metadata-storage/parameter_estimates
```

### Individual Script Usage

- `create_parameter_definition_batch.py`: Creates parameter definition batch requests. Requires input.csv, params.csv, and reactions.csv arguments
- `create_parameter_batch.py`: Creates parameter extraction batch requests. Requires only input.csv (uses stored parameter definitions)
- `create_pooling_metadata_batch.py`: Creates batch requests to add pooling metadata to existing study YAML files
- `upload_batch.py`: Uploads to OpenAI batch API (slower, handles large volumes). Requires JSONL file path, expects OpenAI API key in `.env` file
- `upload_immediate.py`: Processes requests immediately via Responses API (faster feedback, good for testing). Requires JSONL file path, expects OpenAI API key in `.env` file
- `batch_monitor.py`: Requires batch ID, automatically downloads results when batch is complete
- `unpack_results.py`: Extracts YAML from batch results to `qsp-metadata-storage/parameter_estimates/to-review/` directory structure
- `inspect_jsonl.py`: Debug utility for examining batch request/response files
- `extract_prompt.py`: Extracts prompt text from batch requests and saves to `scratch/` directory for examination

## Architecture

### Modular Prompt Assembly System
This repository uses a generalized prompt assembly system that builds prompts from modular components:

```
prompts/                         # Base prompt files with placeholders
templates/
├── configs/prompt_assembly.yaml # Configuration for prompt assembly  
├── parameter_metadata_template.yaml
├── prior_metadata_template.yaml
└── examples/k_ECM_fib_sec_example.yaml
scripts/
├── prompt_assembly.py          # Prompt assembly engine
├── batch_creator.py            # Base classes for batch creation
└── parameter_utils.py          # Parameter processing utilities
```

### Data Flow

**Parameter Definition Workflow:**
1. Input CSV with cancer_type and parameter_name columns
2. Generate parameter definitions with canonical scales using `create_parameter_definition_batch.py`
3. Store parameter definitions in `../qsp-metadata-storage/parameter_estimates/parameter-definitions/{cancer_type}/{parameter_name}/definition.yaml`

**Parameter Extraction Workflow:**
1. Same input CSV with cancer_type and parameter_name columns
2. Scripts load complete parameter definitions (name, units, definition, canonical_scale, mathematical_role) from storage
3. **Prompt assembly system** combines base prompts + templates + examples + parameter definition data
4. Batch processing via OpenAI API creates structured YAML outputs
5. Results are unpacked to `../qsp-metadata-storage/parameter_estimates/to-review/{cancer_type}/{parameter_name}/` for review

### Key Data Files
- `data/simbio_parameters.csv`: Parameter definitions with Name, Units, Definition, References columns (used for parameter definition creation)
- `data/model_context.csv`: Reaction context with Parameter, Reaction, ReactionRate, OtherParameters, OtherSpeciesWithNotes columns (used for parameter definition creation)
- `../qsp-metadata-storage/parameter_estimates/parameter-definitions/`: Stored parameter definitions (used for parameter extraction)
- `templates/configs/prompt_assembly.yaml`: Configuration controlling how prompts are assembled
- `templates/parameter_metadata_template.yaml`: YAML template for parameter metadata
- `templates/prior_metadata_template.yaml`: YAML template for prior metadata generation
- `templates/examples/`: Example filled templates for different parameters

### Class-based Batch Creation Architecture
Batch creation uses a modular class-based system:

- `scripts/batch_creator.py`: Base `BatchCreator` class with common functionality
- `ParameterBatchCreator`: For parameter extraction requests (uses prompt assembly system)
- `PoolingMetadataBatchCreator`: For adding statistical metadata to existing YAMLs
- CLI scripts (`create_parameter_batch.py`, `create_pooling_metadata_batch.py`) provide simple interfaces

### Script Dependencies
- `scripts/parameter_utils.py`: Utilities for parameter processing (CSV loading, model context generation)
- `scripts/prompt_assembly.py`: Modular prompt assembly from templates and examples  
- `scripts/batch_creator.py`: Class-based batch creation with shared functionality
- All API scripts expect `OPENAI_API_KEY` in `.env` file (current directory)
- `unpack_results.py` writes to `../qsp-metadata-storage/parameter_estimates/to-review/`
- `create_pooling_metadata_batch.py` reads from `../qsp-metadata-storage/parameter_estimates/to-review/`

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
- Writes extracted parameters to `../qsp-metadata-storage/parameter_estimates/to-review/` directory
- Assumes `qsp-metadata-storage` repository exists as sibling directory

## Standard Usage

- `unpack_results.py`: Extracts to `../qsp-metadata-storage/parameter_estimates` by default
- `create_pooling_metadata_batch.py`: Reads from `../qsp-metadata-storage/parameter_estimates/to-review/`

Example usage:
```bash
python scripts/unpack_results.py batch_results.jsonl ../qsp-metadata-storage/parameter_estimates
python scripts/create_pooling_metadata_batch.py ../qsp-metadata-storage/parameter_estimates/to-review
```

# Important Instructions

## Code Standards
- **No backward compatibility**: Use clean, modern interfaces without legacy support
- **Class-focused architecture**: Prefer class-based designs over functional approaches  
- **No main runners in libraries**: Only CLI scripts should have `if __name__ == "__main__":` blocks. Never add them to class files, utility modules, or library code
- **Explicit interfaces**: Require all necessary arguments, avoid complex default logic