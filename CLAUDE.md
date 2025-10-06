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
python scripts/prepare/create_parameter_batch.py input.csv

# Upload to OpenAI batch API
python scripts/run/upload_batch.py batch_jobs/batch_requests.jsonl

# Monitor batch progress and download when complete
python scripts/run/batch_monitor.py batch_<id>

# Unpack results to YAML files in central metadata storage
python scripts/process/unpack_results.py batch_jobs/batch_<id>_results.jsonl ../qsp-metadata-storage/parameter_estimates input.csv
```

### Script Organization

Scripts are organized by workflow stage:

**Prepare** (`scripts/prepare/`): Create batch requests
- `create_parameter_batch.py`: Parameter extraction batch requests
- `create_parameter_definition_batch.py`: Parameter definition batch requests
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
├── lib/
│   ├── prompt_assembly.py      # Prompt assembly engine
│   ├── batch_creator.py        # Base classes for batch creation
│   └── parameter_utils.py      # Parameter processing utilities
├── prepare/                     # Batch creation scripts
├── run/                         # Batch execution scripts
└── process/                     # Result processing scripts
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
5. Results are unpacked directly to `../qsp-metadata-storage/parameter_estimates/` with filename format: `{param_name}_{author_year}_{definition_hash}.yaml`

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
- Writes extracted parameters directly to `../qsp-metadata-storage/parameter_estimates/` with flat structure
- Filename format: `{param_name}_{author_year}_{definition_hash}.yaml`
- Assumes `qsp-metadata-storage` repository exists as sibling directory

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
