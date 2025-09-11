# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains LLM workflow automation tools for extracting quantitative systems pharmacology (QSP) parameters from scientific literature using OpenAI's batch API. The tools are designed to extract parameters to the central `qsp-parameter-storage` repository.

## Key Commands

### Core Batch Workflow
The main workflow commands are documented in `scripts/batch_workflow_commands.sh`:

```bash
# Create batch requests from CSV using new prompt assembly system
python scripts/create_batch.py [input.csv] [params.csv] [reactions.csv] [output.jsonl]

# Upload to OpenAI batch API
python scripts/upload_batch.py batch_jobs/batch_requests.jsonl

# Monitor batch progress and download when complete
python scripts/batch_monitor.py batch_<id>

# Unpack results to YAML files in central parameter storage
python scripts/unpack_results.py batch_jobs/batch_<id>_results.jsonl ../qsp-parameter-storage
```

### Individual Script Usage

- `create_batch.py`: Uses new modular prompt assembly system. Can be called with no args (uses defaults), with just input CSV, or with all 4 args (input CSV, params CSV, reactions CSV, output JSONL)
- `upload_batch.py`: Requires JSONL file path, expects OpenAI API key in `.env` file
- `batch_monitor.py`: Requires batch ID, automatically downloads results when batch is complete
- `unpack_results.py`: Extracts YAML from batch results to `qsp-parameter-storage/to-review/` directory structure

## Architecture

### Modular Prompt Assembly System
This repository uses a generalized prompt assembly system that builds prompts from modular components:

```
prompts/base/                    # Base prompt files with placeholders
templates/
├── configs/prompt_assembly.yaml # Configuration for prompt assembly  
├── parameter_metadata_template.yaml
└── examples/k_ECM_fib_sec_example.yaml
scripts/prompt_assembly.py       # Prompt assembly engine
```

### Data Flow
1. Input CSV with cancer_type and parameter_name columns
2. Scripts generate parameter info and model context from CSV data files
3. **Prompt assembly system** combines base prompts + templates + examples + runtime data
4. Batch processing via OpenAI API creates structured YAML outputs
5. Results are unpacked to `../qsp-parameter-storage/to-review/{cancer_type}/{parameter_name}/` for review

### Key Data Files
- `data/simbio_parameters.csv`: Parameter definitions with Name, Units, Definition, References columns
- `data/model_context.csv`: Reaction context with Parameter, Reaction, ReactionRate, OtherParameters, OtherSpeciesWithNotes columns
- `templates/configs/prompt_assembly.yaml`: Configuration controlling how prompts are assembled
- `templates/parameter_metadata_template.yaml`: YAML template for parameter metadata
- `templates/examples/`: Example filled templates for different parameters

### Script Dependencies
- `create_batch.py` uses `prompt_assembly.py` and functions from `generate_prompts.py`
- All API scripts expect `OPENAI_API_KEY` in `.env` file (current directory)
- `unpack_results.py` writes to `../qsp-parameter-storage/to-review/`
- `create_pooling_metadata_batch.py` reads from `../qsp-parameter-storage/to-review/`

### Batch Processing Model
- Uses OpenAI's batch API with GPT-5 model and high reasoning effort
- Custom IDs follow format: `{cancer_type}_{parameter_name}_{index}`
- Results are saved to `batch_jobs/` directory (gitignored)
- Batch IDs are tracked in `.batch_id` files alongside JSONL files

### Prompt Assembly Benefits
- **Modular**: Templates and examples are reusable across different prompt types
- **Maintainable**: Changes to templates only need to be made once
- **Extensible**: New prompt types can be added through configuration
- **Flexible**: Components can be mixed and matched for different use cases

## Integration Points

## Integration with Parameter Storage

This repository integrates with the central parameter storage system:
- Reads API key from `.env` file (current directory)
- Writes extracted parameters to `../qsp-parameter-storage/to-review/` directory
- Assumes `qsp-parameter-storage` repository exists as sibling directory

## Standard Usage

- `unpack_results.py`: Extracts to `../qsp-parameter-storage` by default
- `create_pooling_metadata_batch.py`: Reads from `../qsp-parameter-storage/to-review/`

Example usage:
```bash
python scripts/unpack_results.py batch_results.jsonl ../qsp-parameter-storage
python scripts/create_pooling_metadata_batch.py ../qsp-parameter-storage/to-review
```