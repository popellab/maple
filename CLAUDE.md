# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains LLM workflow automation tools for extracting quantitative systems pharmacology (QSP) parameters from scientific literature using OpenAI's batch API. The tools are designed to work with any QSP project.

## Key Commands

### Core Batch Workflow
The main workflow commands are documented in `scripts/batch_workflow_commands.sh`:

```bash
# Create batch requests from CSV
python scripts/create_batch.py [input.csv]

# Upload to OpenAI batch API
python scripts/upload_batch.py batch_jobs/batch_requests.jsonl

# Monitor batch progress and download when complete
python scripts/batch_monitor.py batch_<id> --download

# Unpack results to YAML files in target QSP project/to-review/
python scripts/unpack_results.py batch_jobs/batch_<id>_results.jsonl ../your-qsp-project
```

### Individual Script Usage

- `create_batch.py`: Can be called with no args (uses defaults), with just input CSV, or with all 5 args (input CSV, params CSV, reactions CSV, template MD, output JSONL)
- `upload_batch.py`: Requires JSONL file path, expects OpenAI API key in `../.env` file
- `batch_monitor.py`: Requires batch ID, use `--download` flag to save results when complete
- `unpack_results.py`: Extracts YAML from batch results to specified QSP project's `to-review/` directory structure

## Architecture

### Data Flow
1. Input CSV with cancer_type and parameter_name columns
2. Scripts generate prompts by combining parameter definitions from `data/simbio_parameters.csv` with model context from `data/model_context.csv` 
3. Prompts use templates from `prompts/` directory
4. Batch processing via OpenAI API creates structured YAML outputs
5. Results are unpacked to `{target_project}/to-review/{cancer_type}/{parameter_name}/` for review

### Key Data Files
- `data/simbio_parameters.csv`: Parameter definitions with Name, Units, Definition, References columns
- `data/model_context.csv`: Reaction context with Parameter, Reaction, ReactionRate, OtherParameters, OtherSpeciesWithNotes columns
- `prompts/qsp_parameter_extraction_prompt.md`: Main prompt template with PARAMETER_TO_SEARCH and MODEL_CONTEXT placeholders

### Script Dependencies
- `create_batch.py` imports functions from `generate_prompts.py` for prompt creation
- All API scripts expect `OPENAI_API_KEY` in `../.env` file (goes up to parent directory)
- `unpack_results.py` writes to specified target project's `to-review/` directory
- `create_pooling_metadata_batch.py` reads from specified target project's `to-review/` directory

### Batch Processing Model
- Uses OpenAI's batch API with GPT-5 model and high reasoning effort
- Custom IDs follow format: `{cancer_type}_{parameter_name}_{index}`
- Results are saved to `batch_jobs/` directory (gitignored)
- Batch IDs are tracked in `.batch_id` files alongside JSONL files

## Integration Points

## Integration with QSP Projects

This repository integrates with any QSP project:
- Reads API key from `../.env` (parent directory)
- Writes extracted parameters to specified target project's `to-review/` directory
- Requires explicit specification of target QSP project path in commands

## Required Arguments

- `unpack_results.py`: Requires target project directory as second argument
- `create_pooling_metadata_batch.py`: Requires to-review directory path as first argument

Example usage:
```bash
python scripts/unpack_results.py batch_results.jsonl ../my-qsp-project
python scripts/create_pooling_metadata_batch.py ../my-qsp-project/to-review
```