# QSP LLM Workflows

This repository contains scripts and workflows for automated parameter extraction from scientific literature using Large Language Models.

## Overview

This toolkit automates the extraction of quantitative systems pharmacology (QSP) parameters from research papers using OpenAI's batch API. It's designed to extract parameters to the central `qsp-parameter-storage` repository.

## Directory Structure

```
├── scripts/           # Core automation scripts
│   ├── create_batch.py       # Create batch extraction requests
│   ├── prompt_assembly.py    # Modular prompt assembly system
│   ├── upload_batch.py       # Upload to OpenAI batch API
│   ├── batch_monitor.py      # Monitor batch progress
│   ├── unpack_results.py     # Extract results to YAML files
│   └── batch_workflow_commands.sh  # Complete workflow
├── prompts/base/     # Base prompt files with placeholders
├── templates/        # Modular prompt components
│   ├── configs/              # Prompt assembly configuration
│   ├── parameter_metadata_template.yaml
│   └── examples/             # Example filled templates
├── data/            # Reference data and examples
├── examples/        # Example workflows and outputs
└── batch_jobs/      # Batch processing files (gitignored)
```

## Workflow

1. **Create batch requests** from parameter CSV files
2. **Upload** to OpenAI batch API for processing
3. **Monitor** batch completion status
4. **Extract results** to YAML parameter files
5. **Review** extracted parameters in the central parameter storage

## Usage

The toolkit uses a **modular prompt assembly system** that builds prompts from reusable components:

### Basic Usage
```bash
# Create batch requests (uses new prompt assembly system)
python scripts/create_batch.py [input.csv]

# Upload and process
python scripts/upload_batch.py batch_jobs/batch_requests.jsonl
python scripts/batch_monitor.py batch_<id>

# Extract results to parameter storage
python scripts/unpack_results.py batch_jobs/batch_<id>_results.jsonl ../qsp-parameter-storage
```

The complete workflow is documented in `scripts/batch_workflow_commands.sh`.

## Integration

This repository is designed to work with the central `qsp-parameter-storage` repository. Extracted parameters are written to the `to-review/` directory in the parameter storage for validation and integration into the central parameter database.

## Modular Prompt System

The toolkit features a **generalized prompt assembly system** that:
- **Separates concerns**: Base prompts, templates, examples, and runtime data are modular
- **Enables reusability**: Templates and examples can be shared across different prompt types
- **Simplifies maintenance**: Changes to templates only need to be made once
- **Supports extensibility**: New prompt types can be added through configuration

### Prompt Components
- **Base prompts** (`prompts/base/`): Core instructions with placeholder markers
- **Templates** (`templates/`): YAML templates and configuration files
- **Examples** (`templates/examples/`): Example filled templates
- **Assembly engine** (`scripts/prompt_assembly.py`): Combines components into final prompts

## Integration

The workflow tools target the central parameter storage repository:

```bash
# Extract parameters to central storage
python scripts/unpack_results.py batch_results.jsonl ../qsp-parameter-storage

# Process pooling metadata from central storage
python scripts/create_pooling_metadata_batch.py ../qsp-parameter-storage/to-review
```

This supports the three-tier QSP architecture where individual projects reference parameters from the central storage rather than storing duplicate copies.