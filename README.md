# QSP LLM Workflows

This repository contains scripts and workflows for automated parameter extraction from scientific literature using Large Language Models.

## Overview

This toolkit automates the extraction of quantitative systems pharmacology (QSP) parameters from research papers using OpenAI's batch API. It's designed to work with any QSP project repository.

## Directory Structure

```
├── scripts/           # Core automation scripts
│   ├── create_batch.py       # Create batch extraction requests
│   ├── upload_batch.py       # Upload to OpenAI batch API
│   ├── batch_monitor.py      # Monitor batch progress
│   ├── unpack_results.py     # Extract results to YAML files
│   └── batch_workflow_commands.sh  # Complete workflow
├── prompts/          # LLM prompts and templates
├── data/            # Reference data and examples
├── examples/        # Example workflows and outputs
└── batch_jobs/      # Batch processing files (gitignored)
```

## Workflow

1. **Create batch requests** from parameter CSV files
2. **Upload** to OpenAI batch API for processing
3. **Monitor** batch completion status
4. **Extract results** to YAML parameter files
5. **Review** extracted parameters in the target QSP project

## Usage

See individual script files for detailed usage instructions. The complete workflow is documented in `scripts/batch_workflow_commands.sh`.

## Integration

This repository is designed to work with any QSP project. Extracted parameters are written to the `to-review/` directory in the specified target project for validation and integration.

## Configuration

The workflow tools require you to specify the target QSP project directory when unpacking results:

```bash
# Specify your QSP project path when unpacking results
python scripts/unpack_results.py batch_results.jsonl ../your-qsp-project

# For pooling metadata, specify the to-review directory
python scripts/create_pooling_metadata_batch.py ../your-qsp-project/to-review
```