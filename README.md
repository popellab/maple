# QSP LLM Workflows

This repository contains scripts and workflows for automated parameter extraction from scientific literature using Large Language Models.

## Overview

This toolkit automates the extraction and validation of quantitative systems pharmacology (QSP) metadata from scientific literature using OpenAI's batch API. It supports multiple workflow types:

- **Parameter extraction**: Extract parameter values, ranges, and statistical distributions
- **Quick estimates**: Generate rapid parameter estimates for model initialization
- **Test statistics**: Create validation constraints from experimental data
- **Pooling metadata**: Add statistical pooling information to existing extractions

All extracted metadata is stored in the central `qsp-metadata-storage` repository.

## Directory Structure

```
├── scripts/           # Core automation scripts
│   ├── prepare/              # Batch creation scripts
│   │   ├── create_parameter_batch.py
│   │   ├── create_parameter_definition_batch.py
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
│   ├── matlab/               # MATLAB integration scripts
│   │   ├── compute_test_statistic_from_yaml.m
│   │   ├── generate_calibration_target_from_yaml.m
│   │   └── simple_test_harness.m
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
  ../qsp-metadata-storage/quick-estimates input.csv

# Aggregate estimates (in qspio-pdac)
python ../qspio-pdac/metadata/aggregate_quick_estimates.py input.csv \
  ../qsp-metadata-storage/quick-estimates output/
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

- **Parameter estimates**: `../qsp-metadata-storage/parameter_estimates/{param_name}_{author_year}_{cancer_type}_{definition_hash}.yaml`
- **Quick estimates**: `../qsp-metadata-storage/quick-estimates/{param_name}_{cancer_type}_{hash}_deriv{N}.yaml`
- **Test statistics**: `../qsp-metadata-storage/test_statistics/{test_stat_id}_{cancer_type}_{context_hash}.yaml`

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