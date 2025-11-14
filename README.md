# QSP LLM Workflows

[![Tests](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml/badge.svg)](https://github.com/popellab/qsp-llm-workflows/actions/workflows/test.yml)

Automated tools for extracting and validating quantitative systems pharmacology (QSP) metadata from scientific literature using Large Language Models.

## Installation

### For Development

```bash
# Clone the repository
git clone https://github.com/yourorg/qsp-llm-workflows.git
cd qsp-llm-workflows

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### For End Users

```bash
# Install directly from GitHub
pip install git+https://github.com/yourorg/qsp-llm-workflows.git

# Or from PyPI (if published)
pip install qsp-llm-workflows
```

After installation, CLI commands are available system-wide:
- `qsp-extract` - Run extraction workflows
- `qsp-validate` - Validate extracted metadata
- `qsp-fix` - Fix validation errors
- `qsp-enrich-csv` - Enrich input CSVs with model context
- `qsp-export-model` - Export model definitions from MATLAB
- `qsp-batch-monitor` - Monitor batch job progress

## Quick Start

**New users:** See [docs/automated_workflow.md](docs/automated_workflow.md) for complete setup instructions and beginner-friendly guide.

**Developers:** See [CLAUDE.md](CLAUDE.md) for detailed architecture, package structure, and development guidelines.

## What This Package Does

This toolkit automates metadata extraction from scientific papers using OpenAI's API:

- **Parameter extraction**: Extract parameter values, ranges, and distributions from literature
- **Test statistics**: Create validation constraints from experimental data
- **Validation suite**: 8 automated validators ensure quality and completeness
- **Git integration**: Automated branch creation and review workflow

All extracted metadata is validated and stored in the companion `qsp-metadata-storage` repository.

## Basic Usage

**Simple extraction workflow:**
```bash
# Run automated extraction
qsp-extract input.csv --type parameter

# Validate results
qsp-validate parameter_estimates

# Fix validation errors (if needed)
qsp-fix parameter_estimates --immediate
```

**Use immediate mode for faster processing:**
```bash
qsp-extract input.csv --type parameter --immediate
```

See [docs/automated_workflow.md](docs/automated_workflow.md) for complete instructions including:
- First-time setup (Python, Git, API keys)
- Creating input CSV files
- Running extractions
- Validation and review process
- Troubleshooting

## Package Structure

This repository is organized as an installable Python package:

```
qsp-llm-workflows/
├── src/qsp_llm_workflows/    # Main package
│   ├── core/                  # Core libraries
│   ├── prepare/               # Batch preparation
│   ├── run/                   # Batch execution
│   ├── process/               # Result processing
│   ├── validate/              # Validation checks
│   ├── cli/                   # CLI entry points
│   ├── templates/             # YAML templates
│   └── prompts/               # Prompt files
├── pyproject.toml             # Package metadata
└── docs/                      # Documentation
```

## Features

- **Installable Package**: `pip install` for easy distribution
- **CLI Commands**: System-wide commands after installation
- **Modular Architecture**: Clean separation of concerns
- **Automated Workflows**: End-to-end extraction and validation
- **Git Integration**: Automated branch creation and review
- **Validation Suite**: 8 validators including DOI resolution, code execution, and manual verification
- **Error Fixing**: Automatically resubmit failed extractions to OpenAI for correction

## For Developers

See [CLAUDE.md](CLAUDE.md) for:
- Package architecture and organization
- Modular prompt assembly system
- Class-based batch creation
- Validation suite implementation
- Integration with qsp-metadata-storage
- Development guidelines and code standards

## Documentation

- **[docs/automated_workflow.md](docs/automated_workflow.md)** - Beginner-friendly usage guide
- **[CLAUDE.md](CLAUDE.md)** - Developer documentation and architecture
- **[PACKAGE_STRUCTURE.md](PACKAGE_STRUCTURE.md)** - Package migration guide

## License

MIT
