# QSP LLM Workflows

Automated tools for extracting and validating quantitative systems pharmacology (QSP) metadata from scientific literature using Large Language Models.

## Quick Start

**New users:** See [docs/automated_workflow.md](docs/automated_workflow.md) for complete setup instructions and beginner-friendly guide.

**Developers:** See [CLAUDE.md](CLAUDE.md) for detailed architecture, script organization, and development guidelines.

## What This Repository Does

This toolkit automates metadata extraction from scientific papers using OpenAI's API:

- **Parameter extraction**: Extract parameter values, ranges, and distributions from literature
- **Test statistics**: Create validation constraints from experimental data

All extracted metadata is validated and stored in the companion `qsp-metadata-storage` repository.

## Basic Usage

**Simple workflow:**
```bash
# Activate environment
source venv/bin/activate

# Run automated extraction
python scripts/run_extraction_workflow.py \
  docs/example_parameter_input.csv \
  --type parameter
```

See [docs/automated_workflow.md](docs/automated_workflow.md) for complete instructions including:
- First-time setup (Python, Git, API keys)
- Creating input CSV files
- Running extractions
- Validation and review process
- Troubleshooting

## For Developers

See [CLAUDE.md](CLAUDE.md) for:
- Repository architecture and script organization
- Modular prompt assembly system
- Class-based batch creation
- Validation suite details
- Integration with qsp-metadata-storage
