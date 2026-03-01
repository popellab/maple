# Automated Extraction Workflow (Legacy)

> **Note**: This documentation covers the legacy parameter extraction and test statistic workflows. For new projects, see [calibration_workflow.md](calibration_workflow.md) which covers the recommended **IsolatedSystemTarget** workflow for extracting calibration targets with ODE submodels.

This guide walks you through extracting parameter estimates and test statistics from scientific literature using OpenAI's API via Pydantic AI. The workflow handles prompt generation, processing, and result unpacking automatically—you just provide an input CSV and run one command.

## First-Time Setup

### Prerequisites

You'll need Python 3.9+, Git with SSH keys configured for GitHub, and an OpenAI API key.

**Python:** Check with `python3 --version`. If not installed, get it from [python.org](https://www.python.org/downloads/) or via Homebrew (`brew install python3`).

**Git and SSH:** If you can run `ssh -T git@github.com` and see a success message, you're set. Otherwise, generate keys with `ssh-keygen -t ed25519 -C "your.email@example.com"` and add the public key to [GitHub Settings → SSH keys](https://github.com/settings/keys).

**OpenAI API Key:** JHU users should follow the [JHU API Key Guide](https://support.cmts.jhu.edu/hc/en-us/articles/38383798293133-Guide-to-Managing-API-Keys-and-Usage-Limits-on-platform-openai-com). Other institutions typically have their own provisioning process—check with IT.

### Installation

```bash
cd ~/Projects  # or wherever you keep code
git clone git@github.com:popellab/maple.git
cd qsp-llm-workflows

python3 -m venv venv
source venv/bin/activate
pip install -e .

# Store your API key (never commit this file)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Verify with `qsp-extract --help`—you should see usage information.

### Optional: Logfire for Debugging

For debugging and monitoring LLM workflows, see [Pydantic AI Logfire integration](https://ai.pydantic.dev/logfire/#using-logfire). Provides real-time visibility into prompts, responses, validation errors, and token usage.

## The Workflow

Every extraction follows the same pattern: prepare input → run extraction → validate → review.

### Step 1: Prepare Input CSV

The extraction needs context about your model. Start by exporting model definitions from your MATLAB file or SimBiology project:

```bash
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/model.m \
  --output jobs/input_data/model_definitions.json
```

This also creates `species_units.json` with unit information for each model species.

**For parameter extraction**, create a simple CSV with parameter names, then enrich it:

```csv
parameter_name
k_C_growth
k_C_death
```

```bash
qsp-enrich-csv parameter \
  params.csv \
  jobs/input_data/model_definitions.json \
  PDAC \
  -o jobs/input_data/enriched_params.csv
```

**For test statistics**, you write the `compute_test_statistic` function directly. This ensures the computation is exactly what you intend:

```csv
test_statistic_id,output_unit,model_output_code
tumor_volume_day14,millimeter ** 3,"def compute_test_statistic(time, species_dict, ureg):
    import numpy as np
    cells = species_dict['V_T.C1']
    idx = np.argmin(np.abs(time.magnitude - 14))
    return cells[idx] * (1e-6 * ureg.millimeter**3 / ureg.cell)"
```

Then enrich with scenario context:

```bash
qsp-enrich-csv test_statistic \
  test_stats.csv \
  scenario.yaml \
  jobs/input_data/species_units.json \
  -o jobs/input_data/enriched_test_stats.csv
```

The enrichment validates your code: it checks the function signature, verifies accessed species exist, and confirms the output has the right units.

### Step 2: Run Extraction

```bash
qsp-extract \
  jobs/input_data/enriched_params.csv \
  --type parameter \
  --output-dir ../your-model-repo/metadata-storage
```

The workflow processes requests via Pydantic AI and unpacks results to `metadata-storage/to-review/`.

### Step 3: Validate

Run the validation suite after extraction completes:

```bash
# For parameters
qsp-validate parameter_estimates \
  --dir ../your-model-repo/metadata-storage/to-review/parameter_estimates

# For test statistics (needs species_units.json for unit checking)
qsp-validate test_statistics \
  --dir ../your-model-repo/metadata-storage/to-review/test_statistics \
  --species-units-file jobs/input_data/species_units.json
```

The suite runs 9 validators: schema compliance, code execution, DOI resolution, text snippet verification, source reference checks, value consistency, duplicate source detection, model output code validation, and automated snippet source verification.

Results go to `validation-outputs/` as JSON files. If validation finds issues, review the JSON reports to understand what needs to be corrected.

### Step 4: Review and Approve

Before trusting the automated validation, open a few YAML files and do a quick sanity check. Does the `derivation_explanation` actually explain how the value was derived? Are the assumptions reasonable? Is the value in a plausible biological range?

Move approved files to their final location:

```bash
mv to-review/k_C_growth_*.yaml parameter_estimates/
```

Delete rejected files or re-run extraction for them.

## Command Reference

**Main commands:**

```bash
qsp-extract input.csv --type parameter --output-dir path/to/storage
qsp-extract input.csv --type test_statistic --output-dir path/to/storage
qsp-validate parameter_estimates --dir path/to/files
```

**Options:**

| Flag | What it does |
|------|--------------|
| `--preview-prompts` | Preview prompts without sending to API |
| `--reasoning-effort` | Set reasoning effort (low/medium/high, default: high) |

## Troubleshooting

**"No module named 'openai'"** — You need to activate the virtual environment. Run `source venv/bin/activate` every time you open a new terminal.

**"OPENAI_API_KEY not found"** — Create the `.env` file in the qsp-llm-workflows directory: `echo "OPENAI_API_KEY=sk-..." > .env`

**"Permission denied (publickey)"** — Your SSH keys aren't configured. Run `ssh-keygen -t ed25519` and add the public key to GitHub.

**Validation failures** — Check `validation-outputs/*.json` for details on what needs to be corrected.

## File Locations

Workflow files are organized as:

```
qsp-llm-workflows/
├── jobs/
│   ├── input_data/           # Your enriched CSVs go here
│   └── *.jsonl               # Request/response files (gitignored)
└── validation-outputs/       # Validation reports (gitignored)

your-model-repo/
└── metadata-storage/
    ├── to-review/            # Newly extracted files pending review
    ├── parameter_estimates/  # Approved parameter files
    └── test_statistics/      # Approved test statistic files
```
