# Automated Extraction Workflow

## Overview

This workflow automates LLM-based parameter extraction from scientific papers. Instead of running 7+ manual commands, you run **one command** and the system:

1. Creates batch requests
2. Uploads to OpenAI API
3. Monitors progress automatically
4. Unpacks files to a staging area
5. Creates a git branch and pushes for review

**After the workflow completes,** you manually run validation to check the extracted data.

**Time savings:** From 20-30 minutes of manual work → 1 minute to launch, then walk away.

**Note on examples:** All code examples in this document use generic placeholder paths like `your-model-repo` and `YOUR_CANCER_TYPE`. Replace these with your actual repository and model names when running commands.

### What Happens When You Run a Workflow?

**Prerequisites:** Export model definitions and enrich your CSV (see [Input Files](#preparing-your-input-file))

```
┌─────────────────────────────────────────────────────────────┐
│  You have an enriched CSV with parameter definitions        │
│  cancer_type,parameter_name,definition_hash,...             │
│  PDAC,k_C_growth,abc123,...                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  You run ONE command:                                       │
│  qsp-extract input.csv --type parameter                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  The system automatically:                                  │
│  1. Creates batch requests and uploads to OpenAI           │
│  2. Monitors progress (shows you updates)                   │
│  3. Downloads results when complete                         │
│  4. Unpacks files to to-review/ directory                   │
│  5. Creates git branch and pushes for review                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  You run validation manually:                               │
│  cd ../qsp-metadata-storage                                 │
│  git checkout review/batch-parameter-2025-10-27-abc123      │
│  qsp-validate parameter_estimates                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Review validation reports and approve/reject files         │
│  Check output/validation/ for detailed reports              │
└─────────────────────────────────────────────────────────────┘
```

---

## First-Time Setup

If this is your first time using the workflow, follow these setup steps carefully. You only need to do this once.

### Step 1: Install Python (if not already installed)

Check if Python 3 is installed:

```bash
python3 --version
```

If you see something like `Python 3.9.x` or higher, you're good. If not, install Python:
- **Mac:** Install via [Homebrew](https://brew.sh/): `brew install python3`
- **Linux:** `sudo apt install python3 python3-pip`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

### Step 2: Set Up GitHub SSH Keys

SSH keys let you clone and push to GitHub without entering your password every time.

**Check if you already have SSH keys:**

```bash
ls -al ~/.ssh
```

If you see files like `id_rsa` and `id_rsa.pub` (or `id_ed25519` and `id_ed25519.pub`), you already have keys. Skip to "Add SSH key to GitHub" below.

**Generate new SSH keys (if needed):**

```bash
# Generate SSH key (replace with your GitHub email)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Press Enter to accept default file location
# Press Enter twice to skip passphrase (or set one if you prefer)
```

**Add SSH key to GitHub:**

```bash
# Copy your public key to clipboard
# Mac:
cat ~/.ssh/id_ed25519.pub | pbcopy

# Linux (with xclip):
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard

# Or just display it and copy manually:
cat ~/.ssh/id_ed25519.pub
```

Then:
1. Go to [GitHub Settings → SSH and GPG keys](https://github.com/settings/keys)
2. Click "New SSH key"
3. Paste your public key
4. Click "Add SSH key"

**Test your connection:**

```bash
ssh -T git@github.com
```

You should see: `Hi username! You've successfully authenticated...`

### Step 3: Clone the Repositories

You need **two** repositories as sibling directories:

```bash
# Navigate to where you want your projects (e.g., Documents or Projects)
cd ~/Projects  # or wherever you keep code

# Clone the workflows repository
git clone git@github.com:popellab/qsp-llm-workflows.git

# Clone the metadata storage repository
git clone git@github.com:popellab/qsp-metadata-storage.git

# Your directory structure should now be:
# Projects/
# ├── qsp-llm-workflows/
# └── qsp-metadata-storage/
```

**Important:** These repos must be **siblings** (in the same parent directory) for the workflow to work.

### Step 4: Set Up Python Virtual Environment

A virtual environment keeps Python packages for this project separate from your system.

```bash
# Navigate to the workflows repository
cd qsp-llm-workflows

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Your prompt should now show (venv) at the beginning
```

**Install required packages:**

```bash
# Make sure venv is activated (you should see (venv) in your prompt)
pip install -r requirements.txt
```

### Step 5: Provision Your OpenAI API Key

Each user provisions and manages their own OpenAI API key.

**For JHU users:**
Follow the [JHU Guide to Managing API Keys and Usage Limits](https://support.cmts.jhu.edu/hc/en-us/articles/38383798293133-Guide-to-Managing-API-Keys-and-Usage-Limits-on-platform-openai-com)

**For other institutions:**
Check with your institution's IT support for guidance on provisioning OpenAI API keys through institutional accounts.

**Once you have your API key:**

```bash
# From the qsp-llm-workflows directory:
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

Replace `sk-your-key-here` with your actual API key.

**Important:**
- Never commit the `.env` file to git (it's already in `.gitignore`)
- Never share your API key with others
- Each user should have their own key for usage tracking

**Verify everything works:**

```bash
# Make sure you're in qsp-llm-workflows directory
# Make sure virtual environment is activated (venv)
qsp-extract --help
```

You should see help text without errors. If so, you're ready to run extractions!

---

## Quick Start (For Regular Use)

Once you've completed first-time setup, starting a workflow is simple:

```bash
# 1. Navigate to the repository
cd ~/Projects/qsp-llm-workflows

# 2. Activate virtual environment (do this every time you open a new terminal)
source venv/bin/activate

# 3. Run an example workflow
# First export model definitions from your MATLAB model file or SimBiology project
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/your_model_file.m \
  --output batch_jobs/input_data/model_definitions.json

# Or if you have a SimBiology project file:
# qsp-export-model \
#   --simbiology-project ../your-model-repo/models/your_model.sbproj \
#   --output batch_jobs/input_data/model_definitions.json

# Then enrich your input CSV with model context
qsp-enrich-csv parameter \
  your_parameter_input.csv \
  batch_jobs/input_data/model_definitions.json \
  YOUR_CANCER_TYPE \
  -o batch_jobs/input_data/enriched_input.csv

# Finally run extraction
qsp-extract \
  batch_jobs/input_data/enriched_input.csv \
  --type parameter \
  --immediate
```

**Example CSV files:** We've included simple example CSVs in the `docs/` directory:
- `docs/example_parameter_input.csv` - Simple list of 2 parameter names
- `docs/example_test_statistic_input.csv` - Partial test statistic definitions (3 metrics)

**Note:** These examples are generic templates. To run them, you'll need a model repository with your MATLAB model file, model context, and scenario definitions. The examples demonstrate the two-step workflow: enrichment → extraction.

**For faster testing:** Use the `--immediate` flag to process via the Responses API instead of waiting for batch completion (minutes instead of hours, perfect for testing). The batch API is cheaper for large production runs.

---

## Preparing Your Input File

The workflow requires a **two-step process**:
1. **Step 1: Create enriched CSV** with model definitions (this section)
2. **Step 2: Run extraction workflow** (next section)

### Parameter Extraction Input File

**IMPORTANT:** Parameter extraction requires an enriched CSV with model definitions. The simple example CSVs in this repository (like `docs/example_parameter_input.csv`) are **NOT sufficient** for actual extraction - they're only for demonstration.

#### Step 1: Create Enriched CSV

Start with a simple CSV listing parameter names:

```csv
parameter_name
k_C_growth
k_C_death
k_CD8_act
```

Then enrich it with model definitions:

```bash
# Enrich simple CSV with model definitions
qsp-enrich-csv parameter \
  simple_parameter_input.csv \
  model_definitions.json \
  PDAC \
  -o batch_jobs/input_data/pdac_extraction_input.csv
```

**Where do model definitions come from?**

Model definitions are exported from MATLAB model files or SimBiology project files using the export script:

```bash
# From qsp-llm-workflows repository:

# Option 1: From MATLAB script (runs the .m file to create model)
qsp-export-model \
  --matlab-model ../your-model-repo/scripts/your_model_file.m \
  --output batch_jobs/input_data/model_definitions.json

# Option 2: From SimBiology project file (faster if model is already compiled)
qsp-export-model \
  --simbiology-project ../your-model-repo/models/your_model.sbproj \
  --output batch_jobs/input_data/model_definitions.json
```

The MATLAB script option works with any SimBiology model file that creates a variable named `model`. The SimBiology project option loads a saved `.sbproj` file directly, which can be faster for large models. Replace paths with your actual model repository location.

#### Enriched CSV Format

The enrichment script creates a CSV with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `cancer_type` | Cancer type | `PDAC` |
| `parameter_name` | Parameter name | `k_C_growth` |
| `definition_hash` | Hash of parameter definition | `abc123` |
| `parameter_units` | Units | `1/day` |
| `parameter_description` | Description | `Cancer cell growth rate` |
| `model_context` | JSON with reactions/rules | `{"reactions_and_rules":[...]}` |

#### Example Files

The repository includes example CSVs in `docs/`:
- `example_parameter_input.csv` - Simple parameter names (input for enrichment)
- `example_test_statistic_input.csv` - Partial test statistics (input for enrichment)

See the [Quick Start](#quick-start-for-regular-use) section for usage examples.

### Test Statistics Input File

**IMPORTANT:** Test statistic extraction requires an enriched CSV with model and scenario context.

#### Step 1: Create Enriched CSV

Start with a partial CSV listing test statistics:

```csv
test_statistic_id,required_species,derived_species_description
tumor_volume_day14,V_T.C,Tumor volume in mm³ at 14 days post-implantation
cd8_infiltration,V_T.CD8,CD8+ T cell count per mm³ tumor tissue
```

Then enrich it with model and scenario context:

```bash
# Enrich partial CSV with context
qsp-enrich-csv test_statistic \
  partial_test_stats.csv \
  model_context.txt \
  baseline_no_treatment.yaml \
  -o batch_jobs/input_data/test_statistic_input.csv
```

**Where do context files come from?**

Context files are stored in model-specific repositories (e.g., `qspio-pdac`):
- `model_context.txt`: Description of model structure (species, compartments, units)
- Scenario YAMLs: Experimental conditions with `scenario_context` and `indication` fields

See your model repository's documentation for context file locations.

#### Enriched CSV Format

The enrichment script creates a CSV with these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `test_statistic_id` | Unique ID | `tumor_volume_day14` |
| `cancer_type` | Cancer type from scenario | `PDAC` |
| `model_context` | Model structure text | Full model description |
| `scenario_context` | Experimental conditions | `Untreated PDAC tumor growth` |
| `required_species` | Species from model | `V_T.C` |
| `derived_species_description` | What it represents | `Tumor volume in mm³ at 14 days` |
| `context_hash` | Hash of model+scenario | `abc123` |

#### Example Files

See `docs/example_test_statistic_input.csv` for a simple example of partial input format.

### Common Issues

**Problem:** Script says "No such file"
- **Solution:** Make sure you're running the command from the `qsp-llm-workflows` directory
- **Solution:** Check the file path - try `ls parameter_input.csv` to verify it exists

**Problem:** CSV parsing errors
- **Solution:** Make sure column headers are **exactly** as shown (case-sensitive, no spaces)
- **Solution:** Check for hidden characters - save as plain text CSV, not Excel format
- **Solution:** No empty rows at the beginning or end of file

**Problem:** "Missing model definitions" or "Required column missing"
- **Solution:** Ensure you've run the CSV enrichment step first (see "Preparing Your Input File" above)
- **Solution:** For production extractions, use enriched CSVs with all required columns
- **Solution:** Example CSVs are for testing only - they lack model context needed for quality extractions

### Where to Put Your Input Files

You can put input files anywhere, but we recommend:

```bash
qsp-llm-workflows/
├── batch_jobs/
│   └── input_data/          # Put your CSV files here
│       ├── parameter_input.csv
│       └── test_stat_input.csv
```

Then reference them in commands:

```bash
qsp-extract \
  batch_jobs/input_data/parameter_input.csv \
  --type parameter
```

---

## Usage

### Basic Syntax

```bash
qsp-extract <input.csv> --type <workflow_type> [options]
```

### Workflow Types

- `parameter` - Full parameter extraction with v3 schema
- `test_statistic` - Test statistic extraction with v2 schema

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--timeout SECONDS` | Max wait time for batch completion | 3600 (1 hour) |
| `--immediate` | Use Responses API for immediate processing (faster, more expensive) | False |
| `--no-push` | Create branch locally without pushing | False |
| `--branch-prefix PREFIX` | Custom prefix for review branches | `review/batch` |

**Note:** The `--skip-validation` option has been removed. Validation is now always a separate manual step after the workflow completes.

## Examples

### Parameter Extraction

```bash
# Simple example with 2 parameters (good for testing)
# Step 1: Enrich with model definitions
qsp-enrich-csv parameter \
  docs/example_parameter_input.csv \
  batch_jobs/input_data/model_definitions.json \
  YOUR_CANCER_TYPE \
  -o batch_jobs/input_data/example_enriched.csv

# Step 2: Extract (use --immediate for faster testing)
qsp-extract \
  batch_jobs/input_data/example_enriched.csv \
  --type parameter \
  --immediate

# Production example with enriched CSV
qsp-extract \
  batch_jobs/input_data/production_parameters.csv \
  --type parameter \
  --timeout 7200
```

### Test Statistics

```bash
# Simple example with 3 test statistics (good for testing)
# Step 1: Enrich with model context and scenario
qsp-enrich-csv test_statistic \
  docs/example_test_statistic_input.csv \
  ../your-model-repo/model_context.txt \
  ../your-model-repo/scenarios/baseline_scenario.yaml \
  -o batch_jobs/input_data/example_enriched.csv

# Step 2: Extract
qsp-extract \
  batch_jobs/input_data/example_enriched.csv \
  --type test_statistic \
  --immediate

# Production example with enriched CSV
qsp-extract \
  batch_jobs/input_data/production_test_statistics.csv \
  --type test_statistic
```

**Note:** Validation is always a separate manual step. Run it after the workflow completes if needed.

### Local-Only Review

```bash
# Create review branch but don't push (review locally first)
qsp-extract \
  input.csv \
  --type parameter \
  --no-push
```

## Workflow Output

### Success Output

```
======================================================================
AUTOMATED EXTRACTION WORKFLOW
======================================================================
Type: parameter
Input: batch_jobs/input_data/core_extraction_input.csv
Timeout: 3600s
Validation: Enabled
Push: Enabled
======================================================================

Creating parameter batch requests...
✓ Batch requests created: parameter_requests.jsonl
Uploading batch: parameter_requests.jsonl...
✓ Batch uploaded: batch_ABC123
Monitoring batch batch_ABC123...
  Status: validating (0/15 completed)
  Status: in_progress (5/15 completed)
  Status: in_progress (10/15 completed)
  Status: in_progress (15/15 completed)
  Status: finalizing (15/15 completed)
✓ Results downloaded: batch_ABC123_results.jsonl
Unpacking results to to-review/...
✓ Unpacked 15 files to to-review/
Creating review branch and committing files...
✓ Pushed to origin/review/batch-parameter-2025-10-27-abc123

======================================================================
WORKFLOW SUMMARY
======================================================================
✓ Status: SUCCESS
✓ Workflow type: parameter
✓ Files extracted: 15
✓ Duration: 450.2s
✓ Review branch: review/batch-parameter-2025-10-27-abc123
✓ Pushed to origin/review/batch-parameter-2025-10-27-abc123

Next steps:
  1. cd ../qsp-metadata-storage
  2. git checkout review/batch-parameter-2025-10-27-abc123
  3. Manual high-level review: Open a few YAMLs, check derivation_explanation and assumptions
  4. Run validation: qsp-validate parameter_estimates
  5. Manual snippet verification: Click DOI links, verify snippets with Ctrl+F
  6. Review detailed validation reports in output/validation/
  7. Move approved files to appropriate directories (reject bad ones)
  8. Commit changes and open a pull request on GitHub
  9. Merge PR after team review
======================================================================
```

## Review Process

After the workflow completes, review and validate the extracted files. **Recommended order:**

1. Manual high-level review (quick quality check)
2. Automated validation suite
3. Manual snippet source verification (if needed)
4. Detailed review of validation reports
5. Approve/reject files
6. Commit and open PR

### 1. Checkout Review Branch

```bash
cd ../qsp-metadata-storage
git checkout review/batch-parameter-2025-10-27-abc123
```

### 2. Manual High-Level Review (Do This First!)

**Before running automated validation**, do a quick manual review to catch obvious issues:

Open a few YAML files in `to-review/` and check:

- [ ] Does `derivation_explanation` clearly describe how the value was obtained?
- [ ] Are mathematical transformations and statistical methods explained and justified?
- [ ] Are key assumptions explicitly stated and reasonable?
- [ ] Are unit conversions and data transformations transparent and traceable?
- [ ] Is the value in a reasonable biological range?

**Red flags to watch for:**
- Vague explanations like "extracted from paper" or "calculated from data"
- Missing justification for unit conversions
- No assumptions listed (very few extractions have zero assumptions)
- Values that seem biologically implausible

See `scripts/validate/MANUAL_REVIEW_CHECKLIST.md` for the complete checklist.

**If you find major issues at this stage:**
- Consider discarding the results and re-running the extraction with improved prompts
- Or use the validation fix workflow: `qsp-fix <workflow_type>`

### 3. Run Automated Validation Suite

After the manual review, run the automated validators:

```bash
# For parameter estimates
qsp-validate parameter_estimates

# For test statistics
qsp-validate test_statistics
```

The validation suite will run all 8 validators and generate detailed reports in `output/validation/`.

**The automated validation includes:**
1. Schema compliance (YAML structure)
2. Source references (all source_refs are valid)
3. Text snippets (snippets contain the reported values)
4. DOI validity (DOIs resolve to real papers)
5. Code execution (R/Python derivation code runs)
6. Value consistency (cross-checks against other extractions)
7. Duplicate primary sources (checks for already-used primary data sources)
8. Manual snippet source verification (interactive paper verification)

### 4. Manual Snippet Source Verification

**Important:** The final validator prompts you to **manually verify snippets in papers**.

When you run `run_all_validations.py`, the last step will:
1. Print a report with DOI links and text snippets grouped by source
2. **Wait for you** to click DOI links and verify snippets appear in papers
3. Ask you to confirm (y/n) that snippets are accurate

**How to verify:**
- Click the DOI link to open the paper in your browser
- Use Ctrl+F (or Cmd+F) to search for the snippet text
- Verify the snippet actually appears in the paper
- Check that the context matches the claimed value

If verified, the validator will tag all files with `manual_snippet_source_verification`.

### 5. Review Detailed Validation Reports

Check files in `to-review/`:
- Verify citations are real and accessible
- Confirm extracted values match sources
- Validate derivation logic and assumptions
- Check unit consistency
- Review validation reports in `output/validation/`
  - `schema_compliance.json` - Template compliance results
  - `source_references.json` - Source reference validation
  - `text_snippets.json` - Text snippet verification
  - `doi_validity.json` - DOI resolution results
  - `code_execution.json` - Code execution test results
  - `value_consistency.json` - Value consistency checks
  - `duplicate_primary_sources.json` - Duplicate primary source checks
  - `snippet_sources.json` - Manual snippet source verification results

### 6. Approve/Reject Files

After reviewing both manual and automated validation results, approve or reject each file.

**Approve files** by moving to final location:

```bash
# Parameter estimates
mv to-review/k_C_growth_*.yaml parameter_estimates/

# Test statistics
mv to-review/tumor_volume_*.yaml test_statistics/
```

**Reject files** by deleting:

```bash

# Delete rejected file
rm to-review/k_C_growth_Smith2020_PDAC_abc123.yaml
```

### 7. Commit and Open Pull Request

```bash
# Commit your review changes
git add .
git commit -m "Review complete: approved 12 files, rejected 3 files"

# Push to remote
git push

# Open a pull request on GitHub
# Option 1: Using GitHub web interface
#   Go to: https://github.com/popellab/qsp-metadata-storage/pulls
#   Click "New pull request"
#   Select your review branch (e.g., review/batch-parameter-2025-10-27-abc123)
#   Add description of what was reviewed and approved
#   Request review from team members
#   Click "Create pull request"

# Option 2: Using GitHub CLI (if installed)
gh pr create --title "Review: batch-parameter-2025-10-27" \
  --body "Reviewed and approved 12 files, rejected 3 files. See commit message for details."
```

**After PR is approved and merged:**

```bash
# Switch back to main and update
git checkout main
git pull

# The review branch will be automatically deleted after merge
```

---

## Error Handling

If the workflow fails, you'll see:

```
✗ Workflow failed: Batch batch_ABC123 did not complete within 3600s

======================================================================
WORKFLOW SUMMARY
======================================================================
✗ Status: FAILED
✗ Error: Batch batch_ABC123 did not complete within 3600s
✗ Duration: 3600.5s
======================================================================
```

Common errors:
- **Timeout**: Batch didn't complete in time → increase `--timeout`
- **No results**: Input CSV has issues → verify CSV format
- **Git errors**: Branch already exists or can't push → check git status
- **Unpacking errors**: Template mismatch → ensure you're using the correct workflow type

## Batch Jobs Directory

Workflow creates/uses these files in `batch_jobs/`:

- `parameter_requests.jsonl` - Extraction batch requests
- `parameter_requests.batch_id` - Batch metadata (ID, type, CSV)
- `batch_ABC123_results.jsonl` - Raw extraction results

Validation reports are written to `output/validation/`:
- `schema_compliance.json`
- `source_references.json`
- `text_snippets.json`
- `doi_validity.json`
- `code_execution.json`
- `value_consistency.json`
- `duplicate_primary_sources.json`
- `snippet_sources.json`

All batch files are gitignored and available for debugging if needed.

---

## Troubleshooting

### Common Issues for Beginners

#### "python: command not found" or "python3: command not found"

**Problem:** Python is not installed or not in your PATH.

**Solution:**
```bash
# Check if Python 3 is installed
which python3

# If not found, install Python (Mac with Homebrew)
brew install python3

# Try using python3 explicitly to check installation
python3 --version
```

#### "No module named 'openai'" or similar import errors

**Problem:** You haven't activated the virtual environment or haven't installed dependencies.

**Solution:**
```bash
# Make sure you're in the right directory
cd ~/Projects/qsp-llm-workflows

# Activate virtual environment (you should see (venv) in your prompt)
source venv/bin/activate

# If you still get errors, reinstall dependencies
pip install -r requirements.txt
```

**Important:** You must activate the virtual environment (`source venv/bin/activate`) **every time** you open a new terminal window.

#### "OPENAI_API_KEY not found"

**Problem:** The `.env` file doesn't exist or is in the wrong location.

**Solution:**
```bash
# Make sure you're in qsp-llm-workflows directory
pwd  # Should show: /Users/yourname/Projects/qsp-llm-workflows

# Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env

# Verify the file was created
cat .env
```

**If you don't have an API key yet:**
- **JHU users:** Follow the [JHU API Key Provisioning Guide](https://support.cmts.jhu.edu/hc/en-us/articles/38383798293133-Guide-to-Managing-API-Keys-and-Usage-Limits-on-platform-openai-com)
- **Other institutions:** Check with your institution's IT support for OpenAI API key provisioning

#### "qsp-metadata-storage not found"

**Problem:** The metadata storage repository doesn't exist or is in the wrong location.

**Solution:**
```bash
# Check your directory structure
cd ~/Projects  # Or wherever you cloned repos
ls -la

# You should see both:
# qsp-llm-workflows/
# qsp-metadata-storage/

# If qsp-metadata-storage is missing, clone it:
git clone git@github.com:popellab/qsp-metadata-storage.git
```

Both repos **must** be siblings (in the same parent directory).

#### "Permission denied (publickey)" when pushing to GitHub

**Problem:** SSH keys aren't set up correctly.

**Solution:**
```bash
# Test SSH connection
ssh -T git@github.com

# If it fails, check if you have SSH keys
ls -al ~/.ssh

# If no keys exist, generate them (see "Step 2: Set Up GitHub SSH Keys" above)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add public key to GitHub at: https://github.com/settings/keys
cat ~/.ssh/id_ed25519.pub
```

#### "fatal: not a git repository"

**Problem:** You're running commands from the wrong directory.

**Solution:**
```bash
# Always run workflow commands from qsp-llm-workflows directory
cd ~/Projects/qsp-llm-workflows

# Verify you're in the right place
ls -la  # Should see: scripts/, templates/, batch_jobs/, etc.
```

#### Timeout Issues

**Problem:** Large batches don't complete within the default 1-hour timeout.

**Solution 1 - Use immediate mode for faster processing:**
```bash
# Process via Responses API (faster, but more expensive)
qsp-extract input.csv \
  --type parameter \
  --immediate
```

The `--immediate` flag bypasses the batch API and processes requests immediately via the Responses API. This is much faster (minutes instead of hours) but costs more. **Good for testing or small batches.**

**Solution 2 - Increase timeout for batch API:**
```bash
# Increase timeout to 2 hours (7200 seconds)
qsp-extract input.csv \
  --type parameter \
  --timeout 7200

# For very large batches, try 4 hours
qsp-extract input.csv \
  --type parameter \
  --timeout 14400
```

**Note:** OpenAI's batch API can take up to 24 hours for large batches, but typically completes in 1-2 hours. The batch API is 50% cheaper than the Responses API, so use it for large production batches.

#### Validation Errors

**Problem:** You ran validation and found errors in the extracted files.

**Solution:**
1. Review the validation reports in `output/validation/` to understand what failed
2. For files with errors, you can either:
   - Fix them manually
   - Use the validation fix workflow: `qsp-fix <workflow_type>`
   - Delete and re-extract problematic files

See CLAUDE.md for details on the validation fix workflow.

### Getting Help

If you're still stuck:

1. **Check this documentation** - Most common issues are covered above
2. **Ask a labmate** - Someone else may have encountered the same issue
3. **Check the error message carefully** - Often it tells you exactly what's wrong
4. **For API key issues** - Follow your institution's API key provisioning guide (see Step 5 above)
5. **For repository access** - Ask your PI or lab manager for GitHub access

### Common Command Reference

Here's a quick reference of commands you'll use frequently:

```bash
# Navigate to workflows directory
cd ~/Projects/qsp-llm-workflows

# Activate virtual environment (do this every time!)
source venv/bin/activate

# Check if a file exists
ls parameter_input.csv

# Run parameter extraction
qsp-extract \
  batch_jobs/input_data/parameter_input.csv \
  --type parameter

# Check git status
git status

# Switch to metadata storage and checkout review branch
cd ../qsp-metadata-storage
git checkout review/batch-parameter-2025-10-27-abc123

# List files in to-review/
ls to-review/

# Go back to workflows directory
cd ../qsp-llm-workflows
```

