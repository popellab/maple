# Automated Extraction Workflow

## Overview

This workflow automates LLM-based parameter extraction from scientific papers. Instead of running 8+ manual commands, you run **one command** and the system:

1. Creates batch requests
2. Uploads to OpenAI API
3. Monitors progress automatically
4. Validates the results
5. Unpacks files to a staging area
6. Creates a git branch and pushes for review

**Time savings:** From 20-30 minutes of manual work → 1 minute to launch, then walk away.

### What Happens When You Run a Workflow?

```
┌─────────────────────────────────────────────────────────────┐
│  You create a CSV file with parameters to extract           │
│  cancer_type,parameter_name                                 │
│  PDAC,k_C_growth                                            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  You run ONE command:                                       │
│  python scripts/run_extraction_workflow.py \                │
│    input.csv --type parameter                               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  The system automatically:                                  │
│  1. Creates batch requests and uploads to OpenAI           │
│  2. Monitors progress (shows you updates)                   │
│  3. Downloads results when complete                         │
│  4. Validates the extracted data                            │
│  5. Unpacks files to to-review/ directory                   │
│  6. Creates git branch and pushes for review                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  You review the extracted files and approve/reject them     │
│  cd ../qsp-metadata-storage                                 │
│  git checkout review/batch-parameter-2025-10-27-abc123      │
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
python scripts/run_extraction_workflow.py --help
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

# 3. Run the workflow
python scripts/run_extraction_workflow.py input.csv --type parameter
```

The script will handle everything automatically and show you progress updates.

---

## Preparing Your Input File

The workflow needs a CSV file telling it what to extract. The format depends on which workflow type you're using.

### Parameter Extraction Input File

For parameter extraction (`--type parameter`), your CSV needs these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `cancer_type` | Cancer type being studied | `PDAC`, `Melanoma`, `NSCLC` |
| `parameter_name` | Name of parameter to extract | `k_C_growth`, `K_CD8_TEFF` |
| `notes` (optional) | Additional context for extraction | Parameter description |

**Example:** `parameter_input.csv`

```csv
cancer_type,parameter_name,notes
PDAC,k_C_growth,Cancer cell growth rate
PDAC,k_C_death,Cancer cell death rate
PDAC,k_CD8_act,CD8 activation rate
PDAC,k_APC_mat,Maximum rate of APC maturation
```

**Real example from our PDAC project** (see `scratch/pdac_parameters_modules_v2.csv`):

```csv
cancer_type,parameter_name,notes
PDAC,k_C_CD8,"Rate of cancer cell death by T cells (Robertson-Tessi 2012, PMID: 22051568)"
PDAC,k_CD8_pro,"CD8 proliferation rate (Marchingo 2014, PMID: 25430770)"
PDAC,k_APC_mat,"Maximum rate of APC maturation (Chen 2014, PMID: 25184733)"
PDAC,DAMPs,"Concentration of cytokines released per dying cancer cell (Milo 2013, PMID: 24114984)"
```

**Creating the file:**

Option 1 - Using a text editor:
1. Open your favorite text editor (TextEdit, VS Code, nano, etc.)
2. Type or paste the CSV content exactly as shown above
3. Save as `parameter_input.csv` in the `qsp-llm-workflows` directory

Option 2 - Using Excel/Google Sheets:
1. Create a spreadsheet with column headers in first row: `cancer_type`, `parameter_name`
2. Fill in your data
3. Save/Export as CSV format
4. Move the file to `qsp-llm-workflows` directory

### Test Statistics Input File

For test statistics (`--type test_statistic`), your CSV needs these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `test_statistic_id` | Unique ID for this test stat | `tumor_volume_day14` |
| `cancer_type` | Cancer type | `PDAC` |
| `scenario_context` | Description of experimental scenario | `Untreated PDAC tumor growth` |
| `required_species` | Species needed from model | `V_T.C(t=14)` |
| `derived_species_description` | What the statistic represents | `Tumor volume in mm³ at 14 days` |
| `model_context` (optional) | Model structure details | See full example below |

**Simplified example:** `test_stat_input.csv`

```csv
test_statistic_id,cancer_type,scenario_context,required_species,derived_species_description
tumor_volume_day14,PDAC,Untreated PDAC tumor growth in KPC mice,V_T.C,Tumor volume in mm³ at 14 days
cd8_infiltration,PDAC,Baseline immune infiltration in treatment-naive PDAC,V_T.CD8,CD8+ T cell count per mm³ tumor
cdc1_cdc2_ratio,PDAC,Dendritic cell composition in untreated PDAC tumors,"V_T.cDC1,V_T.cDC2",Ratio of type 1 to type 2 dendritic cells
```

**Note:** For test statistics, the workflow can automatically generate the `model_context` field from your model specification files. If you need custom model context, you can include it as an additional column (see `scratch/test_statistic_input_baseline_no_treatment_24664d08.csv` for a full example with model context).

### Quick Estimates Input File

For quick estimates (`--type quick_estimate`), use the same format as parameter extraction (without notes column):

```csv
cancer_type,parameter_name
PDAC,k_C_growth
PDAC,k_C_death
PDAC,k_CD8_act
PDAC,k_APC_mat
```

Quick estimates are useful for getting rapid ballpark parameter values for model initialization before doing full literature extraction.

### Common Issues

**Problem:** Script says "No such file"
- **Solution:** Make sure you're running the command from the `qsp-llm-workflows` directory
- **Solution:** Check the file path - try `ls parameter_input.csv` to verify it exists

**Problem:** CSV parsing errors
- **Solution:** Make sure column headers are **exactly** as shown (case-sensitive, no spaces)
- **Solution:** Check for hidden characters - save as plain text CSV, not Excel format
- **Solution:** No empty rows at the beginning or end of file

**Problem:** "Required column missing"
- **Solution:** Double-check you have all required columns for your workflow type
- **Solution:** Column names must match exactly (e.g., `cancer_type` not `Cancer Type`)

### Where to Put Your Input Files

You can put input files anywhere, but we recommend:

```bash
qsp-llm-workflows/
├── batch_jobs/
│   └── input_data/          # Put your CSV files here
│       ├── parameter_input.csv
│       ├── test_stat_input.csv
│       └── quick_estimate_input.csv
```

Then reference them in commands:

```bash
python scripts/run_extraction_workflow.py \
  batch_jobs/input_data/parameter_input.csv \
  --type parameter
```

---

## Usage

### Basic Syntax

```bash
python scripts/run_extraction_workflow.py <input.csv> --type <workflow_type> [options]
```

### Workflow Types

- `parameter` - Full parameter extraction with v3 schema
- `test_statistic` - Test statistic extraction with v2 schema
- `quick_estimate` - Quick ballpark estimates

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--timeout SECONDS` | Max wait time for batch completion | 3600 (1 hour) |
| `--skip-validation` | Skip automatic checklist validation | False |
| `--no-push` | Create branch locally without pushing | False |
| `--branch-prefix PREFIX` | Custom prefix for review branches | `review/batch` |

## Examples

### Parameter Extraction

```bash
# Standard parameter extraction
python scripts/run_extraction_workflow.py \
  scratch/pdac_parameters_modules_v2.csv \
  --type parameter

# With longer timeout for large batches (77 parameters in this example)
python scripts/run_extraction_workflow.py \
  scratch/pdac_parameters_modules_v2.csv \
  --type parameter \
  --timeout 7200
```

### Test Statistics

```bash
# Test statistic extraction with full model context
python scripts/run_extraction_workflow.py \
  scratch/test_statistic_input_baseline_no_treatment_24664d08.csv \
  --type test_statistic

# Or use the copy in batch_jobs/
python scripts/run_extraction_workflow.py \
  batch_jobs/input_data/test_statistic_input_baseline_no_treatment_30185312.csv \
  --type test_statistic
```

### Quick Estimates

```bash
# Quick estimates for rapid prototyping
python scripts/run_extraction_workflow.py \
  my_quick_estimates.csv \
  --type quick_estimate \
  --skip-validation
```

### Local-Only Review

```bash
# Create review branch but don't push (review locally first)
python scripts/run_extraction_workflow.py \
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
Running checklist validation...
✓ Batch requests created: checklist_from_json_requests.jsonl
Uploading batch: checklist_from_json_requests.jsonl...
✓ Batch uploaded: batch_DEF456
Monitoring batch batch_DEF456...
  Status: in_progress (8/15 completed)
  Status: in_progress (15/15 completed)
✓ Results downloaded: batch_DEF456_results.jsonl
✓ Validation complete
Unpacking validated results to to-review/...
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
  3. Review files in to-review/
  4. Move approved files to appropriate directories
  5. Merge to main when approved
======================================================================
```

## Review Process

After the workflow completes, review the extracted files:

### 1. Checkout Review Branch

```bash
cd ../qsp-metadata-storage
git checkout review/batch-parameter-2025-10-27-abc123
```

### 2. Review Files

Check files in `to-review/`:
- Verify citations are real and accessible
- Confirm extracted values match sources
- Validate derivation logic and assumptions
- Check unit consistency
- Review validation summary in `scratch/checklist_reviews_*.md`

### 3. Approve/Reject Files

**Approve files** by moving to final location:

```bash
# Parameter estimates
mv to-review/k_C_growth_*.yaml parameter_estimates/

# Test statistics
mv to-review/tumor_volume_*.yaml test_statistics/

# Quick estimates
mv to-review/k_death_*.yaml quick_estimates/
```

**Reject files** by documenting issues and deleting:

```bash
# Document why rejected
echo "k_C_growth_Smith2020: Invalid source citation" >> review_notes.md

# Delete rejected file
rm to-review/k_C_growth_Smith2020_PDAC_abc123.yaml
```

### 4. Commit and Merge

```bash
git add .
git commit -m "Review complete: approved 12 files, rejected 3 files"
git checkout main
git merge review/batch-parameter-2025-10-27-abc123
git push

# Clean up review branch
git branch -d review/batch-parameter-2025-10-27-abc123
git push origin --delete review/batch-parameter-2025-10-27-abc123
```

## Validation

The workflow automatically runs checklist validation to catch common errors:

- **Citation verification**: Checks if sources are real publications
- **Data consistency**: Verifies extracted values match text snippets
- **Code validation**: Ensures derivation code is executable
- **Schema compliance**: Validates all required fields are present
- **Logical soundness**: Checks assumptions and derivation logic

Files that fail validation are still unpacked but marked in the validation summary (`scratch/checklist_reviews_*.md`).

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
- **Validation fails**: LLM extraction had errors → review validation summary
- **Git errors**: Branch already exists or can't push → check git status

## Batch Jobs Directory

Workflow creates/uses these files in `batch_jobs/`:

- `parameter_requests.jsonl` - Extraction batch requests
- `parameter_requests.batch_id` - Batch metadata (ID, type, CSV)
- `batch_ABC123_results.jsonl` - Raw extraction results
- `checklist_from_json_requests.jsonl` - Validation batch requests
- `checklist_from_json_requests.batch_id` - Validation batch metadata
- `batch_DEF456_results.jsonl` - Validation results

These files are gitignored and used for debugging if needed.

## Comparison to Manual Workflow

### Old Manual Workflow (6+ steps)

```bash
# 1. Create batch
python scripts/prepare/create_parameter_batch.py input.csv

# 2. Upload batch
python scripts/run/upload_batch.py batch_jobs/parameter_requests.jsonl

# 3. Monitor batch (manual polling)
python scripts/run/batch_monitor.py batch_ABC123

# 4. Create validation batch
python scripts/prepare/create_checklist_from_json_batch.py \
  batch_jobs/batch_ABC123_results.jsonl input.csv

# 5. Upload validation
python scripts/run/upload_batch.py batch_jobs/checklist_from_json_requests.jsonl

# 6. Monitor validation
python scripts/run/batch_monitor.py batch_DEF456

# 7. Unpack results
python scripts/process/unpack_results.py \
  batch_jobs/batch_DEF456_results.jsonl \
  ../qsp-metadata-storage/parameter_estimates \
  input.csv "" templates/parameter_metadata_template_v3.yaml

# 8. Manual git operations
cd ../qsp-metadata-storage
git checkout -b review-branch
git add parameter_estimates/
git commit -m "Add extractions"
git push -u origin review-branch
```

### New Automated Workflow (1 step)

```bash
python scripts/run_extraction_workflow.py input.csv --type parameter
```

## Future: HPC Integration

The orchestrator is designed to support HPC execution:

```bash
# Submit workflow to HPC queue (future feature)
sbatch scripts/hpc/run_workflow.sh input.csv parameter

# Monitor from local machine
python scripts/hpc/check_workflow_status.py job_12345

# Pull completed review branch when done
cd ../qsp-metadata-storage
git fetch origin
git checkout review/batch-parameter-2025-10-27-abc123
```

HPC integration wraps the same `WorkflowOrchestrator` class used locally.

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

# Try using python3 explicitly in all commands
python3 scripts/run_extraction_workflow.py input.csv --type parameter
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

**Solution:**
```bash
# Increase timeout to 2 hours (7200 seconds)
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --timeout 7200

# For very large batches, try 4 hours
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --timeout 14400
```

Note: OpenAI's batch API can take up to 24 hours for large batches, but typically completes in 1-2 hours.

#### Validation Failures

**Problem:** The workflow reports validation errors and you want to review the raw extractions first.

**Solution:**
```bash
# Skip automatic validation
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --skip-validation
```

Then manually review files in `../qsp-metadata-storage/to-review/` before moving them to final locations.

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
python scripts/run_extraction_workflow.py \
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

## Advanced Usage

### Custom Branch Names

```bash
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --branch-prefix "experiment/batch"

# Creates: experiment/batch-parameter-2025-10-27-abc123
```

### Local Review First

```bash
# Create branch locally without pushing
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --no-push

# Review locally
cd ../qsp-metadata-storage
git checkout review/batch-parameter-2025-10-27-abc123

# Push when ready
git push -u origin review/batch-parameter-2025-10-27-abc123
```

### Programmatic Use

Import the orchestrator in Python scripts:

```python
from pathlib import Path
from scripts.lib.workflow_orchestrator import WorkflowOrchestrator

# Setup
base_dir = Path("/path/to/qsp-llm-workflows")
storage_dir = Path("/path/to/qsp-metadata-storage")
api_key = "sk-..."

orchestrator = WorkflowOrchestrator(base_dir, storage_dir, api_key)

# Run workflow
results = orchestrator.run_complete_workflow(
    input_csv=Path("input.csv"),
    workflow_type="parameter",
    timeout=3600,
    skip_validation=False,
    push=True,
    progress_callback=lambda msg: print(msg)
)

print(f"Extracted {results['file_count']} files")
print(f"Review branch: {results['branch_name']}")
```
