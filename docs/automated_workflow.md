# Automated Extraction Workflow

## Overview

The automated extraction workflow reduces the manual friction of running parameter/test statistic extractions by handling the complete pipeline in a single command:

1. **Create batch** → 2. **Upload** → 3. **Monitor** → 4. **Validate** → 5. **Unpack** → 6. **Commit & Push**

Instead of running 6+ manual steps, you run one command and get a git branch ready for review.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run parameter extraction workflow
python scripts/run_extraction_workflow.py input.csv --type parameter

# The script will:
# - Create and upload batch requests
# - Poll until completion (shows progress)
# - Run automatic validation
# - Unpack validated results to ../qsp-metadata-storage/to-review/
# - Create review branch and push to remote
```

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
  batch_jobs/input_data/core_extraction_input.csv \
  --type parameter

# With longer timeout for large batches
python scripts/run_extraction_workflow.py \
  batch_jobs/input_data/large_batch.csv \
  --type parameter \
  --timeout 7200
```

### Test Statistics

```bash
# Test statistic extraction
python scripts/run_extraction_workflow.py \
  batch_jobs/pdac_pretreatment_metrics.csv \
  --type test_statistic
```

### Quick Estimates

```bash
# Quick estimates without validation
python scripts/run_extraction_workflow.py \
  batch_jobs/input_data/quick_params.csv \
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

### "OPENAI_API_KEY not found"

Ensure `.env` file exists with your API key:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### "qsp-metadata-storage not found"

The script expects the metadata storage repo as a sibling directory:

```
Projects/
├── qsp-llm-workflows/     # This repo
└── qsp-metadata-storage/  # Metadata storage (must exist)
```

### Timeout Issues

Large batches may need longer timeouts:

```bash
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --timeout 7200  # 2 hours
```

### Validation Failures

Skip validation if you want to review raw extractions:

```bash
python scripts/run_extraction_workflow.py input.csv \
  --type parameter \
  --skip-validation
```

Then review and manually validate files before moving to final locations.

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
