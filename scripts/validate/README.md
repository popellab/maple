# Validation Scripts

Core automated validation tools for LLM-extracted metadata (parameters, test statistics, quick estimates).

## Quick Start

Run all 6 core validations:

```bash
python scripts/validate/run_all_validations.py \
  ../qsp-metadata-storage/parameter_estimates \
  templates/parameter_metadata_template.yaml \
  output/validation_results/
```

## Core Validation Suite (6 Checks)

### 1. Template Compliance

Validates YAML files conform to template schema.

**Checks:**
- All required fields present
- Correct field types
- Numeric values valid
- Validation weights in [0, 1]

**Works for:** Parameters, test statistics, quick estimates

**Usage:**
```bash
python scripts/validate/check_schema_compliance.py \
  ../qsp-metadata-storage/parameter_estimates \
  templates/parameter_metadata_template.yaml \
  output/schema_compliance.json
```

### 2. Code Execution

Tests that Python derivation code executes correctly.

**Validates:**
- Code executes without errors
- Function returns required fields (mean, variance, ci95)
- Computed values match declared values in YAML (within 5% threshold)

**Works for:** Parameters (v3), test statistics (v2)

**Usage:**
```bash
python scripts/validate/test_code_execution.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/code_execution.json \
  --threshold 5.0
```

**Note:** R code support has been deprecated. All new extractions use Python.

### 3. Text Snippet Validation

Validates that text snippets contain their declared values.

**Checks:**
- value_snippet contains the declared value
- units_snippet contains the declared value
- Handles multiple formats (scientific notation, percentages, decimals)

**Works for:** Parameters, test statistics

**Usage:**
```bash
python scripts/validate/check_text_snippets.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/text_snippets.json
```

### 4. Source Reference Validation

Validates source reference integrity.

**Checks:**
- Every non-null source_ref has a matching source definition
- All sources have required fields (title, first_author, year, doi)
- No orphaned references

**Works for:** Parameters, test statistics

**Usage:**
```bash
python scripts/validate/check_source_references.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/source_references.json
```

### 5. DOI Resolution

Validates DOIs resolve and metadata matches.

**Checks:**
- DOIs resolve via CrossRef API
- Returned metadata matches YAML (title, author, year)
- Uses rate limiting (1 request/second)

**Works for:** Parameters, test statistics

**Usage:**
```bash
python scripts/validate/check_doi_validity.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/doi_validity.json \
  --rate-limit 1.0
```

### 6. Value Consistency

Compares new extractions against existing data.

**Compares against:**
- Legacy database values (from `{type}_legacy/` directory)
- Other derivations with same context_hash

**Reports warnings for:**
- Values >20% different from legacy
- Values outside range of same-context derivations
- Values >50% different from same-context mean

**Works for:** Parameters, test statistics

**Usage:**
```bash
python scripts/validate/check_value_consistency.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/value_consistency.json
```

**Note:** This validator automatically discovers the corresponding legacy directory:
- `parameter_estimates/` → `parameter_estimates_legacy/`
- `test_statistics/` → `test_statistics_legacy/`
- `quick_estimates/` → `quick_estimates_legacy/`

## Legacy Directory Structure

Legacy parameters are stored in separate directories to keep them distinct from new extractions:

```
qsp-metadata-storage/
├── parameter_estimates/          # New parameter extractions
├── parameter_estimates_legacy/   # Legacy parameter database
├── test_statistics/              # New test statistic extractions
├── test_statistics_legacy/       # Legacy test statistics (if any)
├── quick_estimates/              # Quick parameter estimates
└── quick_estimates_legacy/       # Legacy quick estimates (if any)
```

**Validation behavior:**
- Legacy files are **not validated** (they serve as reference data)
- Legacy files are used for **comparison** in value consistency checks
- Validators automatically detect and skip legacy directories

## Output Files

Each validation script produces a JSON file with:
- **summary**: Total/passed/failed counts, pass rate
- **passed**: List of items that passed (with details)
- **failed**: List of items that failed (with error messages)
- **warnings**: List of items with warnings (for consistency check)

### Master Summary

`run_all_validations.py` produces `master_validation_summary.json` with aggregated results from all 6 validators.

## Validation Reports

Reports include:
- **Total**: Number of items validated
- **Passed**: Number passing validation
- **Failed**: Number failing validation
- **Pass rate**: Percentage passing

Failed items include detailed error messages for debugging.

## Dependencies

**Python packages:**
- `pyyaml` - YAML parsing
- `numpy` - Numerical computations
- `requests` - DOI resolution

Install dependencies:
```bash
pip install pyyaml numpy requests
```

Or use the project's virtual environment:
```bash
source venv/bin/activate  # Assuming venv is set up
```

## Utilities Module

`validation_utils.py` provides shared functionality:
- `load_yaml_file()`: Load single YAML file
- `load_yaml_directory()`: Load all YAMLs from directory
- `parse_numeric_value()`: Robust numeric parsing
- `ValidationReport`: Container for validation results with summary statistics

## Integration with Automated Workflow

These validators are automatically run as part of the extraction workflow:

1. **After unpacking**: Results are unpacked to `to-review/` subdirectories
2. **Before git commit**: All 6 validators run on unpacked files
3. **In commit message**: Validation summary included for visibility

See `scripts/lib/validation_runner.py` for integration details.

## Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

**Note:** Value consistency check (validator #6) always passes - it only reports warnings, never fails.

## Design Philosophy

These validators enforce **data quality at ingestion time**:

1. **Correctness**: Code executes, values are accurate
2. **Traceability**: Sources are valid, snippets contain declared values
3. **Completeness**: All required fields present
4. **Consistency**: Values align with existing data

They do **not** include paper-specific validations (those belong in paper repositories).
