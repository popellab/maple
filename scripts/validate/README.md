# Validation Scripts

Core automated validation tools for LLM-extracted parameter metadata.

## Quick Start

Run all core validations:

```bash
python scripts/validate/run_all_validations.py \
  ../qsp-metadata-storage/parameter_estimates \
  templates/parameter_metadata_template_v2.yaml \
  output/validation_results/
```

## Individual Validation Scripts

### 1. Legacy Parameter Comparison

Compare LLM extractions to legacy database (both in v2 format).

**Metrics:**
- Absolute % difference
- Agreement within confidence intervals
- Pearson and Spearman correlation

**Usage:**
```bash
python scripts/validate/compare_to_legacy.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/legacy_comparison.json
```

**Note:** Legacy files are identified by `_legacy` suffix in filename.

### 2. Template Compliance Validation

Validate YAML files conform to template schema.

**Checks:**
- All required fields present
- Correct field types
- Numeric values valid
- Pooling weights in [0, 1]

**Usage:**
```bash
python scripts/validate/check_schema_compliance.py \
  ../qsp-metadata-storage/parameter_estimates \
  templates/parameter_metadata_template_v2.yaml \
  output/schema_compliance.json
```

### 3. R Code Execution Testing

Test that R bootstrap code executes without errors.

**Validates:**
- Code executes successfully
- Required variables created (mc_draws, mean, variance, ci95)
- No runtime errors

**Usage:**
```bash
python scripts/validate/test_r_code_execution.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/r_execution.json
```

**Requirements:** R must be installed and `Rscript` available in PATH.

### 4. R Code Reproducibility Testing

Test that R bootstrap code produces consistent results across runs with different random seeds.

**Validates:**
- Code produces consistent distributions with different seeds
- Coefficient of variation (CV) of mean estimates is low (<5%)
- Bootstrap sampling is working correctly

**Usage:**
```bash
python scripts/validate/test_r_code_reproducibility.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/r_reproducibility.json \
  --n-runs 5 \
  --cv-threshold 5.0
```

**Note:** Removes `set.seed()` calls from R code and runs with different seeds to test true reproducibility.

**Requirements:** R must be installed and `Rscript` available in PATH.

### 5. De Novo Consistency Analysis

Analyze consistency across multiple independent extractions of the same parameter.

**Metrics:**
- Coefficient of variation (CV%)
- Mean absolute deviation
- Confidence interval overlap rate

**Usage:**
```bash
python scripts/validate/check_denovo_consistency.py \
  ../qsp-metadata-storage/parameter_estimates \
  output/consistency.json \
  --cv-threshold 50
```

**Note:** Only analyzes parameters with multiple extractions (non-legacy files).

## Output Files

Each validation script produces two JSON files:

1. **Detailed results** (e.g., `legacy_comparison.json`): Complete data for all comparisons
2. **Summary report** (e.g., `legacy_comparison_summary.json`): Pass/fail summary with statistics

### Master Summary

`run_all_validations.py` produces `master_validation_summary.json` with aggregated results from all validators.

## Validation Reports

Reports include:
- **Total**: Number of items validated
- **Passed**: Number passing validation
- **Failed**: Number failing validation
- **Pass rate**: Percentage passing

Failed items include detailed error messages.

## Dependencies

**Python packages:**
- `pyyaml`
- `numpy`
- `scipy`
- `pandas` (optional, for biological plausibility checker)

**External tools:**
- R (for code execution validator)

Install Python dependencies:
```bash
pip install pyyaml numpy scipy pandas
```

## Utilities Module

`validation_utils.py` provides shared functionality:
- `load_yaml_file()`: Load single YAML file
- `load_yaml_directory()`: Load all YAMLs from directory
- `extract_parameter_name_from_filename()`: Parse parameter name
- `parse_numeric_value()`: Robust numeric parsing
- `ValidationReport`: Container for validation results

## Skipping Validations

**Skip R execution** (if R not installed):
```bash
python scripts/validate/run_all_validations.py \
  ../qsp-metadata-storage/parameter_estimates \
  templates/parameter_metadata_template_v2.yaml \
  output/ \
  --skip-r-execution
```

**Skip legacy files:** All validators automatically skip files with `_legacy` in filename.

## Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

## Future Enhancements

**Not yet implemented:**
- Biological plausibility checker (range validation)
- Source verification (DOI validity, PDF extraction)
- Metadata completeness scoring
