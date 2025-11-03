# Manual Review Checklist

This checklist covers validation items that require human judgment and cannot be easily automated.

## For All Metadata Files

### Source Quality
- [ ] Primary source is appropriate (peer-reviewed publication, clinical trial data, etc.)
- [ ] Citation format is correct and complete
- [ ] DOI resolves to the correct paper (automated check helps, but verify correctness)

### Text Snippets
- [ ] `value_snippet` contains the declared numerical value
- [ ] `units_snippet` mentions or implies the declared units (may not match exactly - e.g., "mg/kg" vs "milligrams per kilogram")
- [ ] Snippets provide sufficient context to understand what was measured
- [ ] No hallucinated or misinterpreted text from the paper

### Semantic Correctness
- [ ] Extracted values match the biological/clinical context
- [ ] Parameter/test statistic names match what was actually measured
- [ ] Species and cancer types are correctly identified
- [ ] Experimental conditions are accurately captured

## Parameter Estimates Specific

### Parameter Values
- [ ] Value is in the correct units (matches template/model requirements)
- [ ] Order of magnitude is plausible for the biological parameter
- [ ] Uncertainty/range is reasonable (if provided)
- [ ] Statistical distribution choice is appropriate (normal, lognormal, etc.)

### Model Context
- [ ] Parameter is relevant to the QSP model
- [ ] Extraction aligns with how parameter is used in model equations

## Test Statistics Specific

### R Code Quality
- [ ] Bootstrap code executes without errors (automated check)
- [ ] Statistical approach is appropriate for the data type
- [ ] Uncertainty quantification method is sound
- [ ] Code matches the described methodology in the paper

### Derived Values
- [ ] Calculated mean/median matches reported values (automated check)
- [ ] Bootstrap distribution shape is reasonable
- [ ] Statistical assumptions are justified

## Common Issues to Watch For

### Units Mismatches
- Dimensionless quantities incorrectly given units
- Unit conversions applied incorrectly
- Percentages vs fractions (0.28 vs 28%)
- Concentration units (M vs mM vs μM vs mg/mL, etc.)

### Context Confusion
- Values extracted from wrong experimental condition
- Species mixup (mouse vs human vs xenograft)
- Treatment vs control values confused
- Time points misidentified

### Metadata Errors
- Incorrect author/year in filename
- Cancer type doesn't match paper content
- Hash collisions (same hash for different contexts)

## Approval Criteria

A file is ready to merge from `to-review/` to production if:
1. All automated validation checks pass
2. Manual review checklist items are verified
3. At least one domain expert has reviewed the extraction
4. Any concerns or uncertainties are documented in comments
