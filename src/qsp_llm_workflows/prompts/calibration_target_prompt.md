# Calibration Target Extraction

Extract a **raw observable** from scientific literature for QSP model calibration.

**Cancer type:** {{CANCER_TYPE}}

## Observable to Find

{{OBSERVABLE_DESCRIPTION}}

## Model Context (for reference)

{{MODEL_CONTEXT}}

## Task

1. Search peer-reviewed literature for this observable in {{CANCER_TYPE}} or related contexts
2. Extract the reported value with uncertainty (SD, SE, 95% CI, IQR, or range)
3. Document the experimental context (species, indication, compartment, system, treatment history, stage)
4. Provide **verbatim snippets** from the paper containing the extracted value

## Key Requirements

- **Real sources only** - do not fabricate data or citations
- **Verbatim text** - value_snippet must be exact quoted text from the paper
- **Pint-parseable units** - e.g., "cell/mm^2", "pg/mL", "1/day", "dimensionless"
- **Source traceability** - source_ref must reference a primary_data_sources entry

{{SOURCE_AND_VALIDATION_RUBRICS}}
