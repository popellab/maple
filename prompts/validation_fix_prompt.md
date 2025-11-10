You are helping to fix validation errors in a QSP metadata file.

**IMPORTANT:** Header fields (schema_version, cancer_type, tags, context_hash, etc.) are preserved separately and will be added back automatically. You only need to fix the scientific content fields shown below.

**Original Content (headers preserved separately):**
```yaml
{{YAML_CONTENT}}
```

**Template for Reference (content fields only):**
```yaml
{{TEMPLATE_CONTENT}}
```

## Task

Fix the validation errors listed below, and while doing so:
- **Review ALL units_snippets** (even if not flagged) to ensure they mention or imply the units
- Preserve all other content exactly as-is, including:
  - All data sources and references
  - All derivation code and quality metrics
  - All existing text snippets that are correct

## Guidelines by Error Type

### Schema Compliance Errors
- Add missing required fields using the template as reference
- Ensure field types match template (string, number, list, dict)
- If adding new fields, use appropriate null values or infer from context

### Code Execution Errors
- **Verify the derivation code makes sense** for the parameter being estimated (correct formula, appropriate statistical method)
- **Run the code mentally** to check for calculation errors or incorrect variable references
- **Ensure reported values exactly match calculated values** - if there's a mismatch, investigate:
  - Is the code wrong? Fix the code
  - Is the reported value wrong? Fix the reported value
  - Are the input values inconsistent? Fix the inputs
- Ensure all required outputs are generated (mean, variance, median, etc.)
- The goal is **exact numerical consistency** between code output and reported values

### Text Snippet Errors
- **Look up the actual source** document (DOI or citation) to find the text
- Extract text snippets **verbatim** from the source - do not paraphrase or modify
- For value_snippet: find the exact sentence/phrase containing the numeric value
- For units_snippet: find the exact text that mentions the units or measurement context
- Verify the snippet is from the correct context (right experiment, condition, species)
- Values may appear in different formats in text (scientific notation, percentages, ranges, etc.)
- For dimensionless quantities, units_snippet should contain measurement context rather than the word "dimensionless"

### Source Reference Errors
- Ensure every input has a valid source_ref field
- source_ref should reference a key in data_sources or methodological_sources
- Add missing source entries if needed, using "INFERRED" or similar placeholders

### DOI Resolution Errors
- Verify the DOI points to a real, published source
- Check that citation fields (first_author, year, journal, title) match the DOI
- Ensure the reference makes sense for the data it's linked to (right topic, species, context)
- Verify DOI format is correct (should be like "10.1234/journal.2023.12345")
- If DOI cannot be found or verified, use your knowledge to find the correct DOI or mark as unavailable

### Value Consistency Errors
- Check if reported values match expected ranges
- Compare against legacy data if mentioned
- Document any intentional differences in notes

---

## Validation Errors to Fix

{{VALIDATION_ERRORS}}

---

## Output Format

Return the corrected content as JSON inside a code fence. Requirements:
- **Use JSON format** (the unpacker will convert to YAML and add headers automatically)
- **Include** a \`\`\`json code fence around your response
- **Do NOT include** explanations outside the code fence
- **Do NOT include header fields** (schema_version, cancer_type, tags, context_hash, etc.) - these are preserved from the original and added back automatically
- **CRITICAL STRUCTURE RULE**: Match the EXACT structure shown in "Original Content" above
  - If `model_output` is at the root level, keep it at root level
  - If `test_statistic_definition` is at the root level, keep it at root level
  - Do NOT nest root-level fields under `model_output`
  - For test statistics: `model_output` should ONLY contain `code`, all other fields (test_statistic_definition, study_overview, test_statistic_estimates, primary_data_sources, etc.) are separate root-level fields
- **Fix** only the specific errors mentioned while preserving the original structure

Example output format for test statistics:
\`\`\`json
{
  "model_output": {
    "code": "import numpy as np\n\ndef compute_test_statistic(...):\n    ..."
  },
  "test_statistic_definition": "...",
  "study_overview": "...",
  "study_design": "...",
  "test_statistic_estimates": {
    "inputs": [...],
    "derivation_code": "...",
    "median": 1.23,
    "iqr": 0.45,
    "ci95": [0.5, 2.0],
    "units": "...",
    "key_assumptions": {
      "1": "...",
      "2": "...",
      "3": "..."
    }
  },
  "derivation_explanation": "...",
  "key_study_limitations": "...",
  "primary_data_sources": [...],
  "secondary_data_sources": [...],
  "methodological_sources": [...],
  "validation_weights": {
    "species_match": {"value": 1.0, "justification": "..."},
    ...
  }
}
\`\`\`
