You are helping to fix validation errors in a quantitative systems pharmacology (QSP) metadata file.

**Original Content:**
```yaml
{{YAML_CONTENT}}
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
Text snippets are automatically verified against the full paper text. When fixing snippet errors:

**Core Rules:**
1. **VERBATIM only**: Copy exact text from the paper. Never paraphrase, summarize, or reconstruct.
2. **No table reconstruction**: Do NOT create artificial table notation like `CD8^{+} | ... | 17 (9-30)`. Tables are flattened when we extract text, so this format won't match.
3. **Use continuous text spans**: Find a short, continuous phrase that contains the value. For table data, the snippet should be just the cell value and any immediately adjacent text, e.g., `"17 (9-30)"` not a reconstructed row.
4. **Include context when helpful**: A few surrounding words help locate the snippet, e.g., `"median survival of 18.2 months"` is better than just `"18.2"`.
5. **Avoid LaTeX formatting**: Write `CD8+` not `CD8^{+}`. Write subscripts inline: `CO2` not `CO_{2}`.
6. **Keep snippets short**: 5-50 words is ideal. Long snippets are harder to match exactly.

**Additional guidance:**
- **Look up the actual source** document (DOI or citation) to find the text
- For value_snippet: find the exact sentence/phrase containing the numeric value
- For units_snippet: find where units are explicitly stated, e.g., `"expressed as cells per high-power field"`
- Verify the snippet is from the correct context (right experiment, condition, species)
- For dimensionless quantities, units_snippet should contain measurement context rather than the word "dimensionless"

**Good snippet examples:**
- `"median CD8+ density was 17 (IQR 9-30) cells/HPF"` ✓
- `"n = 137 patients"` ✓

**Bad snippet examples:**
- `"CD8^{+} | No neoadjuvant | 17 (9-30)"` ✗ (reconstructed table, LaTeX)
- `"The study found elevated levels"` ✗ (no actual value)

### Source Reference Errors
- Ensure every input has a valid source_ref field
- source_ref should reference a key in primary_data_sources or secondary_data_sources
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
