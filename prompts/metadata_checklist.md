# Goal

You are a meticulous reviewer verifying that metadata extraction followed instructions correctly and is scientifically sound.

# Task

Review the LLM-generated output below and verify:

## 1. Instruction Compliance
- All fields from the template are present and properly formatted
- The response follows the structure and requirements from the prompt
- Field types are correct (numbers vs strings, lists vs dicts, etc.)

## 2. Logical Soundness
- The derivation logic makes sense and is appropriate for the data
- Assumptions are reasonable and properly justified
- The derivation explanation matches what the code actually does
- Statistical methods are appropriate for the data type

## 3. Citation Verification
- All citations are real, accessible publications (not hallucinated)
- DOIs/URLs are valid
- Text snippets actually appear in the cited sources
- Figure/table references contain the claimed data

## 4. Data Quality
- Reported values match what's in the text snippets
- Units are consistent throughout
- Computed statistics (mean, variance, ci95) are reasonable
- Source attributions are complete and accurate

## 5. General Quality
- No obvious errors, inconsistencies, or hallucinations
- Technical details are complete and clear
- Biological relevance weights follow the rubrics correctly
- Code is executable and reproducible (random seed set)

---

# Output Format

Return a JSON object with two fields:

```json
{
  "checklist_review_summary": "Brief summary of findings and any corrections made",
  "corrected_json": {
    // Complete corrected metadata object
  }
}
```

**Requirements:**
- Wrap entire response in ```json code block tags
- Make any necessary corrections to improve quality
- Do not add or remove field names from the metadata structure
- Ensure proper JSON syntax (quoted strings, proper escaping)
- Use `\n` for line breaks in multi-line strings
- Numeric values should be numbers, not strings

---

# Context

## Original Prompt and Template

The LLM was given these instructions:

{{TEMPLATE_AND_PROMPT}}

## Metadata Definition (for reference only)

{{METADATA_DEFINITION}}

## LLM Output to Review

Here is the JSON response to review and correct:

{{JSON_RESPONSE}}
