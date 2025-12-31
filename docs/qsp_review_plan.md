# Plan: Scientific Review CLI Command (`qsp-review`)

## Overview

Create a new CLI command that uses Claude Code in headless mode to:
1. Review parameter/test statistic extractions for scientific soundness against a rubric
2. Run automated validation and intelligently address failures
3. Recommend prompt improvements based on patterns observed

This provides an intelligent, Claude-powered review process for extraction quality assurance.

## User Decisions

- **Command name**: `qsp-review`
- **Pass criteria**: Strict (FAILs block proceeding; CONCERNs require explicit user confirmation with default=No)
- **Recommendations output**: Both console AND saved to `review_recommendations.md`

## Architecture

### Command: `qsp-review`

```bash
qsp-review parameter_estimates --file path/to/extraction.yaml
qsp-review test_statistics --file path/to/extraction.yaml
```

Interactive, one file at a time. Uses Claude Code headless mode (`claude -p "..." --allowedTools "Read,Bash" --output-format stream-json`).

### Two-Phase Workflow

**Phase 1: Scientific Soundness Review**
- Claude reads the YAML file
- Evaluates against embedded rubric (see below)
- Outputs: PASS/FAIL + detailed reasoning + specific concerns
- User decides whether to accept or reject

**Phase 2: Prompt Improvement Recommendations**
- Based on issues found in Phase 1
- Claude reads the current prompt to understand existing guidance
- Suggests AT MOST 2 low-complexity changes (1-2 sentences each)
- Outputs to console AND saves to `review_recommendations.md`

Note: Automated validation is handled separately via `qsp-validate`.

## Scientific Soundness Rubric (Embedded)

The rubric covers 3 deep dimensions requiring scientific judgment, scored as PASS/CONCERN/FAIL. Technical checks (code execution, DOI resolution, snippet verification) are handled by automated validators.

### 1. Data Source Appropriateness
- **Cancer type & indication**: Does source match target? Cross-indication justification?
- **Experimental system hierarchy**: Human clinical > ex vivo > humanized mouse > syngeneic > organoid > cell line
- **Species translation**: Interspecies scaling addressed? Known species differences?
- **Patient population**: Disease stage, treatment history, biomarker status match?
- **Compartment & matrix**: Right biological compartment and sample type?
- **Source quality**: Adequately powered? Peer-reviewed? Superseded by newer data?

### 2. Mechanism-to-Model Alignment
- **Quantity matching**: Does measured quantity match model parameter definition?
- **Proxy measurements**: If indirect measurement, is proxy validated? (mRNA→protein, peripheral→tumor, in vitro→in vivo)
- **Temporal dynamics**: Steady-state vs transient? Acute vs chronic? Right time scale?
- **Dose & concentration context**: Physiologically relevant? Right part of dose-response curve?
- **Cell type specificity**: Right cell type isolated? Contamination possible? Cell state considerations?
- **Environmental context**: In vitro vs in vivo TME differences? Oxygen, nutrients, matrix?

### 3. Biological Plausibility
- **Range checking**: Within expected biological range? Compare to literature values
- **Internal consistency**: Makes sense relative to other model parameters?
- **Physiological constraints**: Respects diffusion limits, blood flow limits, metabolic limits?
- **Scale appropriateness**: Molecular vs cell vs tissue level consistent?
- **Uncertainty range**: Spans meaningful values? Excludes impossible values?
- **Literature consensus**: Aligns with or contradicts broader literature?

## File Structure

```
src/qsp_llm_workflows/
├── cli/
│   └── review.py              # New CLI entry point
├── core/
│   └── scientific_review.py   # Review orchestration logic
└── prompts/
    └── scientific_review_rubric.md  # Embedded rubric for Claude
```

Add to `pyproject.toml`:
```toml
qsp-review = "qsp_llm_workflows.cli.review:main"
```

## Implementation Details

### cli/review.py

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("workflow_type", choices=["parameter_estimates", "test_statistics"])
    parser.add_argument("--file", required=True, help="YAML file to review")
    args = parser.parse_args()

    reviewer = ScientificReviewer(args.workflow_type)
    reviewer.review_file(Path(args.file))
```

### core/scientific_review.py

```python
class ScientificReviewer:
    def __init__(self, workflow_type: str):
        self.workflow_type = workflow_type
        self.rubric = self._load_rubric()

    def review_file(self, yaml_path: Path):
        # Phase 1: Scientific soundness
        result = self._run_scientific_review(yaml_path)
        self._display_review(result)

        if not self._user_confirms("Proceed with validation?"):
            return

        # Phase 2: Validation + fix
        validation_result = self._run_validation(yaml_path)
        if validation_result.has_failures:
            self._run_intelligent_fix(yaml_path, validation_result)

        # Phase 3: Prompt recommendations
        recommendations = self._generate_prompt_recommendations(result, validation_result)
        self._display_recommendations(recommendations)

    def _run_scientific_review(self, yaml_path: Path) -> dict:
        # Invoke Claude Code headless
        prompt = self._build_review_prompt(yaml_path)
        result = subprocess.run([
            "claude", "-p", prompt,
            "--allowedTools", "Read",
            "--output-format", "json"
        ], capture_output=True, text=True)
        return json.loads(result.stdout)
```

### Claude Code Invocation Pattern

For Phase 1 (review):
```bash
claude -p "Review this QSP parameter extraction for scientific soundness.

File: {yaml_path}

Evaluate against this rubric:
{rubric_content}

Output your assessment as JSON:
{
  \"overall\": \"PASS\" | \"CONCERN\" | \"FAIL\",
  \"dimensions\": {
    \"statistical_methodology\": {\"score\": \"...\", \"reasoning\": \"...\"},
    ...
  },
  \"critical_issues\": [...],
  \"recommendations\": [...]
}" --allowedTools "Read" --output-format json
```

For Phase 2 (fix):
```bash
claude -p "Fix validation errors in this QSP extraction.

File: {yaml_path}
Validation errors: {validation_json}

Understand WHY each error occurred, then make targeted fixes.
Do not blindly regenerate - preserve what's correct.

After fixing, explain what you changed and why." \
  --allowedTools "Read,Edit" \
  --output-format stream-json
```

For Phase 3 (recommendations):
```bash
claude -p "Based on reviewing this extraction, suggest prompt improvements.

Issues found:
{issues_summary}

Current prompt sections relevant to these issues:
{relevant_prompt_excerpts}

Recommend changes that:
1. Would prevent similar issues in future extractions
2. Are generalizable (not just for this specific parameter)
3. Balance improvement vs. added complexity

Format as actionable suggestions with rationale." \
  --allowedTools "Read" \
  --output-format json
```

## User Interaction Flow

```
$ qsp-review parameter_estimates --file metadata-storage/to-review/k_C_growth_Smith2020_PDAC_abc123.yaml

═══════════════════════════════════════════════════════════════
SCIENTIFIC SOUNDNESS REVIEW
═══════════════════════════════════════════════════════════════

File: k_C_growth_Smith2020_PDAC_abc123.yaml

Reviewing against scientific soundness rubric...

DIMENSION                      SCORE
───────────────────────────────────────────────────────────────
Statistical Methodology        PASS       Bootstrap with 10k samples, appropriate for ratio
Data Source Appropriateness    CONCERN    Cross-indication (CRC→PDAC) without strong justification
Mechanism-to-Model Alignment   PASS       Growth rate measured matches model's k_C_growth
Assumption Transparency        PASS       4 assumptions documented and referenced
Derivation Reproducibility     PASS       Code executes, values match
Source Traceability            CONCERN    One snippet appears paraphrased

OVERALL: CONCERN

Critical Issues:
  1. Cross-indication data from CRC used for PDAC model without biological justification
  2. value_snippet for input "doubling_time" may be paraphrased

───────────────────────────────────────────────────────────────
2 CONCERNs found. Proceed despite concerns? [y/N] y

Running 9 validators...
  ✓ Schema compliance
  ✓ Code execution
  ✗ Text snippets: value_snippet missing exact value
  ✓ Source references
  ✓ DOI validity
  ...

1 validation failure found.

───────────────────────────────────────────────────────────────
Attempt intelligent fix? [Y/n] y

Analyzing failure: Text snippet "tumor doubling time was approximately 2 days"
does not contain exact value "2.1"

Claude's assessment: The snippet uses "approximately 2" but the input value
is "2.1". This is likely because the paper rounds in text but provides
precise value in a figure/table.

Suggested fix: Update value_snippet to include the exact source, or adjust
value to match snippet if "approximately 2" is the only available precision.

Apply this fix? [Y/n]

───────────────────────────────────────────────────────────────
PROMPT IMPROVEMENT RECOMMENDATIONS
───────────────────────────────────────────────────────────────

Based on this review, consider these prompt changes:

1. CROSS-INDICATION JUSTIFICATION (Medium priority, High generalizability)
   Issue: Extraction used CRC data for PDAC without explicit justification
   Suggestion: Add to prompt: "When using cross-indication data, you MUST
   include an assumption explaining the biological rationale for why
   [source indication] data is applicable to [target indication]."
   Complexity: Low (adds ~2 sentences)

2. SNIPPET PRECISION (Low priority, Medium generalizability)
   Issue: Snippet contained rounded value, input had precise value
   Suggestion: Add: "If paper text rounds values but you extract precise
   values from figures/tables, note this in the input description."
   Complexity: Low (adds ~1 sentence)

Recommendations saved to: review_recommendations.md
───────────────────────────────────────────────────────────────
```

## Dependencies

- Claude Code CLI installed and authenticated
- Existing validation infrastructure (`qsp-validate`)
- Read access to YAML files and validation outputs

## Testing

1. Unit tests for `ScientificReviewer` class (mock Claude Code calls)
2. Integration test with a sample YAML file
3. Test each phase independently

## Workflow Integration

`qsp-review` provides quality-focused, interactive review with Claude Code assistance.

Document in automated_workflow.md as the recommended approach for reviewing extractions.
