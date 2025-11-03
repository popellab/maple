# Manual Review Checklist

This checklist covers validation items requiring human judgment. Focus on derivation explanations and key assumptions rather than mechanical checks (which are automated).

## Primary Review Focus

### 1. Derivation Explanation Quality

**For parameters:**
- [ ] `derivation_explanation` clearly describes how the value was obtained
- [ ] Mathematical transformations are explained and justified
- [ ] Any conversions or calculations are transparent and traceable
- [ ] Reasoning connects the source data to the final parameter value

**For test statistics:**
- [ ] `derivation_explanation` describes the statistical methodology
- [ ] Bootstrap approach is appropriate for the data type
- [ ] Uncertainty quantification method matches the paper's approach
- [ ] Any data transformations are explained

**Red flags:**
- Vague explanations like "extracted from paper" or "calculated from data"
- Missing justification for unit conversions or transformations
- Unclear connection between source text and derived value
- Unexplained statistical methods

### 2. Key Assumptions

**Review assumptions section for:**
- [ ] All major assumptions are explicitly stated
- [ ] Biological/clinical assumptions are scientifically sound
- [ ] Statistical assumptions are appropriate for the data
- [ ] Model-specific assumptions align with QSP model structure
- [ ] Simplifications are reasonable and documented

**Examples of good assumptions:**
- "Assumed linear dose-response relationship based on Fig 2A"
- "Used xenograft data as proxy for human values (no human data available)"
- "Assumed steady-state conditions after 7-day treatment period"
- "Pooled male and female data (paper reported no significant sex differences)"

**Red flags:**
- No assumptions listed (very few extractions have zero assumptions)
- Assumptions that contradict paper content
- Unjustified simplifications (e.g., "assumed normal distribution" without basis)
- Missing critical context assumptions

## Secondary Review Items

### Source Quality
- [ ] Primary source is appropriate for the extraction
- [ ] Text snippets provide adequate context
- [ ] No apparent misinterpretation of paper content

### Biological/Clinical Plausibility
- [ ] Value is in reasonable range for the biological parameter
- [ ] Units are appropriate and correctly applied
- [ ] Species and experimental context are correctly identified

## Approval Criteria

A file is ready to move from `to-review/` to production if:

1. ✅ All automated validation checks pass
2. ✅ Derivation explanation is clear and complete
3. ✅ Key assumptions are explicitly stated and justified
4. ✅ Biological/clinical context is correctly captured
5. ✅ At least one domain expert has reviewed the extraction

**Document any concerns** in YAML comments for future reference.

## Common Clarification Needs

Files often need clarification on:

- **Unit conversions**: "How was µM converted to molecules/cell?"
- **Time points**: "Which measurement time was used (24h vs 48h vs steady-state)?"
- **Statistical pooling**: "How were multiple measurements combined?"
- **Species translation**: "How was mouse dose scaled to human equivalent?"
- **Model alignment**: "Why was clearance rate converted to half-life for model input?"

These should be addressed in `derivation_explanation` and `key_assumptions`.
