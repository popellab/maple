# Scientific Soundness Review Rubric

You are reviewing a QSP parameter extraction or test statistic for scientific soundness. Evaluate the YAML file against each dimension below, scoring as **PASS**, **CONCERN**, or **FAIL**.

This review focuses on scientific judgment that requires domain expertise. Technical checks (code execution, DOI resolution, snippet verification) are handled separately by automated validators.

## Scoring Criteria

- **PASS**: Meets all requirements, no issues
- **CONCERN**: Minor issues that warrant attention but don't invalidate the extraction
- **FAIL**: Critical issues that make the extraction unreliable or unusable

---

## 1. Statistical Methodology

Evaluate whether the uncertainty quantification is appropriate for the data and derivation.

**Check:**
- Is the statistical approach (bootstrap, parametric, etc.) appropriate for the sample size?
- Are distributional assumptions justified by the data?
- Is the derivation hierarchy level appropriate?
  - **direct**: Value taken directly from source
  - **converted**: Unit conversion only
  - **derived**: Calculated from reported values
  - **constructed**: Built from multiple sources with assumptions
  - **assumed**: Expert judgment with no direct data
- Is uncertainty inflated appropriately for lower hierarchy levels?

**FAIL if:**
- Bootstrap used with n < 10 samples
- Parametric assumptions clearly violated
- No uncertainty provided for derived/constructed values
- Derivation hierarchy misclassified

---

## 2. Data Source Appropriateness

Evaluate whether the data source is suitable for the target model.

**Check:**
- Does the source cancer type match the target cancer type?
- Is the experimental system relevant?
  - **in vivo human** > **in vivo mouse** > **ex vivo** > **organoid** > **cell line**
- If cross-indication data is used, is it biologically justified?
- Is the compartment/matrix appropriate for the measurement?

**FAIL if:**
- Cross-indication data used without explicit justification
- Cell line data used where in vivo data is clearly needed
- Wrong compartment (e.g., plasma concentration used for tumor parameter)

**CONCERN if:**
- Cross-indication data with reasonable but unstated rationale
- In vitro data extrapolated to in vivo context

---

## 3. Mechanism-to-Model Alignment

Evaluate whether the biological mechanism in the source matches the model equation.

**Check:**
- Does the measured quantity match what the model parameter represents?
- If using a proxy measurement, is it well-validated?
- Are experimental conditions (dose, timing, stimulus) representative of model context?

**FAIL if:**
- Measured quantity fundamentally different from model parameter
- Proxy used without validation evidence
- Conditions so different that values are not transferable

---

## 4. Assumption Transparency

Evaluate whether assumptions are clearly documented.

**Check:**
- Are key assumptions explicitly listed?
- Are assumptions numbered and referenced in derivation code comments?
- Is uncertainty inflated for each assumption?
- Are cascading assumptions (4+) flagged as high uncertainty?

**FAIL if:**
- Major assumptions not documented
- Assumptions made but uncertainty not increased
- More than 5 unacknowledged assumptions

**CONCERN if:**
- Assumptions documented but not referenced in code
- Uncertainty inflation seems insufficient

---

## 5. Biological Plausibility

Evaluate whether the extracted value makes biological sense.

**Check:**
- Is the value in a reasonable biological range for this parameter type?
- Does the magnitude align with known biology? (e.g., cell doubling times typically 0.5-7 days for cancer cells)
- Are derived quantities internally consistent? (e.g., if growth and death rates are both extracted, does net growth match expected tumor behavior?)
- Does the uncertainty range span biologically meaningful values?

**FAIL if:**
- Value is orders of magnitude outside expected biological range
- Derived quantities are internally contradictory
- Uncertainty range includes physically impossible values (e.g., negative rates)

**CONCERN if:**
- Value is at the edge of expected range without explanation
- Limited biological context provided to justify unusual values

---

## Output Format

Respond with a JSON object:

```json
{
  "overall": "PASS" | "CONCERN" | "FAIL",
  "dimensions": {
    "statistical_methodology": {
      "score": "PASS" | "CONCERN" | "FAIL",
      "reasoning": "Brief explanation"
    },
    "data_source_appropriateness": {
      "score": "PASS" | "CONCERN" | "FAIL",
      "reasoning": "Brief explanation"
    },
    "mechanism_to_model_alignment": {
      "score": "PASS" | "CONCERN" | "FAIL",
      "reasoning": "Brief explanation"
    },
    "assumption_transparency": {
      "score": "PASS" | "CONCERN" | "FAIL",
      "reasoning": "Brief explanation"
    },
    "biological_plausibility": {
      "score": "PASS" | "CONCERN" | "FAIL",
      "reasoning": "Brief explanation"
    }
  },
  "critical_issues": [
    "Issue 1 description",
    "Issue 2 description"
  ],
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ]
}
```

**Overall score logic:**
- **FAIL** if any dimension is FAIL
- **CONCERN** if any dimension is CONCERN (and none are FAIL)
- **PASS** if all dimensions are PASS
