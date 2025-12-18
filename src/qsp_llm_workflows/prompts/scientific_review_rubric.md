# Scientific Soundness Review Rubric

You are reviewing a QSP parameter extraction or test statistic for scientific soundness. Evaluate the YAML file against each dimension below, scoring as **PASS**, **CONCERN**, or **FAIL**.

This review focuses on scientific judgment that requires domain expertise. Technical checks (code execution, DOI resolution, snippet verification) are handled separately by automated validators.

## Scoring Criteria

- **PASS**: Meets all requirements, no issues
- **CONCERN**: Minor issues that warrant attention but don't invalidate the extraction
- **FAIL**: Critical issues that make the extraction unreliable or unusable

---

## 1. Data Source Appropriateness

Evaluate whether the data source is suitable for the target model context.

### Cancer Type & Indication
- Does the source cancer type match the target? Same histological subtype is ideal.
- If cross-indication data is used, is there biological justification? (e.g., shared driver mutations, similar tumor microenvironment)
- Are there known differences between indications that would affect this parameter?

### Experimental System Hierarchy
Prefer data from systems closer to the clinical context:
1. **Human clinical data** (patients, clinical trials)
2. **Human ex vivo** (patient-derived samples, surgical specimens)
3. **Humanized mouse models** (PDX, humanized immune system)
4. **Syngeneic mouse models** (mouse tumors in mice)
5. **Organoids / 3D culture** (patient-derived or cell line)
6. **2D cell lines** (established cancer cell lines)

### Species Translation
- If using animal data for a human model, is interspecies scaling addressed?
- Are there known species differences for this parameter? (e.g., immune cell trafficking, drug metabolism)

### Patient Population
- Does the source population match the model's intended population?
- Consider: disease stage, treatment history (naive vs refractory), age, biomarker status
- Are there subpopulation effects that matter? (e.g., PD-L1 high vs low)

### Compartment & Matrix
- Is the measurement from the right biological compartment? (tumor vs blood vs lymph node)
- Is the matrix appropriate? (plasma vs serum vs whole blood; fresh vs frozen tissue)

### Source Quality
- Is the source study adequately powered for this measurement?
- Is the data from a peer-reviewed publication or preprint/conference abstract?
- Has this measurement been superseded by more recent/definitive studies?

**FAIL if:**
- Cross-indication data used without any biological rationale
- Wrong compartment entirely (e.g., plasma PK used for intratumoral concentration)
- Cell line data used where human clinical data clearly exists and differs significantly
- Major species differences ignored

**CONCERN if:**
- Cross-indication data with reasonable but unstated rationale
- Lower-tier experimental system used when higher-tier data may exist
- Source study is underpowered or preliminary
- Patient population mismatch (e.g., treatment-naive data for refractory setting)

---

## 2. Mechanism-to-Model Alignment

Evaluate whether what was measured in the source study matches what the model parameter represents.

### Quantity Matching
- Does the measured biological quantity correspond to the model parameter's definition?
- If the parameter is a rate constant, was a rate actually measured (not just a static level)?
- If the parameter is a concentration, what form was measured? (total vs free, intracellular vs extracellular)

### Proxy Measurements
If using a proxy (indirect measurement), evaluate:
- Is the proxy well-validated for this target?
- What assumptions link the proxy to the true parameter?
- How much uncertainty does the proxy relationship introduce?

Examples of proxies requiring scrutiny:
- mRNA expression as proxy for protein levels
- Peripheral blood cells as proxy for tumor-infiltrating cells
- In vitro killing rate as proxy for in vivo efficacy
- Ki67 staining as proxy for proliferation rate

### Temporal Dynamics
- Does the measurement time scale match the model's dynamics?
- Steady-state vs transient measurements: which does the model need?
- Acute vs chronic effects: does the source capture the right timeframe?
- Is there time-dependence that the single value doesn't capture?

### Dose & Concentration Context
- Were measurements made at physiologically relevant concentrations?
- If dose-dependent, which part of the dose-response curve applies to the model?
- Are saturation effects relevant? (Michaelis-Menten kinetics, receptor occupancy)

### Cell Type Specificity
- Is the parameter specific to a cell type? Was that cell type isolated/measured?
- Could contaminating cells affect the measurement?
- Are there cell state considerations? (naive vs activated, M1 vs M2 polarization)

### Environmental Context
- In vitro conditions rarely match in vivo tumor microenvironment
- Consider: oxygen levels, nutrient availability, cell density, matrix composition
- Systemic vs local effects: does the source capture the right context?

**FAIL if:**
- Fundamental mismatch between measured quantity and model parameter
- Proxy used without any validation or with known poor correlation
- Measurement conditions so artificial that values are not transferable
- Static measurement used for a highly dynamic process without justification

**CONCERN if:**
- Proxy is reasonable but introduces substantial uncertainty
- Temporal or dose context is imperfect but defensible
- Cell type impurity possible but likely minor effect
- Environmental differences acknowledged but not quantified

---

## 3. Biological Plausibility

Evaluate whether the extracted value makes biological sense in the context of known biology and the model.

### Range Checking
- Is the value within the expected biological range for this parameter type?
- Compare to published literature values for similar parameters
- Consider typical ranges:
  - Cell proliferation rates: 0.1-2 /day for most cancer cells
  - Cell death rates: typically lower than proliferation in growing tumors
  - Diffusion coefficients: constrained by molecular size
  - Binding affinities: typically nM-μM range for relevant interactions

### Internal Consistency
- Does this parameter make sense relative to other parameters in the model?
- If multiple rates determine a net effect, do they combine plausibly?
- Example: if growth rate > death rate, does that match expected tumor behavior?

### Physiological Constraints
Consider hard limits from physics and physiology:
- Diffusion limits for molecular transport
- Blood flow limits for delivery
- Metabolic limits for production rates
- Receptor density limits for binding
- Volume constraints for concentrations

### Scale Appropriateness
- Is the parameter at the right biological scale for the model?
- Molecular-level parameters vs cell-level vs tissue-level: are they consistent?
- Are aggregation/averaging assumptions appropriate?

### Uncertainty Range
- Does the uncertainty span biologically meaningful values?
- Is the uncertainty range realistic given the measurement method?
- Are physically impossible values excluded? (negative rates, concentrations > solubility)

### Literature Consensus
- Does this value align with or contradict the broader literature?
- If it contradicts, is there a good explanation?
- Are there meta-analyses or systematic reviews that provide context?

**FAIL if:**
- Value is orders of magnitude outside expected biological range without explanation
- Violates known physiological constraints
- Uncertainty range includes physically impossible values
- Directly contradicts well-established literature without justification

**CONCERN if:**
- Value is at the edge of expected range without explanation
- Internal consistency is questionable but not clearly wrong
- Limited literature context available for comparison
- Uncertainty range seems too narrow or too wide for the measurement type

---

## Output Format

Respond with a JSON object:

```json
{
  "overall": "PASS" | "CONCERN" | "FAIL",
  "dimensions": {
    "data_source_appropriateness": {
      "score": "PASS" | "CONCERN" | "FAIL",
      "reasoning": "Brief explanation"
    },
    "mechanism_to_model_alignment": {
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
