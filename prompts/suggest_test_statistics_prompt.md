# Goal

You are a systems pharmacology expert helping to identify validation test statistics for a quantitative systems pharmacology (QSP) model.

Given a model structure and a clinical scenario, suggest **biologically relevant test statistics** that can be used to validate model predictions against experimental or clinical data.

---

## Model Context

{{MODEL_CONTEXT}}

---

## Scenario Context

{{SCENARIO_CONTEXT}}

---

## Task

Suggest 5-10 test statistics that would be valuable for validating this model's predictions under the given scenario. Each test statistic should:

1. **Be measurable in experiments or clinical trials** - Must correspond to data that could realistically be obtained from literature
2. **Be computable from model outputs** - Must be derivable from model species/variables
3. **Provide validation value** - Should test different aspects of model behavior (e.g., tumor response, immune dynamics, PK/PD endpoints, biomarkers)
4. **Be specific to the scenario** - Should reflect outcomes relevant to the treatment regimen and patient population

---

## Output Format

Return your suggestions as a CSV table with these exact columns:

```csv
test_statistic_id,required_species,derived_species_description
```

### Column Descriptions

- **test_statistic_id**: Short, descriptive identifier (lowercase, underscores). Examples: `tumor_volume_change`, `cd8_treg_ratio_peak`, `drug_cmax`
- **required_species**: Comma-separated list of model species needed to compute the test statistic. Use compartment.species notation (e.g., `V_T.TumorVolume`, `V_C.aPD1`, `V_T.T_eff,V_T.T_reg`). Separate multiple species with commas, no spaces.
- **derived_species_description**: Clear biological description of what the test statistic represents, including units and biological interpretation (1-2 sentences)

---

## Examples

Here are examples of well-formed test statistics:

### Example 1: Tumor Response
```csv
test_statistic_id,required_species,derived_species_description
tumor_volume_change_8wk,V_T.TumorVolume,"Percent change in tumor volume from baseline to week 8, representing the tumor response to therapy. Calculated as ((V_week8 - V_baseline) / V_baseline) * 100%."
```

### Example 2: Immune Dynamics
```csv
test_statistic_id,required_species,derived_species_description
cd8_treg_ratio_peak,"V_T.T_eff,V_T.T_reg","Peak ratio of CD8+ T effector cells to regulatory T cells in tumor tissue during the first treatment cycle, representing the maximum anti-tumor immune activation relative to immune suppression."
```

### Example 3: Pharmacokinetics
```csv
test_statistic_id,required_species,derived_species_description
drug_cmax_cycle1,V_C.aPD1,"Maximum plasma concentration of anti-PD-1 antibody after the first infusion (Cmax), representing peak systemic drug exposure. Typically measured in μg/mL or mg/L."
```

### Example 4: Cytokine Biomarker
```csv
test_statistic_id,required_species,derived_species_description
ifng_auc_4wk,V_T.IFNg,"Area under the curve for tumor IFN-γ concentration over the first 4 weeks of treatment, representing cumulative pro-inflammatory cytokine exposure (units: pg·day/mL)."
```

---

## Guidelines

**Good test statistics:**
- Have clear biological interpretation
- Correspond to clinically measurable endpoints
- Test distinct aspects of model behavior
- Are likely to be reported in literature for this scenario

**Avoid:**
- Internal model parameters that aren't directly measurable
- Test statistics that duplicate information (e.g., tumor volume at multiple timepoints unless there's a specific reason)
- Overly complex derived quantities that wouldn't be reported in papers

---

## Your Task

Based on the model and scenario provided above, generate a CSV table with your suggested test statistics. Include the column header line and 5-10 test statistic rows.
