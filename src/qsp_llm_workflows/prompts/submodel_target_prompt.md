# Submodel Target Extraction

Find experimental data to calibrate the specified model parameters and define a submodel for Bayesian inference.

**Parameters to calibrate:** {{PARAMETERS}}
{{#NOTES}}**Notes:** {{NOTES}}{{/NOTES}}

---

## Model Context

{{MODEL_CONTEXT}}

---

## Parameter Context

The following parameters need to be calibrated. Study their mechanistic usage
in the model BEFORE designing your submodel.

{{PARAMETER_CONTEXT}}

Your submodel MUST replicate how these parameters are used mechanistically,
not just use the same parameter names with different dynamics.

**Using additional parameters:** Your submodel should calibrate the specified
parameters, but you MAY (and often should) include additional model parameters
when mechanistically necessary. All parameters in `calibration.parameters`
will be jointly inferred during Bayesian calibration.

{{USED_PRIMARY_STUDIES}}

---

## Your Task

1. **Understand the parameters** - Review the Parameter Context above
2. **Find relevant experimental data** - Search literature for experiments that constrain these parameters
3. **Extract the data** - Pull quantitative values with full provenance into `inputs`
4. **Build a calibration spec** - Define the model, parameters, and measurements for inference

---

## Schema Overview

```
SubmodelTarget
├── target_id: str
├── inputs: List[Input]                    # Extracted values with provenance
│   ├── name, value, units
│   ├── uncertainty: {ci95: [lo, hi]} | {sd: float}
│   ├── input_type: direct_measurement | proxy_measurement | experimental_condition | inferred_estimate | assumed_value
│   ├── role: initial_condition | target | fixed_parameter | auxiliary
│   ├── extraction_method: manual | webplotdigitizer | digitizer | other
│   └── source_ref, source_location, value_snippet
├── calibration
│   ├── parameters: [{name, units, prior: {distribution, mu, sigma, ...}}]
│   ├── state_variables: [{name, units, initial_condition}]  # For ODE models
│   ├── model: Model                       # See Model Types below
│   ├── independent_variable: {name, units, span}
│   ├── measurements: [{uses_inputs, evaluation_points, sample_size, likelihood}]
│   └── identifiability_notes: str
├── experimental_context: {species, system, cell_lines, cell_types, ...}
├── source_relevance: SourceRelevanceAssessment  # REQUIRED - see below
├── study_interpretation, key_assumptions, key_study_limitations
└── primary_data_source, secondary_data_sources
```

---

## Model Types

### ODE-based (require state_variables, independent_variable with span)

| Type | Equation | Use When |
|------|----------|----------|
| `exponential_growth` | dy/dt = k * y | Unconstrained growth |
| `first_order_decay` | dy/dt = -k * y | Clearance, death, decay |
| `logistic` | dy/dt = k * y * (1 - y/K) | Growth with carrying capacity |
| `michaelis_menten` | dy/dt = -Vmax * y / (Km + y) | Saturable kinetics |
| `two_state` | dA/dt = -k*A, dB/dt = +k*A | State transitions |
| `saturation` | dy/dt = k * (1 - y) | Approach to saturation |
| `custom` | User-defined | Complex dynamics |

### Non-ODE

| Type | Use When |
|------|----------|
| `direct_conversion` | Analytical formula (e.g., k = ln(2) / t_half). **Requires distribution_code** |
| `direct_fit` | Curve fitting (Hill equation for IC50, etc.) |

---

## Input Roles

| Role | Description | Example |
|------|-------------|---------|
| `initial_condition` | Starting value for ODE | Seeding density |
| `target` | Calibration target | Cell count at day 3 |
| `fixed_parameter` | Fixed in model | Known rate from literature |
| `auxiliary` | Supporting data | SD values, sample sizes |

---

## Prior Distributions

| Distribution | Parameters | Use For |
|--------------|------------|---------|
| `lognormal` | mu (log-scale), sigma | Rates, densities (positive) |
| `normal` | mu, sigma | Unconstrained parameters |
| `uniform` | lower, upper | Bounded parameters |
| `half_normal` | sigma | Positive with mode at zero |

Always include `rationale` explaining the prior choice.

---

## Direct Conversion Models

**Important:** `direct_conversion` models MUST have `distribution_code` in the measurement to implement the unit conversion. Without it, validation will fail.

### distribution_code Return Signature

The `derive_distribution` function MUST return a dict with these keys:
- `median`: median value (float)
- `ci95_lower`: 2.5th percentile (float)
- `ci95_upper`: 97.5th percentile (float)

```python
return {
    'median': float(np.median(samples)),
    'ci95_lower': float(np.percentile(samples, 2.5)),
    'ci95_upper': float(np.percentile(samples, 97.5)),
}
```

### Example

```yaml
model:
  type: direct_conversion
  formula: "k = ln(2) / half_life"
  data_rationale: "Paper reports half-life directly"
  submodel_rationale: "First-order kinetics in full model"

measurements:
  - name: derived_rate
    uses_inputs: [half_life, half_life_sd]
    distribution_code: |
      def derive_distribution(inputs, ureg):
          import numpy as np
          t_half = inputs['half_life'].magnitude
          t_half_sd = inputs['half_life_sd'].magnitude
          # Convert mean±SD to lognormal, then transform
          mu_log = np.log(t_half**2 / np.sqrt(t_half**2 + t_half_sd**2))
          sigma_log = np.sqrt(np.log(1 + t_half_sd**2 / t_half**2))
          samples = np.random.lognormal(mu_log, sigma_log, 10000)
          k_samples = np.log(2) / samples
          return {
              'median': float(np.median(k_samples)),
              'ci95_lower': float(np.percentile(k_samples, 2.5)),
              'ci95_upper': float(np.percentile(k_samples, 97.5)),
          }
```

---

## Example: T Cell Expansion (exponential_growth)

```yaml
target_id: tcell_expansion_smith2020

inputs:
  - name: initial_cells
    value: 100000.0
    units: cell
    input_type: experimental_condition
    role: initial_condition
    source_ref: Smith2020
    source_location: "Methods"
    value_snippet: "1×10^5 CD8+ T cells were seeded"

  - name: final_cells
    value: 750000.0
    units: cell
    uncertainty: {sd: 150000.0}
    input_type: direct_measurement
    role: target
    source_ref: Smith2020
    source_location: "Figure 2A"
    value_snippet: "7.5×10^5 ± 1.5×10^5 cells by day 3"

calibration:
  parameters:
    - name: k_CD8_pro
      units: 1/day
      prior:
        distribution: lognormal
        mu: 0.0
        sigma: 1.0
        rationale: "Wide prior centered at 1/day"

  state_variables:
    - name: T_cells
      units: cell
      initial_condition:
        input_ref: initial_cells
        rationale: "Experimental seeding density"

  model:
    type: exponential_growth
    rate_constant: k_CD8_pro
    data_rationale: "Early expansion before contact inhibition"
    submodel_rationale: "Maps to k_CD8_pro in full model"

  independent_variable:
    name: time
    units: hour
    span: [0, 72]

  measurements:
    - name: cell_count_72h
      units: cell
      uses_inputs: [final_cells]
      evaluation_points: [72]
      sample_size: 3
      observable:
        type: identity
        state_variables: [T_cells]
      likelihood:
        distribution: lognormal
        rationale: "Cell counts are positive"

  identifiability_notes: "k_pro identifiable from fold-expansion; death rate not separable"

experimental_context:
  species: human
  system: in_vitro_primary_cells

source_relevance:
  indication_match: related
  indication_match_justification: |
    Human CD8+ T cells from healthy donors used to inform PDAC model.
    T cell proliferation kinetics are conserved, but tumor-specific
    factors (immunosuppression, antigen load) not captured.
  species_source: human
  species_target: human
  source_quality: primary_human_in_vitro
  perturbation_type: physiological_baseline
  tme_compatibility: null  # Not applicable for in vitro
  estimated_translation_uncertainty_fold: 2.0

study_interpretation: "T cell expansion provides proliferation rate"
key_assumptions:
  - "Exponential growth valid for early expansion"
  - "Negligible death during expansion"

primary_data_source:
  doi: "10.1234/example.2020.12345"
  source_tag: Smith2020
  title: "T cell expansion kinetics"
  year: 2020
```

---

## Context Matching

**CRITICAL:** The functional state of cells must match the model compartment.

| Mismatch | Guidance |
|----------|----------|
| Mouse → Human | May need allometric scaling |
| In vitro → In vivo | Document in limitations |
| Activated → Quiescent | **DO NOT USE** - different biology |

Document mismatches in `key_study_limitations`.

---

## Source Relevance Assessment (REQUIRED)

Every target MUST include a `source_relevance` block evaluating how well the source data translates to the target model.

### Indication Match

| Value | When to Use | Min Uncertainty |
|-------|-------------|-----------------|
| `exact` | Same disease (PDAC data for PDAC model) | 1x |
| `related` | Same organ/disease class (other pancreatic diseases) | 2-3x |
| `proxy` | Different tissue as mechanistic proxy (melanoma for PDAC) | 3-10x |
| `unrelated` | No clear biological connection | 10-100x |

**If `proxy` or `unrelated`:** Provide detailed `indication_match_justification` (min 50 chars).

### Source Quality

| Value | Description |
|-------|-------------|
| `primary_human_clinical` | Peer-reviewed, human clinical data (best) |
| `primary_human_in_vitro` | Peer-reviewed, human cells in vitro |
| `primary_animal_in_vivo` | Peer-reviewed, animal in vivo |
| `primary_animal_in_vitro` | Peer-reviewed, animal cells in vitro |
| `review_article` | Review summarizing primary data |
| `textbook` | Textbook or reference work |
| `non_peer_reviewed` | Wikipedia, preprints (FAILS validation) |

### Perturbation Type

| Value | Requires `perturbation_relevance`? |
|-------|-----------------------------------|
| `physiological_baseline` | No |
| `pathological_state` | No |
| `pharmacological` | **YES** - explain if value is upper/lower bound |
| `genetic_perturbation` | **YES** - explain relevance to wild-type |

### TME Compatibility (immune/stromal parameters)

| Value | When to Use | Min Uncertainty |
|-------|-------------|-----------------|
| `high` | Source TME similar to target | 1x |
| `moderate` | Some TME differences | 2-3x |
| `low` | Major differences (T cell-permissive for PDAC) | 10-100x |

**If `low`:** Provide `tme_compatibility_notes` documenting differences.

### Example

```yaml
source_relevance:
  indication_match: proxy
  indication_match_justification: |
    Using B16 melanoma MDSC data as proxy for PDAC. MDSCs recruited via
    similar mechanisms but melanoma has different stromal density.
  species_source: mouse
  species_target: human
  source_quality: primary_animal_in_vivo
  perturbation_type: physiological_baseline
  tme_compatibility: low
  tme_compatibility_notes: |
    Melanoma is T cell-permissive; PDAC has dense desmoplastic stroma.
    Expect order-of-magnitude differences in recruitment rates.
  estimated_translation_uncertainty_fold: 30.0
```

---

## Validation

The schema automatically validates:
- All references (`input_ref`, `uses_inputs`, `source_ref`) point to existing items
- ODE models have `state_variables` and `independent_variable.span`
- `direct_conversion` models have `distribution_code`
- Prior predictive matches observation scale (catches unit errors)
- Values appear in their `value_snippet` (catches hallucinations)
- Units are valid Pint units
- DOIs resolve via CrossRef

**Source relevance validators (will FAIL if violated):**
- `non_peer_reviewed` source quality is rejected
- Cross-indication (`proxy`/`unrelated`) requires uncertainty >= 3x
- `pharmacological` or `genetic_perturbation` requires `perturbation_relevance`
- `low` TME compatibility requires uncertainty >= 10x and `tme_compatibility_notes`
- Cross-species requires uncertainty >= 2x

---

## YAML Formatting Rules

**CRITICAL:** Avoid special characters that corrupt YAML parsing:

- Use ASCII text only in all string fields
- Replace Greek letters with spelled-out names: `alpha`, `beta`, `gamma`, etc.
- Replace accented characters with ASCII equivalents: `á` → `a`, `ñ` → `n`, etc.
- Do not copy-paste text with hidden Unicode characters from PDFs
- Avoid mathematical symbols: use `>=` not `≥`, use `+/-` not `±`

**Examples:**
- ❌ `TGFβ` → ✓ `TGFbeta`
- ❌ `α-SMA` → ✓ `alpha-SMA`
- ❌ `Estarás` → ✓ `Estaras`
- ❌ `≥50%` → ✓ `>=50%`

---

Generate a SubmodelTarget following the Pydantic schema.
