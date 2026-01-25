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
when mechanistically necessary. For example, if calibrating a proliferation rate,
you may also need death rates, carrying capacities, or other parameters from
the same reactions. See the **Parameter Context** section above for related
parameters in the same reactions. All parameters listed in `calibration.parameters`
will be jointly inferred during Bayesian calibration.

---

## Your Task

1. **Understand the parameters** - Review the Parameter Context above to see reactions and related parameters
2. **Find relevant experimental data** - Search literature for experiments that constrain these parameters
3. **Extract the data** - Pull quantitative values with full provenance into `inputs`
4. **Build a calibration spec** - Define the model, parameters, and measurements for inference

---

## SubmodelTarget Schema Overview

The SubmodelTarget schema separates:
- `inputs`: What was extracted from papers (with full provenance)
- `calibration`: How to use those inputs for inference

```
SubmodelTarget
├── target_id: str
├── inputs: List[Input]                    # Extracted values with provenance
│   ├── name, value, units
│   ├── input_type: direct_measurement | proxy_measurement | experimental_condition | inferred_estimate | assumed_value
│   ├── role: initial_condition | target | fixed_parameter | auxiliary
│   └── source_ref, source_location, value_snippet
├── calibration
│   ├── parameters: List[Parameter]        # Parameters to estimate
│   │   ├── name, units
│   │   └── prior: {distribution, mu, sigma, ...}
│   ├── state_variables: List[StateVariable]  # For ODE models
│   │   └── initial_condition: {value, rationale} | {input_ref, rationale}
│   ├── model: Model                       # One of the typed models below
│   ├── independent_variable: {name, units, span}
│   ├── measurements: List[Measurement]
│   │   ├── uses_inputs: List[str]         # References to inputs by name
│   │   ├── evaluation_points: List[float]
│   │   └── likelihood: {distribution, rationale}
│   └── identifiability_notes: str
├── experimental_context
├── study_interpretation, key_assumptions, key_study_limitations
└── primary_data_source, secondary_data_sources
```

---

## Model Types

Choose the model type that best fits the experimental dynamics:

### ODE-based models (require state_variables, independent_variable with span)

| Type | Equation | Use When |
|------|----------|----------|
| `exponential_growth` | dy/dt = k * y | Unconstrained growth (early expansion) |
| `first_order_decay` | dy/dt = -k * y | Clearance, death, decay |
| `logistic` | dy/dt = k * y * (1 - y/K) | Growth with carrying capacity |
| `michaelis_menten` | dy/dt = -Vmax * y / (Km + y) | Saturable kinetics |
| `two_state` | dA/dt = -k*A, dB/dt = +k*A | State transitions (activation, differentiation) |
| `saturation` | dy/dt = k * (1 - y) | Approach to saturation |
| `custom` | User-defined | Complex multi-parameter dynamics |

### Non-ODE models

| Type | Use When |
|------|----------|
| `direct_conversion` | Analytical formula exists (e.g., k = ln(2) / t_half) |
| `direct_fit` | Curve fitting (Hill equation for IC50, etc.) |

---

## Input Roles

Each input has a `role` that clarifies how it's used:

| Role | Description | Example |
|------|-------------|---------|
| `initial_condition` | Starting value for ODE integration | Seeding density |
| `target` | Calibration target (likelihood term) | Cell count at day 3 |
| `fixed_parameter` | Fixed value in model (not estimated) | Known rate from literature |
| `auxiliary` | Supporting data, not directly used | SD values, sample sizes |

---

## Prior Specification

Priors should reflect biological plausibility:

```yaml
parameters:
  - name: k_proliferation
    units: 1/day
    prior:
      distribution: lognormal
      mu: 0.0        # log(1.0) = center at 1/day
      sigma: 1.0     # ~3-fold uncertainty
      rationale: "Wide prior centered at 1/day; typical for epithelial cells"
```

**Distribution guide:**
| Parameter Type | Distribution | Why |
|----------------|--------------|-----|
| Rates (k, Vmax) | lognormal | Always positive, often log-distributed |
| Proportions (0-1) | beta or logit-normal | Bounded support |
| Unconstrained | normal | No natural bounds |
| Scale parameters | half_normal | Positive, mode at zero |

---

## Choosing Direct Conversion vs ODE Model

### Use `direct_conversion` when:
- Literature reports a derived quantity with simple analytical relationship to parameter
- Examples: doubling time, half-life, Kd from binding assay, mean residence time

```yaml
model:
  type: direct_conversion
  formula: "k = ln(2) / doubling_time"
  data_rationale: "Paper reports doubling time directly"
  submodel_rationale: "Exponential growth assumption valid for early expansion"
```

### Use ODE model when:
- No analytical solution exists
- Multiple measurements over time
- Complex dynamics (saturation, multi-state)

---

## Example: T Cell Expansion (exponential_growth)

**Parameters requested:** `k_CD8_pro`

**Literature found:** Smith et al. (2020)
- "CD8+ T cells expanded from 100,000 to 750,000 ± 150,000 cells over 72 hours"

```yaml
target_id: tcell_expansion_smith2020

inputs:
  - name: initial_cells
    value: 100000.0
    units: cell
    input_type: experimental_condition
    role: initial_condition
    source_ref: Smith2020
    source_location: "Methods, Cell Culture"
    value_snippet: "1×10^5 CD8+ T cells were seeded per well"

  - name: final_cells
    value: 750000.0
    units: cell
    input_type: direct_measurement
    role: target
    source_ref: Smith2020
    source_location: "Figure 2A"
    value_snippet: "CD8+ T cells expanded to 7.5×10^5 ± 1.5×10^5 cells by day 3"

  - name: final_cells_sd
    value: 150000.0
    units: cell
    input_type: direct_measurement
    role: auxiliary
    source_ref: Smith2020
    source_location: "Figure 2A"
    value_snippet: "CD8+ T cells expanded to 7.5×10^5 ± 1.5×10^5 cells by day 3"

calibration:
  parameters:
    - name: k_CD8_pro
      units: 1/day
      prior:
        distribution: lognormal
        mu: 0.0
        sigma: 1.0
        rationale: "Wide prior centered at 1/day for T cell proliferation"

  state_variables:
    - name: T_cells
      units: cell
      initial_condition:
        input_ref: initial_cells
        rationale: "Use experimental seeding density"

  model:
    type: exponential_growth
    rate_constant: k_CD8_pro
    data_rationale: "Early T cell expansion before contact inhibition"
    submodel_rationale: "Maps to k_CD8_pro proliferation reaction in full model"

  independent_variable:
    name: time
    units: hour
    span: [0, 72]
    rationale: "Matches experimental time course"

  measurements:
    - name: cell_count_72h
      units: cell
      uses_inputs: [final_cells, final_cells_sd]
      evaluation_points: [72]
      observable:
        type: identity
        state_variables: [T_cells]
      likelihood:
        distribution: lognormal
        rationale: "Cell counts are positive and often log-distributed"

  identifiability_notes: "Single parameter k_pro identifiable from fold-expansion; death rate not separable"

experimental_context:
  species: human
  system: in_vitro_primary_cells
  cell_types:
    - name: CD8+ T cells
      phenotype: activated
      isolation_method: PBMC isolation + CD8 selection

study_interpretation: "Anti-CD3/CD28 stimulated T cell expansion provides proliferation rate for activated T cells"

key_assumptions:
  - "Exponential growth valid for early expansion (no contact inhibition)"
  - "Negligible death during expansion phase"

primary_data_source:
  doi: "10.1234/example.2020.12345"
  source_tag: Smith2020
  title: "T cell expansion kinetics in vitro"
  year: 2020
```

---

## Example: Direct Conversion (half-life to rate)

**Parameters requested:** `k_drug_clearance`

**Literature found:** Half-life = 12 hours

```yaml
target_id: drug_clearance_jones2019

inputs:
  - name: half_life
    value: 12.0
    units: hour
    input_type: direct_measurement
    role: target
    source_ref: Jones2019
    source_location: "Table 2"
    value_snippet: "Terminal half-life: 12 ± 2 hours"

  - name: half_life_sd
    value: 2.0
    units: hour
    input_type: direct_measurement
    role: auxiliary
    source_ref: Jones2019
    source_location: "Table 2"
    value_snippet: "Terminal half-life: 12 ± 2 hours"

calibration:
  parameters:
    - name: k_drug_clearance
      units: 1/hour
      prior:
        distribution: lognormal
        mu: -2.5  # log(ln(2)/12) ≈ log(0.058)
        sigma: 0.5
        rationale: "Prior centered on expected clearance rate from half-life"

  model:
    type: direct_conversion
    formula: "k = ln(2) / half_life"
    data_rationale: "Terminal half-life directly measured via PK study"
    submodel_rationale: "First-order elimination assumed in full PK model"

  measurements:
    - name: clearance_rate
      units: 1/hour
      uses_inputs: [half_life, half_life_sd]
      evaluation_points: [0]  # Single derived value
      distribution_code: |
        def derive_distribution(inputs, ureg):
            import numpy as np
            rng = np.random.default_rng(42)

            t_half = inputs['half_life']
            t_half_sd = inputs['half_life_sd']

            # Lognormal for positive half-lives
            mu_log = np.log(t_half.magnitude**2 / np.sqrt(t_half.magnitude**2 + t_half_sd.magnitude**2))
            sigma_log = np.sqrt(np.log(1 + t_half_sd.magnitude**2 / t_half.magnitude**2))
            t_samples = rng.lognormal(mu_log, sigma_log, 10000) * t_half.units

            # Convert to rate
            k_samples = np.log(2) / t_samples

            return {
                'median_obs': np.median(k_samples),
                'ci95_lower': np.percentile(k_samples, 2.5),
                'ci95_upper': np.percentile(k_samples, 97.5),
            }
      likelihood:
        distribution: lognormal
        rationale: "Rate constants are positive"

  identifiability_notes: "Clearance rate directly derivable from half-life measurement"

# ... rest of fields
```

---

## Handling Context Mismatch

Document any mismatch between experimental and model context:

| From | To | Typical Adjustment |
|------|----|--------------------|
| Mouse | Human (PK) | Allometric scaling |
| In vitro | In vivo | Rates may differ due to tissue context |
| Cell line | Primary cells | Primary often slower |
| Activated cells | Quiescent cells | DO NOT USE - different biology |

Add context mismatches to `key_study_limitations`.

---

## Functional State Matching

**CRITICAL:** The functional state of cells must match the model compartment.

QSP models distinguish:
- Quiescent vs. activated stellate cells
- Naïve vs. effector vs. exhausted T cells
- M1 vs. M2 macrophages

Using activated cell data for a quiescent parameter is **wrong biology**, not just a scaling issue.

---

## Validation

The schema validates:
1. All `input_ref` and `uses_inputs` references point to existing inputs
2. All `source_ref` values match defined sources
3. ODE models have required `state_variables` and `independent_variable.span`
4. Custom code has valid Python syntax
5. Units are valid Pint units
6. DOIs resolve via CrossRef

---

Generate a SubmodelTarget following the Pydantic schema.
