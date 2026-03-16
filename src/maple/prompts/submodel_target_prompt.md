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
│   ├── forward_model: ForwardModel        # Physics/math: params → predictions
│   │   ├── type: exponential_growth | first_order_decay | steady_state_density | ... | algebraic | custom_ode
│   │   ├── state_variables: [{name, units, initial_condition}]  # For ODE models
│   │   ├── independent_variable: {name, units, span}            # For ODE models
│   │   └── (type-specific fields: ParameterRole fields for structured types, or code/code_julia for algebraic)
│   ├── error_model: List[ErrorModel]      # Statistics: predictions + data → likelihood
│   │   ├── name, units, uses_inputs
│   │   ├── evaluation_points: [float]     # For ODE models only
│   │   ├── observation_code: str          # Returns {value, sd, sd_uncertain}
│   │   ├── sample_size, observable, likelihood
│   │   └── ...
│   └── identifiability_notes: str
├── experimental_context: {species, system, cell_lines, cell_types, ...}
├── source_relevance: SourceRelevanceAssessment  # REQUIRED - see below
├── study_interpretation, key_assumptions, key_study_limitations
└── primary_data_source, secondary_data_sources
```

---

## Forward Model Types

### How to choose a forward model

Ask these questions IN ORDER. Use the FIRST match:

1. Is the data a dose-response or titration curve?
   (EC50, IC50, dose vs effect, concentration vs response)
   --> `direct_fit`
       curve: hill (sigmoidal), linear, or exponential

2. Is the data a biophysical scaling relationship?
   (stiffness vs density, pore size vs concentration, diffusion vs viscosity)
   --> `power_law`

3. Is the data a steady-state cell density from IHC/mIF?
   (cells/mm^2, cells per field)
   --> `steady_state_density`

4. Is the data a % of a parent population from flow cytometry?
   (CD8+/CD3+, Treg/CD4+, Ki67+/total)
   --> `steady_state_fraction` (for Ki67/BrdU, use `steady_state_proliferation_index`)

5. Is the data a concentration from ELISA/Luminex?
   (cytokine pg/mL, chemokine nM in culture supernatant or tissue)
   --> `steady_state_concentration`

6. Is the data a ratio of two cell populations?
   (M1/M2, CD4/CD8, CAF/cancer)
   --> `steady_state_ratio`

7. Is the data from an in vitro secretion/accumulation assay?
   (pg/mL after N hours of culture with known cell count)
   --> `batch_accumulation`

8. Is the data a time course with simple dynamics?
   - Unconstrained growth --> `exponential_growth`
   - Decay/clearance --> `first_order_decay`
   - Growth with plateau --> `logistic`
   - Saturable consumption --> `michaelis_menten`
   - State transition (A->B) --> `two_state`
   - Approach to steady state --> `saturation`

9. Is the data a time course with complex dynamics?
   (multiple interacting species, feedback loops, non-standard kinetics)
   --> `custom_ode`

10. None of the above fit?
    --> `algebraic` (last resort -- requires hand-written code + code_julia)

---

### direct_fit (Dose-Response Curves)

**When to use:** Paper reports effect vs dose/concentration with a sigmoidal, linear, or exponential relationship. Typically constraining an EC50/IC50.

**Example:** TGFb dose (0.02, 0.2, 2 ng/mL) vs DC maturation (IL-12p70 pg/mL). Forward model: IL12 = baseline + (max - baseline) / (1 + (TGFb/EC50)^n). Parameter to infer: TGFb_50_APC (the EC50).

**Auto-generates:** Python + Julia forward model code. No hand-written code needed.

**Required fields:**
- `curve`: hill | linear | exponential
- `x_variable`: input_ref to the dose/concentration input
- For hill: `ec50` (required), `n_hill` (default "1.0"), `baseline` (default "0.0"), `maximum` (default "1.0")
- For linear: `slope` (required), `intercept` (default "0.0")
- For exponential: `amplitude` (required), `rate` (required)

**Common pitfall:** Using `algebraic` with hand-written Hill code instead of `direct_fit`. The structured type enables automatic prior inversion and catches unit mismatches that `algebraic` silently ignores.

### power_law (Biophysical Scaling)

**When to use:** Paper reports a power-law relationship between a physical quantity and a composition variable. Formula: y = coefficient * (x / reference_x) ^ exponent.

**Example:** Tissue stiffness vs collagen volume fraction: E = E_ref * (phi/phi_ref)^n_stiff.

**Auto-generates:** Python + Julia forward model code. No hand-written code needed.

**Required fields:**
- `coefficient`: Reference value (e.g., E_ref)
- `reference_x`: Reference x for normalization (e.g., phi_ref)
- `exponent`: Power-law exponent (parameter to estimate or fixed literal)
- `x_variable`: input_ref to the measured x values

**Common pitfall:** Using `algebraic` when the relationship is a simple power law. The structured type is cleaner and auto-validates.

---

### ODE-based (require state_variables, independent_variable with span)

| Type | Equation | Use When |
|------|----------|----------|
| `exponential_growth` | dy/dt = k * y | Unconstrained growth |
| `first_order_decay` | dy/dt = -k * y | Clearance, death, decay |
| `logistic` | dy/dt = k * y * (1 - y/K) | Growth with carrying capacity |
| `michaelis_menten` | dy/dt = -Vmax * y / (Km + y) | Saturable kinetics |
| `two_state` | dA/dt = -k*A, dB/dt = +k*A | State transitions |
| `saturation` | dy/dt = k * (1 - y) | Approach to saturation |
| `custom_ode` | User-defined ODE | Complex dynamics |

### Non-ODE (Manual)

| Type | Use When |
|------|----------|
| `algebraic` | Custom forward model that does not fit any structured type (e.g., t_half = ln(2) / k). **Requires code and code_julia.** Only use this as a last resort when no structured type applies. |

### Structured Algebraic (preferred over `algebraic` -- no code needed)

| Type | Formula | Use When |
|------|---------|----------|
| `steady_state_density` | density = rate * source * eff * (1-excl) / loss * svf | IHC/mIF cell density (cells/mm^2) |
| `steady_state_fraction` | fraction = rate * drive / (loss * parent_density) | Flow cytometry % of parent population |
| `steady_state_concentration` | conc = sec_rate * source / (clearance * volume) | Serum/tissue ELISA concentration |
| `steady_state_ratio` | ratio = rate_num * drive_num / (rate_den * drive_den) | Cell population ratios (M2:M1, CD4:CD8) |
| `steady_state_proliferation_index` | f = prolif * dur / (prolif * dur + loss) | Ki-67+/BrdU+ fraction |
| `batch_accumulation` | mass = sec_rate * cells * time * MW * ucf / vol | In vitro secretion assay (ELISA) |
| `direct_fit` | hill: y = base + (max-base)/(1+(x/ec50)^n); linear: y = slope*x + intercept; exponential: y = amp*exp(rate*x) | Dose-response curves (EC50/IC50) |
| `power_law` | y = coefficient * (x / reference_x) ^ exponent | Biophysical scaling (stiffness, pore size) |

These models auto-generate code from their fields. Each field is a **ParameterRole**: either a parameter name (string to estimate), an `input_ref` (fixed from extracted input), a `reference_ref` (fixed from curated reference database), or a numeric literal string.

- Use `input_ref` for values extracted from the paper: `{input_ref: "my_input_name"}`
- Use `reference_ref` for curated physiological constants: `{reference_ref: "circulating_cd8_count"}`
- Use a numeric literal string for known constants: `"1.0"`, `"4e-6"`
- Use a bare parameter name string for the parameter(s) to estimate: `"k_rec"`

**Available reference values** (use these exact names with `reference_ref`):

{{REFERENCE_DATABASE}}

**unit_conversion_factor**: Set this when rate parameters have mismatched time units (e.g., target_rate is per-minute but loss_rate is per-day: ucf = 1440.0). Default "1.0".

---

## Input Roles

| Role | Description | Example |
|------|-------------|---------|
| `initial_condition` | Starting value for ODE | Seeding density |
| `target` | Calibration target | Cell count at day 3 |
| `fixed_parameter` | Fixed in model | Known rate from literature |
| `auxiliary` | Supporting data | SD/SEM values, sample sizes |

---

## SEM vs SD (CRITICAL)

Many papers report mean +/- SEM (standard error of the mean) rather than SD (standard deviation). **Using SEM as SD dramatically underestimates population variability** because SEM = SD / sqrt(n).

**How to identify:**
- Look for explicit labels: "SD", "SEM", "SE", "s.d.", "s.e.m."
- If error seems very small relative to mean (CV < 5% for biological data), suspect SEM
- Papers with small n often report SEM to make error bars appear smaller

**When SEM is reported:**
1. Extract `n` (sample size) as a separate input with `role: auxiliary`
2. Name the uncertainty input `*_sem` not `*_sd` to be explicit
3. In `observation_code`, convert: `sd = sem * np.sqrt(n)`
4. Add a `notes` field explaining the conversion

**Example:**
```yaml
inputs:
  - name: treg_fraction_mean
    value: 0.178
    role: target
  - name: treg_fraction_sem
    value: 0.007075
    role: auxiliary
    notes: |
      Reported as mean+/-SEM. SD = SEM * sqrt(n) = 0.007 * sqrt(45) = 0.047
  - name: n_patients
    value: 45
    role: auxiliary
```

```python
# In observation_code:
def derive_observation(inputs, sample_size):
    mean = inputs['treg_fraction_mean']
    sem = inputs['treg_fraction_sem']
    n = int(inputs['n_patients'])
    sd = sem * np.sqrt(n)  # Convert SEM to SD
    return {
        'value': mean,
        'sd': sd,
        'sd_uncertain': True,  # SD derived from SEM, some uncertainty
    }
```

**Red flag:** If reported uncertainty seems implausibly small (e.g., 17.8 +/- 0.7% for a biological fraction), it's almost certainly SEM.

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

## Algebraic Models

When the data involves a non-ODE analytical relationship (parameter → observable), **first check whether a structured algebraic type fits** (see "Structured Algebraic" table above). Structured types auto-generate code and are strongly preferred. Only use `algebraic` if the relationship does not match any structured type.

**Key concept:** The forward model maps **parameters → predicted observable**. For example, if you're inferring rate constant `k` but the paper reports half-life `t_half`, the forward model predicts `t_half = ln(2) / k`.

### Required Fields (for `algebraic` type only -- structured types do not need these)

- `formula`: Human-readable description of the relationship
- `code`: Python forward model: `def compute(params, inputs) -> float`
- `code_julia`: Julia forward model: `function compute(params, inputs) -> value`

### observation_code Return Signature

The `derive_observation` function in `error_model` MUST return:
- `value`: Observed point estimate (plain float) **[required]**
- `sd`: Measurement uncertainty (plain float: dimensionless CV for lognormal, absolute for normal) **[required]**
- `sd_uncertain`: If True, inference adds a prior on sigma (optional, default False)
- `n`: Sample size for reference (optional)

All inputs are plain floats. Unit metadata is in the YAML schema for documentation. Your code works with plain numbers.

```python
def derive_observation(inputs, sample_size):
    return {
        'value': inputs['half_life_mean'],
        'sd': inputs['half_life_sd'] / inputs['half_life_mean'],  # CV for lognormal
        'sd_uncertain': False,
    }
```

### Example: Inferring rate from half-life (`algebraic` fallback)

```yaml
forward_model:
  type: algebraic
  formula: "t_half = ln(2) / k"
  code: |
    def compute(params, inputs):
        import numpy as np
        k = params['k_clearance']
        return np.log(2) / k
  code_julia: |
    function compute(params, inputs)
        return log(2) / params["k_clearance"]
    end
  data_rationale: "Paper reports half-life directly"
  submodel_rationale: "First-order kinetics in full model"

error_model:
  - name: half_life_measurement
    units: day
    uses_inputs: [half_life_mean, half_life_sd]
    observation_code: |
      def derive_observation(inputs, sample_size):
          t_half = inputs['half_life_mean']
          t_half_sd = inputs['half_life_sd']
          # For lognormal likelihood, SD is in log-space (CV)
          cv = t_half_sd / t_half
          return {
              'value': t_half,
              'sd': cv,
          }
    likelihood:
      distribution: lognormal
      rationale: "Half-lives are positive and often log-distributed"
```

### No Hardcoded Values

**All numeric values must come through `inputs`**, not hardcoded in code. This ensures:
- Full provenance tracking
- Sensitivity analysis capability
- Transparent assumptions

Allowed constants: `0`, `1`, `2`, `1.96` (for CI calculations)

❌ **Wrong:**
```python
def derive_observation(inputs, sample_size):
    sd = inputs['mean'] * 0.3  # Hardcoded 30% CV!
    return {'value': inputs['mean'], 'sd': sd}
```

✓ **Correct:**
```yaml
inputs:
  - name: assumed_cv
    value: 0.3
    input_type: assumed_value
    notes: "Assumed 30% CV based on typical biological variability"
```
```python
def derive_observation(inputs, sample_size):
    cv = inputs['assumed_cv']
    return {'value': inputs['mean'], 'sd': cv}
```

---

## Custom ODE Models

Use `custom_ode` when the built-in ODE types don't capture your dynamics.

### Required Fields

- `code`: Python ODE function: `def ode(t, y, params, inputs) -> dy`
- `code_julia`: Julia ODE function: `function ode!(du, u, p, t)`

### Example: Two-compartment model

```yaml
forward_model:
  type: custom_ode
  code: |
    def ode(t, y, params, inputs):
        A, B = y
        k_ab = params['k_transfer']
        k_ba = params['k_return']
        dA = -k_ab * A + k_ba * B
        dB = k_ab * A - k_ba * B
        return [dA, dB]
  code_julia: |
    function two_compartment!(du, u, p, t)
        A, B = u
        k_ab, k_ba = p
        du[1] = -k_ab * A + k_ba * B
        du[2] = k_ab * A - k_ba * B
    end
  data_rationale: "Two-compartment kinetics observed"
  submodel_rationale: "Maps to transfer rates in full model"
  state_variables:
    - name: A
      units: cell
      initial_condition:
        input_ref: initial_A
        rationale: "Initial cells in compartment A"
    - name: B
      units: cell
      initial_condition:
        value: 0.0
        rationale: "No cells initially in compartment B"
  independent_variable:
    name: time
    units: day
    span: [0, 14]
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
    value_snippet: "1x10^5 CD8+ T cells were seeded"

  - name: final_cells_mean
    value: 750000.0
    units: cell
    input_type: direct_measurement
    role: target
    source_ref: Smith2020
    source_location: "Figure 2A"
    value_snippet: "7.5x10^5 +/- 1.5x10^5 cells by day 3"

  - name: final_cells_sd
    value: 150000.0
    units: cell
    input_type: direct_measurement
    role: auxiliary
    source_ref: Smith2020
    source_location: "Figure 2A"
    value_snippet: "7.5x10^5 +/- 1.5x10^5 cells by day 3"

calibration:
  parameters:
    - name: k_CD8_pro
      units: 1/day
      prior:
        distribution: lognormal
        mu: 0.0
        sigma: 1.0
        rationale: "Wide prior centered at 1/day"

  forward_model:
    type: exponential_growth
    rate_constant: k_CD8_pro
    data_rationale: "Early expansion before contact inhibition"
    submodel_rationale: "Maps to k_CD8_pro in full model"
    state_variables:
      - name: T_cells
        units: cell
        initial_condition:
          input_ref: initial_cells
          rationale: "Experimental seeding density"
    independent_variable:
      name: time
      units: hour
      span: [0, 72]

  error_model:
    - name: cell_count_72h
      units: cell
      uses_inputs: [final_cells_mean, final_cells_sd]
      evaluation_points: [72]
      sample_size: 3
      observation_code: |
        def derive_observation(inputs, sample_size):
            mean = inputs['final_cells_mean']
            sd = inputs['final_cells_sd']
            # CV for lognormal likelihood
            cv = sd / mean
            return {
                'value': mean,
                'sd': cv,
            }
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
- ODE models have `state_variables`, `independent_variable.span`, and `evaluation_points`
- ODE models have `observable` defined in error_model
- `algebraic` models have `code` and `code_julia` (structured algebraic types auto-generate code)
- `observation_code` returns `{value, sd}` as plain floats
- `observation_code` has no hardcoded numeric values (all through inputs)
- Prior predictive matches observation scale (catches unit errors)
- AlgebraicModel forward prediction matches data scale
- Values appear in their `value_snippet` (catches hallucinations)
- Units strings are valid (validated with Pint)
- DOIs resolve via CrossRef
- `reference_ref` values exist in the reference database
- Structured model ParameterRole fields resolve to a parameter, input, reference, or numeric literal

**Source relevance validators (will FAIL if violated):**
- `non_peer_reviewed` source quality is rejected
- Cross-indication (`proxy`/`unrelated`) requires uncertainty >= 3x
- `pharmacological` or `genetic_perturbation` requires `perturbation_relevance`
- `low` TME compatibility requires uncertainty >= 10x and `tme_compatibility_notes`
- Cross-species requires uncertainty >= 2x

**Warnings (won't fail, but review carefully):**
- AlgebraicModel has more parameters than measurements (identifiability)
- Parameters not referenced in AlgebraicModel.code
- SD is >100x or <0.001x observed value (units mismatch)
- Input units don't match measurement units
- `unit_conversion_factor` is flagged if not a common conversion (60, 1440, 3600, 86400, powers of 10)

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
