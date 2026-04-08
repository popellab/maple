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
will be jointly inferred during Bayesian calibration. **Prefer targets with
5-7 QSP parameters over 1-2.** More shared parameters per target means more
correlations captured in joint inference and a better joint posterior.

**Nuisance parameters:** If the forward model requires a parameter that is NOT
in the QSP model (e.g., a proliferation rate needed to constrain an activation
rate), mark it `nuisance: true` and provide an inline `prior`:
```yaml
parameters:
  - name: k_activation     # QSP parameter — prior from CSV
    units: 1/day
  - name: k_prolif          # not in QSP model — nuisance
    units: 1/day
    nuisance: true
    prior:
      distribution: lognormal
      mu: -2.3
      sigma: 0.8
```
Nuisance parameters are sampled during MCMC (helping constrain the QSP
parameters) but excluded from the output priors. Only nuisance parameters may
have an inline `prior`; QSP parameters always get priors from the CSV.

**Hierarchical parameter groups:** Some QSP parameters belong to hierarchical
groups defined in `parameter_groups.yaml` (e.g., CAF subtype death rates share
a latent base rate with per-member deviations). During joint inference, grouped
parameters get partial pooling — members with submodel target data pull the
group, while members without data shrink toward the group mean. You do NOT need
to extract data for every member of a group. A single well-constrained target
on one member (e.g., k_myCAF_death from Kisseleva 2012) informs the entire
group. When designing targets, be aware that shared parameters across targets
AND group membership both contribute to the joint posterior.

{{USED_PRIMARY_STUDIES}}

---

## Your Task

1. **Understand the parameters** - Review the Parameter Context above
2. **Find relevant experimental data** - Search literature for experiments that constrain these parameters
3. **Extract the data** - Pull quantitative values with full provenance into `inputs`
4. **Build a calibration spec** - Define the model, parameters, and measurements for inference

---

## Schema Overview

{{SCHEMA_OVERVIEW}}

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
    --> `algebraic` (last resort -- requires hand-written code)

---

### direct_fit (Dose-Response Curves)

**When to use:** Paper reports effect vs dose/concentration with a sigmoidal, linear, or exponential relationship. Typically constraining an EC50/IC50.

**Example:** TGFb dose (0.02, 0.2, 2 ng/mL) vs DC maturation (IL-12p70 pg/mL). Forward model: IL12 = baseline + (max - baseline) / (1 + (TGFb/EC50)^n). Parameter to infer: TGFb_50_APC (the EC50).

**Auto-generates:** Python + Julia forward model code. No hand-written code needed.

**Required fields:**
- `curve`: hill | linear | exponential
- For hill: `ec50` (required), `n_hill` (default "1.0"), `baseline` (default "0.0"), `maximum` (default "1.0")
- For linear: `slope` (required), `intercept` (default "0.0")
- For exponential: `amplitude` (required), `rate` (required)
- Each error model entry must set `x_input` to the name of the input providing the x-value (dose/concentration) for that entry

**Common pitfall:** Using `algebraic` with hand-written Hill code instead of `direct_fit`. The structured type enables automatic prior inversion and catches unit mismatches that `algebraic` silently ignores.

### power_law (Biophysical Scaling)

**When to use:** Paper reports a power-law relationship between a physical quantity and a composition variable. Formula: y = coefficient * (x / reference_x) ^ exponent.

**Example:** Tissue stiffness vs collagen volume fraction: E = E_ref * (phi/phi_ref)^n_stiff.

**Auto-generates:** Python + Julia forward model code. No hand-written code needed.

**Required fields:**
- `coefficient`: Reference value (e.g., E_ref)
- `reference_x`: Reference x for normalization (e.g., phi_ref)
- `exponent`: Power-law exponent (parameter to estimate or fixed literal)
- Each error model entry must set `x_input` to the name of the input providing the x-value for that entry

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
| `algebraic` | Custom forward model that does not fit any structured type (e.g., t_half = ln(2) / k). **Requires code.** Only use this as a last resort when no structured type applies. |

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

## Derived Arithmetic Inputs

Use `input_type: derived_arithmetic` when a value is deterministically calculated
from other extracted inputs — not read directly from the paper. The validator
evaluates the formula and checks it matches the declared value.

**When to use:**
- Physics conversions: G' (storage modulus) to E (Young's modulus) via `E = 3 * G'`
- Arithmetic: "2 ROIs x 2 gels = 4 observations" → `n_obs = n_ROIs * n_gels`
- Unit scaling: nmol/mg to uM via tissue density

**Required fields:** `formula`, `source_inputs`, `rationale`

**Example:**
```yaml
inputs:
  - name: Gprime_stiff_kPa
    value: 17.0
    units: kilopascal
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Table 1"
    value_snippet: "G' = 17.0 kPa for stiff hydrogels"

  - name: E_stiff_kPa
    value: 51.0
    units: kilopascal
    input_type: derived_arithmetic
    source_inputs: [Gprime_stiff_kPa]
    formula: "3 * Gprime_stiff_kPa"
    rationale: "E = 2*(1+nu)*G' with nu=0.5 for incompressible PEG hydrogels"
    source_ref: Smith2020
    source_location: "Table 1"
```

**Key rules:**
- All names in `source_inputs` must be existing inputs in the same YAML
- The formula is evaluated with source input values and must match `value` within 1%
- `formula` may use basic math functions: `abs`, `min`, `max`, `log`, `sqrt`, `exp`, `pi`
- Keep raw paper values as `direct_measurement` (snippet-validated) and derive from them
- Snippet validation is skipped for `derived_arithmetic` inputs

---

## SEM vs SD (CRITICAL)

Many papers report mean +/- SEM (standard error of the mean) rather than SD (standard deviation). **Using SEM as SD dramatically underestimates population variability** because SEM = SD / sqrt(n).

**How to identify:**
- Look for explicit labels: "SD", "SEM", "SE", "s.d.", "s.e.m."
- If error seems very small relative to mean (CV < 5% for biological data), suspect SEM
- Papers with small n often report SEM to make error bars appear smaller

**When SEM is reported:**
1. Extract `n` (sample size) as a separate input
2. Name the uncertainty input `*_sem` not `*_sd` to be explicit
3. In `observation_code`, convert SEM to SD: `sd = sem * np.sqrt(n)`

**Example:**
```yaml
inputs:
  - name: treg_fraction_mean
    value: 0.178
    units: dimensionless
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Table 1"
    value_snippet: "Treg fraction 17.8 +/- 0.7%"
  - name: treg_fraction_sem
    value: 0.007075
    units: dimensionless
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Table 1"
    value_snippet: "Treg fraction 17.8 +/- 0.7%"
  - name: n_patients
    value: 45
    units: dimensionless
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Table 1"
    value_snippet: "n = 45"
```

```python
# In observation_code:
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    import numpy as np
    mean = inputs['treg_fraction_mean']
    sem = inputs['treg_fraction_sem']
    n = int(inputs['n_patients'])
    sd = sem * np.sqrt(n)  # Convert SEM to SD
    return rng.normal(mean, sd / np.sqrt(sample_size), n_bootstrap)
```

**Red flag:** If reported uncertainty seems implausibly small (e.g., 17.8 +/- 0.7% for a biological fraction), it's almost certainly SEM.

---

## Prior Distributions

**Priors come from `pdac_priors.csv`, NOT the YAML.** The `calibration.parameters` list contains only `name` and `units`. The inference pipeline reads the starting prior (distribution, mu, sigma, bounds) from the CSV file. Do NOT add a `prior` block to parameters in the YAML.

---

## Algebraic Models

When the data involves a non-ODE analytical relationship (parameter → observable), **first check whether a structured algebraic type fits** (see "Structured Algebraic" table above). Structured types auto-generate code and are strongly preferred. Only use `algebraic` if the relationship does not match any structured type.

**Key concept:** The forward model maps **parameters → predicted observable**. For example, if you're inferring rate constant `k` but the paper reports half-life `t_half`, the forward model predicts `t_half = ln(2) / k`.

### Required Fields (for `algebraic` type only -- structured types do not need these)

- `formula`: Human-readable description of the relationship
- `code`: Python forward model: `def compute(params, inputs) -> float` (single-output)
  or `def compute(params, inputs) -> dict` (multi-output)

### Multi-output algebraic models

For multi-condition experiments (e.g., control vs treatment at multiple time
points), `compute()` can return a dict of all predictions. Each error model
entry uses an `observable` to select its output — the same pattern as ODE
models:

```yaml
forward_model:
  type: algebraic
  code: |
    def compute(params, inputs):
        k = params['k_act']
        ec50 = params['EC50']
        h_ctrl = basal / (ec50 + basal)
        h_treat = (basal + exog) / (ec50 + basal + exog)
        return {
            'ctrl_d4': 1 - np.exp(-k * h_ctrl * 4),
            'treat_d4': 1 - np.exp(-k * h_treat * 4),
        }

error_model:
  - name: ctrl_day4
    observable:
      type: custom
      code: |
        def compute(t, y, y_start):
            return y['ctrl_d4']
    ...
  - name: treat_day4
    observable:
      type: custom
      code: |
        def compute(t, y, y_start):
            return y['treat_d4']
    ...
```

This replaces `custom_ode` when the ODE has an analytical solution (e.g.,
first-order decay, exponential growth + linear accumulation, Richards logistic).
Algebraic forward models are much faster than diffrax ODE solves in MCMC.

### observation_code Signature

The `derive_observation` function in `error_model` generates **parametric bootstrap samples**:

```python
def derive_observation(inputs, sample_size, rng, n_bootstrap) -> np.ndarray:
    """
    inputs: dict of {name: float} (magnitudes only, no Pint)
    sample_size: int (from sample_size_input)
    rng: numpy.random.Generator (seeded, provided by framework)
    n_bootstrap: int (number of samples to generate)

    Returns: 1D numpy array of parametric bootstrap samples
    """
```

The framework derives median, SD, and CI95 from the samples, and infers the likelihood family (lognormal/gamma/inv-gamma) via AIC. Choose the bootstrap distribution to match the data-generating process:
- `rng.normal(mean, sd / np.sqrt(sample_size), n_bootstrap)` for mean +/- SD data
- Lognormal for positive quantities (see correct parametrization below)
- `rng.poisson(count, n_bootstrap)` for count data

**Lognormal parametrization (IMPORTANT):** When bootstrapping positive quantities (rates, half-lives, concentrations, cell counts), convert CV to lognormal sigma using the exact formula `sigma_ln = sqrt(log(1 + cv^2))`, NOT the approximation `sigma ≈ cv`. The approximation diverges at CV > 0.3 (e.g., at CV=1.0 it overestimates sigma by 20%). Also apply the mean-bias correction `mu_ln = log(mean) - sigma_ln^2/2` so that `E[X] = mean` exactly.

All inputs are plain floats. Unit metadata is in the YAML schema for documentation.

```python
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    import numpy as np
    mean = inputs['half_life_mean']
    sd = inputs['half_life_sd']
    cv = sd / mean
    sigma_ln = np.sqrt(np.log(1 + cv**2))
    mu_ln = np.log(mean) - sigma_ln**2 / 2
    return rng.lognormal(mu_ln, sigma_ln / np.sqrt(sample_size), n_bootstrap)
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
  data_rationale: "Paper reports half-life directly"
  submodel_rationale: "First-order kinetics in full model"

error_model:
  - name: half_life_measurement
    units: day
    uses_inputs: [half_life_mean, half_life_sd, n_samples]
    sample_size_input: n_samples
    observation_code: |
      def derive_observation(inputs, sample_size, rng, n_bootstrap):
          import numpy as np
          t_half = inputs['half_life_mean']
          t_half_sd = inputs['half_life_sd']
          # Lognormal bootstrap: half-lives are positive
          cv = t_half_sd / t_half
          sigma_ln = np.sqrt(np.log(1 + cv**2))
          mu_ln = np.log(t_half) - sigma_ln**2 / 2
          return rng.lognormal(mu_ln, sigma_ln / np.sqrt(sample_size), n_bootstrap)
```

### No Hardcoded Values

**All numeric values must come through `inputs`**, not hardcoded in code. This ensures:
- Full provenance tracking
- Sensitivity analysis capability
- Transparent assumptions

Allowed constants: `0`, `1`, `2`, `1.96` (for CI calculations)

**Wrong:**
```python
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    import numpy as np
    sd = inputs['mean'] * 0.3  # Hardcoded 30% CV!
    return rng.lognormal(np.log(inputs['mean']), 0.3 / np.sqrt(sample_size), n_bootstrap)
```

**Correct:** extract the CV from the paper's reported variability (e.g., SD/mean from error bars, IQR/median, or replicate scatter). Do NOT use "assumed" in input names — the validator rejects inputs with "assumed" because they indicate fabricated data rather than extracted measurements.
```yaml
inputs:
  - name: biological_cv_half_life
    value: 0.3
    units: dimensionless
    input_type: reference_value
    rationale: "CV of 30% derived from reported SD/mean in Table 2 across n=6 subjects"
    source_ref: Smith2020
    source_location: "Table 2"
    value_snippet: "half-life 4.2 ± 1.3 days (mean ± SD, n=6)"
```
```python
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    import numpy as np
    cv = inputs['biological_cv_half_life']
    sigma_ln = np.sqrt(np.log(1 + cv**2))
    mu_ln = np.log(inputs['mean']) - sigma_ln**2 / 2
    return rng.lognormal(mu_ln, sigma_ln / np.sqrt(sample_size), n_bootstrap)
```

---

## Custom ODE Models

Use `custom_ode` when the built-in ODE types don't capture your dynamics.

### Required Fields

- `code`: Python ODE function: `def ode(t, y, params, inputs) -> dy`

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
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Methods"
    value_snippet: "1x10^5 CD8+ T cells were seeded"

  - name: final_cells_mean
    value: 750000.0
    units: cell
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Figure 2A"
    figure_excerpt:
      figure_id: "Figure 2A"
      value: "~7.5x10^5"
      description: "Mean cell count at 72h from bar chart"
      context: "CD8+ T cell expansion, anti-CD3/CD28 stimulation"

  - name: final_cells_sd
    value: 150000.0
    units: cell
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Figure 2A"
    figure_excerpt:
      figure_id: "Figure 2A"
      value: "+/- 1.5x10^5"
      description: "Error bar height at 72h"
      context: "Error bars represent SD (n=3)"

  - name: n_replicates
    value: 3
    units: dimensionless
    input_type: direct_measurement
    source_ref: Smith2020
    source_location: "Methods"
    value_snippet: "experiments were performed in triplicate"

calibration:
  parameters:
    - name: k_CD8_pro
      units: 1/day

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
      uses_inputs: [final_cells_mean, final_cells_sd, n_replicates]
      evaluation_points: [72]
      sample_size_input: n_replicates
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            mean = inputs['final_cells_mean']
            sd = inputs['final_cells_sd']
            # Lognormal bootstrap for positive cell counts
            cv = sd / mean
            sigma_ln = np.sqrt(np.log(1 + cv**2))
            mu_ln = np.log(mean) - sigma_ln**2 / 2
            return rng.lognormal(mu_ln, sigma_ln / np.sqrt(sample_size), n_bootstrap)
      observable:
        type: identity
        state_variables: [T_cells]

  identifiability_notes: "k_pro identifiable from fold-expansion; death rate not separable"

# NOTE: The following fields are all TOP-LEVEL (not nested inside each other):
#   experimental_context, study_interpretation, key_assumptions,
#   key_study_limitations, primary_data_source, secondary_data_sources
# Do NOT nest study_interpretation inside experimental_context — they are siblings.

experimental_context:
  species: human
  system: in_vitro_primary_cells

primary_data_source:
  doi: "10.xxxx/example"
  source_tag: Smith2023
  title: "Example paper"
  year: 2023
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
    perturbation_relevance: |
      Anti-CD3/CD28 stimulation is a standard activation protocol
      that recapitulates TCR signaling without antigen specificity.
    tme_compatibility: moderate
    tme_compatibility_notes: |
      In vitro expansion does not capture immunosuppressive TME of PDAC.
      Expect proliferation rates to be upper bounds for in vivo behavior.
    measurement_directness: direct
    temporal_resolution: endpoint_pair
    experimental_system: in_vitro_primary

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

## Source Relevance Assessment (REQUIRED, per-source)

Every data source (`primary_data_source` and each `secondary_data_sources` entry) MUST include a `source_relevance` block evaluating how well that source's data translates to the target model. Translation sigma is computed per-measurement by combining the source-level sigmas of the inputs it uses (in quadrature, deduplicated by source).

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
primary_data_source:
  doi: "10.xxxx/example"
  source_tag: Jones2021
  title: "MDSC recruitment in melanoma"
  year: 2021
  source_relevance:
    indication_match: proxy
    indication_match_justification: |
      Using B16 melanoma MDSC data as proxy for PDAC. MDSCs recruited via
      similar mechanisms but melanoma has different stromal density.
    species_source: mouse
    species_target: human
    source_quality: primary_animal_in_vivo
    perturbation_type: physiological_baseline
    perturbation_relevance: "Baseline tumor-bearing state, no drug treatment."
    tme_compatibility: low
    tme_compatibility_notes: |
      Melanoma is T cell-permissive; PDAC has dense desmoplastic stroma.
      Expect order-of-magnitude differences in recruitment rates.
    measurement_directness: direct
    temporal_resolution: endpoint_pair
    experimental_system: animal_in_vivo
```

---

## Validation

The schema automatically validates:
- All references (`input_ref`, `uses_inputs`, `source_ref`) point to existing items
- ODE models have `state_variables`, `independent_variable.span`, and `evaluation_points`
- ODE models have `observable` defined in error_model
- `algebraic` models have `code` (structured algebraic types auto-generate code)
- `observation_code` returns 1D numpy array of bootstrap samples
- `observation_code` has no hardcoded numeric values (all through inputs)
- Prior predictive matches observation scale (catches unit errors)
- AlgebraicModel forward prediction matches data scale
- Values appear in their `value_snippet` (catches hallucinations)
- Units strings are valid (validated with Pint)
- DOIs resolve via CrossRef; first author family name is compared.
  Use family name only in authors list (e.g., `["Zhang"]` not `["Zhang J"]` or `["Jianfeng Zhang"]`).
- Input names must NOT contain "assumed" — use descriptive names based on what was measured
- `study_interpretation` must be a top-level field, NOT nested inside `experimental_context`
- Non-nuisance parameters must exist in the QSP model (model_structure.json). If a parameter
  is NOT a real QSP model parameter (e.g., a nuisance scaling factor, Hill coefficient, or
  experimental condition), mark it `nuisance: true` with an inline `prior` block.
- Forward model code must NOT use Python `if`/`else`/ternary (`x if cond else y`) on
  param-dependent values — these cause JAX TracerBoolConversionError during MCMC.
  Use `jnp.where(condition, if_true, if_false)` or restructure to avoid branching.
  For guard clauses (e.g., avoiding division by zero), use `np.maximum(denominator, 1e-30)`
  instead of `if denominator > 0`.
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

## Forward Model Design Principles

**CRITICAL:** The forward model must replicate the QSP model's parameterization
of the target parameter(s), not simplify it. Include ALL Hill functions, gating
terms, and carrying capacity expressions from the actual reaction in the QSP model.

**Why this matters:** A simplified ODE like `dQ = -k_act * Q` absorbs Hill function
terms into the effective rate constant. When another target models the same parameter
with the Hill function explicit, the two targets give inconsistent estimates (e.g.,
0.065/day vs 0.92/day for the same k_act). Including the Hill functions with nuisance
parameters for experimental conditions resolves the discrepancy.

**Example:** If the QSP model has:
```
rate = k_act * TGFb/(TGFb_50 + TGFb) * PDGF/(PDGF_50 + PDGF)
```
Then the submodel ODE should be:
```python
def ode(t, y, params, inputs):
    k_act = params['k_psc_activation_myCAF']
    ec50_tgfb = params['TGFb_50_CAF_act']
    ec50_pdgf = params['PDGF_50']
    tgfb = params['tgfb_basal_culture']   # nuisance
    pdgf = params['pdgf_basal_culture']   # nuisance
    h_tgfb = tgfb / (ec50_tgfb + tgfb)
    h_pdgf = pdgf / (ec50_pdgf + pdgf)
    dQ = -k_act * h_tgfb * h_pdgf * Q
    ...
```

NOT the simplified version:
```python
def ode(t, y, params, inputs):
    k_act = params['k_psc_activation_myCAF']
    dQ = -k_act * Q  # WRONG: absorbs Hill terms into k_act
```

**Steps:**
1. Look up the actual reaction in `core/modules/*.m` that uses the parameter
2. Include ALL Hill functions and gating terms from that reaction
3. Add nuisance parameters for experimental conditions (basal cytokine levels,
   culture volume, etc.) that are not QSP parameters
4. The EC50 parameters (e.g., TGFb_50_CAF_act) become QSP parameters in the
   target, providing additional constraints from the same data

---

## Working with Figure Data

Most rich kinetic data (time courses, dose-response curves) is in figures, not
tables. **Always prefer digitizing figures when they contain richer data than
text summaries.** If a paper reports only a mean in text but has a scatter plot
with individual data points in a figure, digitize the figure — it gives real
patient-level distributions and proper uncertainty without invented assumptions.

Standard workflow for figure digitization:

1. Describe the figure to the user: figure ID, axes, scale (log vs linear),
   what data to capture (individual points, means, error bars)
2. The user digitizes with WebPlotDigitizer (WPD) and downloads a CSV
3. Read the WPD CSV (x,y coordinates), group by condition, compute summary
   statistics (mean, SD, n per group)
4. Use `figure_excerpt` entries (not `value_snippet`) for digitized values —
   these are correctly flagged as MANUAL REVIEW REQUIRED by the validator

**Do NOT create CSV templates with empty rows.** The user uses WPD directly.

**Keep input values in paper units.** Do not convert percent to fraction, hours
to days, ng/ml to nM in the input values. The validator checks that input values
appear in snippets/figure_excerpts, so converted values fail validation. Do all
unit conversions in the forward model code or observation_code.

---

## Lessons from Extraction Practice

- **Translation sigma dominates for proxy data.** Cross-species, cross-organ
  data (e.g., rat liver HSC for human PDAC CAF) gets translation sigma ~0.75-0.80.
  Expect moderate contraction (0.2-0.7), not tight constraints. This is correct —
  the data is informative but the extrapolation adds real uncertainty.

- **Don't invent uncertainty.** If a paper reports only point estimates without
  SD, use the actual spread in the data: bootstrap across conditions, use the
  range of measurements across states, or use subgroup-level values as a
  bootstrap population. If there is genuinely no spread, the target may not be
  cleanly extractable.

- **Multiple targets for the same parameter are valuable.** Different experimental
  designs probe different aspects of the same parameter. Joint inference composes
  them. Disagreement between targets often reveals modeling assumptions that need
  revision (missing Hill functions, incorrect units, etc.).

---

Generate a SubmodelTarget following the Pydantic schema.
