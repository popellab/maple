# Isolated System Target Extraction

Find experimental data to calibrate the specified model parameters and define a Python ODE submodel.

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
parameters in the same reactions. All parameters listed in `submodel.parameters`
will be jointly inferred during Bayesian calibration.

---

## Your Task

1. **Understand the parameters** - Review the Parameter Context above to see reactions and related parameters
2. **Find relevant experimental data** - Search literature for experiments that constrain these parameters
3. **Extract the data** - Pull quantitative values with full provenance
4. **Build a submodel** - Define an ODE using the exact parameter names from the context above

---

## How Calibration Works

Understanding the relationship between the data structure fields:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LITERATURE                                           │
│  "Cell count was 800,000 ± 100,000 at 72h starting from 100,000 cells"     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  inputs (EstimateInput)                                                     │
│    - cell_count_mean: 800000 (from paper)                                   │
│    - cell_count_sd: 100000 (from paper)                                     │
│    - initial_cells: 100000 (experimental condition)                         │
│                                                                             │
│  assumptions (ModelingAssumption)                                           │
│    - n_mc_samples: 10000 (modeling choice)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  distribution_code                                                          │
│    Computes the DATA DISTRIBUTION from literature values                    │
│    → Returns: median_obs, ci95_obs (what we OBSERVED)                       │
│                                                                             │
│    def derive_distribution(inputs, ureg):                                   │
│        mean = inputs['cell_count_mean']  # Keep as Quantity                 │
│        sd = inputs['cell_count_sd']                                         │
│        # Reattach units immediately after sampling → units propagate        │
│        samples = rng.lognormal(mu, sigma, n) * mean.units                   │
│        median_obs = np.array([np.median(samples)])  # Preserves units       │
│        ci95 = np.percentile(samples, [2.5, 97.5])   # Preserves units       │
│        ci95_obs = [[ci95[0], ci95[1]]]                                      │
│        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  empirical_data                                               │
│    The DATA DISTRIBUTION we're trying to match                              │
│    - median: [800000]      ← what literature reports                        │
│    - ci95: [[604000, 996000]]                                               │
│    - units: "cell"                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  Bayesian inference finds parameters
                                    │  that make submodel output ≈ data
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  submodel                                                                   │
│    ODE that SIMULATES the observable                                        │
│                                                                             │
│    def submodel(t, y, params, inputs):                                      │
│        N = y[0]                                                             │
│        k = params['k_proliferation']  ← parameter to calibrate              │
│        return [k * N]                                                       │
│                                                                             │
│  observable                                                                 │
│    How to compute measurement from submodel state                           │
│    → Default: y[0] (cell count)                                             │
│    → Custom: e.g., convert cells to diameter                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** The `distribution_code` computes the **DATA** distribution (what the paper reports), NOT the parameter distribution.

### Pint Units Golden Rule

**Reattach units immediately after operations that strip them → units propagate naturally.**

```python
def derive_distribution(inputs, ureg):
    import numpy as np
    # Inputs are already Pint Quantities - use directly
    mean = inputs['cell_count_mean']
    sd = inputs['cell_count_sd']
    n_samples = int(inputs['n_mc_samples'].magnitude)  # Only extract for int conversion

    # Random sampling strips units - reattach IMMEDIATELY
    rng = np.random.default_rng(42)
    samples = rng.normal(mean.magnitude, sd.magnitude, n_samples) * mean.units

    # Now numpy operations preserve units automatically!
    # np.median, np.percentile, np.mean, np.std all work on Quantities
    return {
        'median_obs': np.array([np.median(samples)]),  # Returns Quantity
        'ci95_obs': [[np.percentile(samples, 2.5), np.percentile(samples, 97.5)]]
    }
```

**Key principle:** Sampling (and sqrt, etc.) strips units. Reattach immediately after, then units propagate through subsequent numpy operations.

### Choosing Probability Distributions

**Use lognormal for positive quantities** (cell counts, concentrations, volumes, rates).

Normal distributions often yield negative draws → require clipping → **introduces bias**.

```python
# BAD: Normal + clipping introduces bias
samples = rng.normal(mean, sd, n)
samples = np.clip(samples, 0, None)  # Red flag!

# GOOD: Lognormal naturally positive
# Convert normal(mean, sd) to lognormal parameters:
mu_log = np.log(mean**2 / np.sqrt(mean**2 + sd**2))
sigma_log = np.sqrt(np.log(1 + sd**2 / mean**2))
samples = rng.lognormal(mu_log, sigma_log, n) * units
```

**Distribution guide:**
| Data Type | Distribution | Why |
|-----------|--------------|-----|
| Sizes, volumes, counts | Lognormal | Always positive, often right-skewed |
| Proportions (0-1) | Beta or logit-normal | Bounded support |
| Symmetric unbounded | Normal | No natural bounds |

**Red flag:** If you need `np.clip()` or `np.maximum(..., 0)`, you probably need lognormal.

---

## Two Modes: Direct Conversion vs Submodel

Choose based on whether there's an analytical formula:

### Direct Conversion (submodel = null)

Use when literature value converts directly to parameter via simple algebra:

| Literature Reports | Conversion Formula |
|--------------------|-------------------|
| Doubling time | `k = ln(2) / t_double` |
| Half-life | `k = ln(2) / t_half` |
| Kd from binding assay | Direct use |
| Mean residence time | `k = 1 / MRT` |

**Workflow:** `distribution_code` applies the formula directly → produces parameter distribution.

**Important constraints for direct conversion mode:**
- Set `submodel: null` (or omit entirely)
- Only include inputs needed for the conversion formula (literature values + uncertainty assumptions)
- Do NOT include "initial condition" inputs - those are only for submodel mode

```yaml
submodel: null  # No ODE needed

empirical_data:
  inputs:
    - name: doubling_time
      value: 8.0
      units: hour
      input_type: proxy_measurement
      conversion_formula: "k = ln(2) / doubling_time"
      source_ref: Smith2020
      value_location: "Figure 2"
      value_snippet: "cells doubled every 8 hours"

  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        rng = np.random.default_rng(42)

        # Keep as Quantities
        t_double = inputs['doubling_time']
        t_sd = inputs['doubling_time_sd']

        # Lognormal for positive doubling times
        mu_log = np.log(t_double.magnitude**2 / np.sqrt(t_double.magnitude**2 + t_sd.magnitude**2))
        sigma_log = np.sqrt(np.log(1 + t_sd.magnitude**2 / t_double.magnitude**2))
        # Reattach units immediately → units propagate
        t_samples = rng.lognormal(mu_log, sigma_log, 10000) * t_double.units

        # Division inverts units automatically (hour → 1/hour)
        k_samples = np.log(2) / t_samples

        # np.median and np.percentile preserve units
        median_obs = np.array([np.median(k_samples)])
        ci95_obs = [[np.percentile(k_samples, 2.5), np.percentile(k_samples, 97.5)]]
        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}
```

### Submodel-Based (submodel provided)

Use when dynamics are complex with no analytical solution:

- **Logistic growth** (carrying capacity)
- **Multi-compartment PK**
- **Saturable kinetics** (Michaelis-Menten)
- **Multiple interacting parameters**

**Workflow:** `distribution_code` computes DATA distribution → Bayesian inference finds parameters that make submodel output ≈ data.

```yaml
submodel:
  code: |
    def submodel(t, y, params, inputs):
        N = y[0]
        k = params['k_growth']
        K = params['K_max']
        return [k * N * (1 - N/K)]  # Logistic - no closed-form solution
  # ... state_variables, parameters, etc.
```

**Rule of thumb:** If you can write `parameter = f(literature_value)` as a simple formula, use direct conversion. If you need to simulate ODEs to predict what literature measured, use a submodel.

---

## Using the Parameter Context

The **Parameter Context** section above provides all the information you need about
the parameters to calibrate:

- **Units** and **description** for each parameter
- **Direct reactions** where the parameter appears, with rate laws
- **Related species** involved in those reactions
- **Other parameters** in the same reactions (candidates for joint calibration)
- **Broader reaction network** - other reactions involving the same species, showing:
  - What else affects these species (upstream influences)
  - What these species do elsewhere in the model (downstream effects)
  - Other parameters that might be relevant for a complete submodel

Use the broader network to understand the biological context and identify whether
your submodel needs additional parameters to capture the dynamics properly.

### Example: Including Additional Parameters

If the Parameter Context shows that `k_CD8_pro` (proliferation) and `k_CD8_death`
(death) appear in related reactions, and your literature data includes both
measurements (e.g., Ki-67 for proliferation AND Annexin V for death), include
BOTH parameters in `submodel.parameters`:

```yaml
parameters:
  - k_CD8_pro    # Primary target
  - k_CD8_death  # Also needed for birth-death dynamics
```

If literature only reports net expansion (e.g., fold-change), use exponential growth
with just `k_CD8_pro` and document the limitation.

**Use exact parameter names** from the Parameter Context in your submodel.

---

## Converting Literature Data to Model Parameters

Literature rarely reports model parameters directly. Common conversion formulas are documented
in the `EstimateInput.conversion_formula` field description in the schema.

**Note:** These conversions are for **direct conversion mode** (`submodel: null`). When you have an analytical formula, apply it directly in `distribution_code` rather than building a submodel.

**Proxy measurements require careful interpretation:**

| Measurement | Meaning | Note |
|-------------|---------|------|
| Ki-67+ fraction | Cells in cycle | `Ki67 ≈ k_pro × T_cycle` (~24h cycle) |
| CFSE dilution | Division count | Requires deconvolution |
| % cytotoxicity | Killing efficiency | NOT a rate constant |

Use proxy measurements to constrain relative rates, not absolute values.

---


## Handling Context Mismatch

Experimental context often differs from model context. Always document mismatches.

### Species Scaling

| From | To | Typical Adjustment |
|------|----|--------------------|
| Mouse | Human (PK) | Allometric: CL_human ≈ CL_mouse × (70/0.025)^0.75 |
| Mouse | Human (binding) | Usually similar (within 2-5×) for conserved targets |
| Mouse | Human (cell rates) | Human often 2-5× slower |

### System Translation

| From | To | Typical Adjustment |
|------|----|--------------------|
| In vitro | In vivo | Rates may differ due to tissue context |
| Cell line | Primary cells | Primary often slower, more variable |
| Recombinant protein | Endogenous | Concentrations may differ 10-1000× |
| Peripheral blood | Tumor | TME concentrations often higher (cytokines) or lower (T cells) |

### Activation State

| From | To | Typical Adjustment |
|------|----|--------------------|
| Activated T cells | Exhausted T cells | Proliferation 5-10× lower |
| Acute infection | Chronic tumor | Sustained antigen alters kinetics |
| Resting cells | Stimulated cells | Rates may differ 10-100× |

### Documentation Requirements

Add to `key_study_limitations` for every context mismatch:
1. Source context (what was measured)
2. Target context (what model represents)
3. Expected direction of bias
4. Any scaling applied

**Example limitation:** "Proliferation rate from acute LCMV infection in mice; tumor microenvironment likely 5-10× slower due to chronic exhaustion and immunosuppression"

---

## Field-Level Guidance

Detailed guidance for individual fields is embedded in the schema itself. Key fields with
embedded documentation:

- **`sample_size`** - Where to look in papers, format, handling missing values
- **`input_type`** - Classification (direct_parameter, proxy_measurement, experimental_condition)
- **`state_variables`** - Self-contained format with provenance fields
- **`pattern`** - Standard ODE patterns (exponential_growth, birth_death, etc.)
- **`parameters`** - Include all mechanistically necessary parameters

See field descriptions in the output schema for detailed requirements and examples.

---

## Complete Examples

### Example A: T Cell Expansion (cell count trajectory)

**Parameters requested:** `k_CD8_pro`

**From the Parameter Context above:**
- `k_CD8_pro`: proliferation rate (1/day)
- Related parameter: `k_CD8_death` (death rate) in same reactions
- Reactions show proliferation and death both affect CD8 dynamics

**Literature found:** Smith et al. (2020) - In vitro T cell expansion
- "CD8+ T cells expanded from 100,000 to 750,000 ± 150,000 cells over 72 hours"
- "Anti-CD3/CD28 stimulation with IL-2"

**What we observe:** Cell count at 72 hours (net expansion only)
**What we calibrate:** Proliferation rate k_CD8_pro

**Parameter decision:** The model has separate `k_CD8_pro` and `k_CD8_death`, but this
literature only reports net expansion (no separate death measurement). We use
exponential growth with just `k_CD8_pro`. If the paper also reported apoptosis rates,
we would include `k_CD8_death` and use Pattern 5 (Birth-Death) instead.

**Submodel (Pattern 3 - Exponential Growth):**
```yaml
submodel:
  code: |
    def submodel(t, y, params, inputs):
        T = y[0]
        k_pro = params['k_CD8_pro']
        return [k_pro * T]
  pattern: exponential_growth
  state_variables:
    - name: T_cells
      units: cell
      initial_value: 100000.0
      source_ref: Smith2020
      value_location: "Methods, Cell Culture"
      value_snippet: "1×10^5 CD8+ T cells were seeded per well"
  parameters:
    - k_CD8_pro
  t_span: [0, 72]
  t_unit: hour
  observable:
    units: cell
    # Default: returns y[0] (cell count)
  identifiability_notes: "Single parameter k_pro identifiable from fold-expansion data; death rate not separable from net growth"
  rationale: "Exponential growth valid for early expansion before contact inhibition"
```

**Inputs for empirical_data (used in distribution_code):**
```yaml
empirical_data:
  inputs:
    - name: final_cell_count
      value: 750000.0
      units: cell
      input_type: direct_parameter
      description: "Mean cell count at 72 hours"
      source_ref: Smith2020
      value_location: "Figure 2A"
      value_snippet: "CD8+ T cells expanded to 7.5×10^5 ± 1.5×10^5 cells by day 3"

    - name: final_cell_count_sd
      value: 150000.0
      units: cell
      input_type: direct_parameter
      description: "SD of cell count at 72 hours"
      source_ref: Smith2020
      value_location: "Figure 2A"
      value_snippet: "CD8+ T cells expanded to 7.5×10^5 ± 1.5×10^5 cells by day 3"
```

Note: Initial cell count is now in `submodel.state_variables` with full provenance.

**Assumptions:**
```yaml
assumptions:
  - name: n_mc_samples
    value: 10000
    units: dimensionless
    description: "Monte Carlo samples for uncertainty propagation"
    rationale: "Standard sample size for stable percentile estimates"
```

**Calibration target (the DATA distribution):**
```yaml
empirical_data:
  median: [750000]
  ci95: [[456000, 1044000]]
  units: cell
  sample_size: 3
  sample_size_rationale: "n=3 replicates per condition, standard for in vitro T cell expansion assays"

  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        rng = np.random.default_rng(42)

        # Keep as Quantities, extract magnitude only when needed
        mean = inputs['final_cell_count']
        sd = inputs['final_cell_count_sd']
        n = int(inputs['n_mc_samples'].magnitude)

        # Use lognormal for positive quantities (cell counts)
        mu_log = np.log(mean.magnitude**2 / np.sqrt(mean.magnitude**2 + sd.magnitude**2))
        sigma_log = np.sqrt(np.log(1 + sd.magnitude**2 / mean.magnitude**2))
        # Reattach units immediately after sampling → units propagate
        samples = rng.lognormal(mu_log, sigma_log, n) * mean.units

        # np.median and np.percentile preserve units automatically
        median_obs = np.array([np.median(samples)])
        ci95_obs = [[np.percentile(samples, 2.5), np.percentile(samples, 97.5)]]

        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}
```

**Caveats:**
- "In vitro expansion with optimal stimulation; tumor microenvironment rates likely lower"
- "Healthy donor T cells; cancer patient T cells may have reduced proliferative capacity"

---

### Example A2: T Cell Dynamics with Death (adding parameters)

**Parameters requested:** `k_CD8_pro`

**From the Parameter Context above:**
- `k_CD8_pro`: proliferation rate
- Related parameter: `k_CD8_death` (death rate) in same reactions

**Literature found:** Chen et al. (2019) - T cell kinetics with proliferation AND death
- "Ki-67+ fraction: 45% ± 8% (proliferating cells)"
- "Annexin V+ fraction: 12% ± 3% (dying cells)"
- "Cell cycle duration: ~18 hours"

**Parameter decision:** This paper provides SEPARATE measurements for proliferation
(Ki-67) and death (Annexin V). We should include BOTH model parameters to properly
capture the dynamics, even though only `k_CD8_pro` was originally requested.

**Submodel (Pattern 5 - Birth-Death):**
```yaml
submodel:
  code: |
    def submodel(t, y, params, inputs):
        T = y[0]
        k_pro = params['k_CD8_pro']
        k_death = params['k_CD8_death']
        return [(k_pro - k_death) * T]
  pattern: birth_death
  state_variables:
    - name: T_cells
      units: cell
      initial_value: 100000.0
      source_ref: Chen2019
      value_location: "Methods, p.2"
      value_snippet: "1×10^5 T cells were seeded per well"
  parameters:
    - k_CD8_pro    # Originally requested
    - k_CD8_death  # Added - data supports separate estimation
  t_span: [0, 72]
  t_unit: hour
  observable:
    units: cell
  identifiability_notes: "Both rates identifiable: Ki-67 constrains proliferation, Annexin V constrains death"
  rationale: "Birth-death model appropriate when literature provides separate proliferation and death measurements"
```

**Key insight:** The submodel includes `k_CD8_death` even though it wasn't in the
original request, because the literature data supports estimating both rates
independently. Both parameters will be jointly inferred during Bayesian calibration.

Generate an IsolatedSystemTarget following the Pydantic schema.
