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
the same reactions. Use the model query service (`query_parameters()`,
`query_reactions()`) to find exact parameter names. All parameters listed in
`submodel.parameters` will be jointly inferred during Bayesian calibration.

---

## Your Task

1. **Understand the parameters** - Query the model to see what reactions use these parameters
2. **Find relevant experimental data** - Search literature for experiments that constrain these parameters
3. **Extract the data** - Pull quantitative values with full provenance
4. **Build a submodel** - Define an ODE using these exact parameter names

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
│        samples = np.random.normal(                                          │
│            inputs['cell_count_mean'].magnitude,                             │
│            inputs['cell_count_sd'].magnitude, n)                            │
│        median_obs = [np.median(samples)] * ureg.cell                        │
│        ci95_obs = [[np.percentile(samples, 2.5), ...]]                      │
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
        t_double = inputs['doubling_time'].magnitude
        t_sd = inputs['doubling_time_sd'].magnitude
        samples = np.random.normal(t_double, t_sd, 10000)
        k_samples = np.log(2) / samples  # Direct conversion!
        # ... return median_obs, ci95_obs
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

## Using Model Query Tools

**Query the model BEFORE searching literature.**

1. **`query_parameters()`** - Get parameter units, values, descriptions
2. **`query_reactions(compartment?, species?)`** - See which reactions use these parameters
3. **`validate_entity(name, type)`** - Verify parameter names exist

### Example: Discovering Additional Parameters

If asked to calibrate `k_CD8_pro`, first query related parameters:

```python
query_parameters(name_pattern="CD8")
# Returns:
# [
#   {"name": "k_CD8_pro", "value": 0.5, "units": "1/day",
#    "description": "CD8+ T cell proliferation rate in tumor"},
#   {"name": "k_CD8_death", "value": 0.1, "units": "1/day",
#    "description": "CD8+ T cell death rate"}
# ]

query_reactions(species="CD8")
# Returns reactions where CD8 appears:
# - "V_T.CD8 -> null : k_CD8_death * V_T.CD8"
# - "V_T.CD8 -> 2*V_T.CD8 : k_CD8_pro * V_T.CD8 * IL2_signal"
```

**Decision point:** The model has separate proliferation and death rates. If your
literature data includes both (e.g., Ki-67 for proliferation AND Annexin V for death),
include BOTH parameters in `submodel.parameters`:

```yaml
parameters:
  - k_CD8_pro    # Primary target
  - k_CD8_death  # Also needed for birth-death dynamics
```

If literature only reports net expansion (e.g., fold-change), use exponential growth
with just `k_CD8_pro` and document the limitation.

### More Query Examples

```python
query_parameters(name_pattern="PD1")
# Returns:
# [
#   {"name": "Kd_PD1_PDL1", "value": 8.2, "units": "nanomolar",
#    "description": "PD-1/PD-L1 dissociation constant"},
#   {"name": "kon_PD1_PDL1", "value": 1e5, "units": "1/(molar*second)",
#    "description": "PD-1/PD-L1 association rate"}
# ]
```

Use **exact parameter names** from query results in your submodel.

---

## Converting Literature Data to Model Parameters

Literature rarely reports model parameters directly. Use these conversions in `distribution_code`.

**Note:** These conversions are for **direct conversion mode** (`submodel: null`). When you have an analytical formula, apply it directly in `distribution_code` rather than building a submodel.

### Rate Constants (units: 1/time)

| Literature Reports | Parameter Type | Conversion |
|--------------------|----------------|------------|
| Doubling time (t₂) | Growth/proliferation | `k = ln(2) / t_double` |
| Half-life (t½) | Decay/death/clearance | `k = ln(2) / t_half` |
| Fold change over time | Net rate | `k = ln(fold) / time` |
| Mean residence time (MRT) | Elimination | `k = 1 / MRT` |
| Division number over time | Division rate | `k = n_divisions / time` |

**Example:** Paper reports "CD8+ T cells doubled every 8 hours"
```python
t_double = 8  # hours
k_pro = np.log(2) / t_double  # = 0.087 /hour = 2.08 /day
```

### Binding Parameters (units: concentration or 1/time)

| Literature Reports | Parameter Type | Conversion |
|--------------------|----------------|------------|
| Kd (equilibrium dissociation) | Affinity | Direct use, or `Kd = koff / kon` |
| Ka (association constant) | Affinity | `Kd = 1 / Ka` |
| kon (association rate) | On-rate | Direct use (units: 1/(M·s) or 1/(nM·day)) |
| koff (dissociation rate) | Off-rate | Direct use, or `koff = Kd * kon` |
| IC50/EC50 | Potency | Often ≈ Kd under certain assumptions |
| Bmax | Receptor density | Convert to concentration using cell count |

**Example:** Paper reports "Kd = 8.2 nM by SPR"
```python
Kd = 8.2  # nanomolar - use directly
# If kon also reported (e.g., 1.5e5 /M/s):
kon = 1.5e5  # 1/(M*s)
koff = Kd * 1e-9 * kon  # = 1.23e-3 /s = 106 /day
```

### Pharmacokinetic Parameters

| Literature Reports | Parameter Type | Conversion |
|--------------------|----------------|------------|
| Terminal half-life | Elimination rate | `kel = ln(2) / t_half` |
| Clearance (CL) + Volume (Vd) | Elimination rate | `kel = CL / Vd` |
| Bioavailability (F) | Absorption fraction | Direct use (dimensionless, 0-1) |
| AUC | Exposure | `CL = Dose * F / AUC` |

**Example:** Paper reports "half-life of 21 days for nivolumab"
```python
t_half = 21  # days
k_el = np.log(2) / t_half  # = 0.033 /day
```

### Production/Secretion Rates (units: amount/time)

| Literature Reports | Parameter Type | Conversion |
|--------------------|----------------|------------|
| Steady-state concentration | Production rate | `k_prod = k_decay * C_ss` |
| Accumulation over time | Production rate | Slope of linear accumulation phase |
| Per-cell secretion rate | Production rate | Direct use (units: pg/cell/day) |

**Example:** Paper reports "IL-2 steady-state of 50 pg/mL with 2-hour half-life"
```python
C_ss = 50  # pg/mL
t_half = 2  # hours
k_decay = np.log(2) / t_half  # = 0.35 /hour
k_prod = k_decay * C_ss  # = 17.3 pg/mL/hour
```

### Proxy Measurements (Interpret with Caution)

| Literature Reports | What It Actually Means |
|--------------------|------------------------|
| Ki-67+ fraction | Proportion in cell cycle, NOT proliferation rate |
| BrdU+ fraction | Cells that divided during labeling window |
| CFSE dilution peaks | Number of divisions, requires deconvolution |
| % cytotoxicity at E:T | Killing efficiency, not killing rate |
| IC50 from viability assay | Potency, may differ from binding Kd |

**Ki-67 interpretation:** `Ki67_fraction ≈ k_pro * T_cycle` where T_cycle is cell cycle duration (~24h). Use to constrain relative rates, not absolute values.

---

## Submodel Design Patterns

Choose the simplest ODE structure that captures the experimental data. The `pattern` field
and `submodel.code` field descriptions include standard patterns with code examples.

**Pattern selection guide:**

| Observable Data Type | Pattern | ODE Structure |
|---------------------|---------|---------------|
| Half-life / decay curve | first_order_decay | dX/dt = -k*X |
| Steady-state + turnover | production_decay | dC/dt = k_prod - k_decay*C |
| Doubling time / fold expansion | exponential_growth | dN/dt = k*N |
| Growth curve with plateau | logistic_growth | dN/dt = k*N*(1-N/K) |
| Separate birth + death rates | birth_death | dN/dt = (k_pro - k_death)*N |
| Kd, kon, koff binding data | binding_equilibrium | Receptor-ligand ODE |
| Saturation curve / Vmax, Km | michaelis_menten | dS/dt = -Vmax*S/(Km+S) |
| Killing assay / E:T response | two_species_interaction | Coupled ODEs |
| Other (CFSE, delays, etc.) | custom | Define as needed |

**Key identifiability notes:**
- Birth-death: Only net rate identifiable unless both measured independently
- Binding: Kd identifiable from equilibrium; kon/koff require kinetic data
- Two-species: k_kill has units 1/(cell × time)

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

## Key Constraints

- **Observable code is optional** - defaults to `return y[0] * ureg(units)`
- **distribution_code returns `median_obs` and `ci95_obs`** (no `iqr_obs`)

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

## Experimental Context Guidelines

The `experimental_context` describes where and how the data was collected.

**For non-cancer data (e.g., viral infection, healthy tissue):**
- Set `indication: other_disease`
- Set `stage: null` - cancer staging (extent/burden) doesn't apply
- Document what the data represents in `study_interpretation` and `key_assumptions`

```yaml
# Example for viral infection data used to inform cancer model
experimental_context:
  species: mouse
  system: animal_in_vivo.syngeneic
  indication: other_disease  # Not cancer
  stage: null  # Cancer staging doesn't apply
  treatment: null
```

**For cancer data:** Use `indication`, `stage`, and `treatment` as appropriate.

---

## Complete Examples

### Example A: T Cell Expansion (cell count trajectory)

**Parameters requested:** `k_CD8_pro`

**First, query the model to understand related parameters:**
```python
query_parameters(name_pattern="CD8")
# Returns: k_CD8_pro (proliferation), k_CD8_death (death rate)

query_reactions(species="CD8")
# Shows: proliferation and death reactions both affect CD8 dynamics
```

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
        np.random.seed(42)

        # Sample from reported distribution (what literature observed)
        mean = inputs['final_cell_count'].magnitude
        sd = inputs['final_cell_count_sd'].magnitude
        n = int(inputs['n_mc_samples'].magnitude)

        samples = np.random.normal(mean, sd, n)
        samples = np.maximum(samples, 0)  # Cell counts can't be negative

        median_obs = np.array([np.median(samples)]) * ureg.cell
        ci95 = np.percentile(samples, [2.5, 97.5])
        ci95_obs = [[ci95[0] * ureg.cell, ci95[1] * ureg.cell]]

        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}
```

**Caveats:**
- "In vitro expansion with optimal stimulation; tumor microenvironment rates likely lower"
- "Healthy donor T cells; cancer patient T cells may have reduced proliferative capacity"

---

### Example A2: T Cell Dynamics with Death (adding parameters)

**Parameters requested:** `k_CD8_pro`

**Query reveals related parameters:**
```python
query_parameters(name_pattern="CD8")
# Returns: k_CD8_pro, k_CD8_death
```

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

---

### Example B: Binding Kinetics (PD-1/PD-L1)

**Parameters:** `Kd_PD1_PDL1`, `kon_PD1_PDL1`

**Literature found:** Cheng et al. (2013) - SPR binding study
- "Kd of 8.2 ± 0.4 nM for human PD-1/PD-L1"
- "kon of 1.5 × 10⁵ M⁻¹s⁻¹"

**Submodel (Pattern 6 - Binding Equilibrium):**
```yaml
submodel:
  code: |
    def submodel(t, y, params, inputs):
        PD1_free = y[0]
        PDL1 = inputs['PDL1_concentration']
        PD1_total = inputs['PD1_total']
        Kd = params['Kd_PD1_PDL1']
        kon = params['kon_PD1_PDL1']
        koff = Kd * kon  # derived from Kd = koff/kon

        PD1_bound = PD1_total - PD1_free
        return [-kon * PD1_free * PDL1 + koff * PD1_bound]
  pattern: binding_equilibrium
  identifiability_notes: "Kd directly measured; kon from kinetic fitting. koff = Kd × kon is derived, not independently measured."
```

**State variables (self-contained with provenance):**
```yaml
state_variables:
  - name: PD1_free
    units: nanomolar
    initial_value: 10.0  # Starts fully free at total PD-1 concentration
    source_ref: Cheng2013
    value_location: "Methods, SPR"
    value_snippet: "PD-1 immobilized at 10 nM equivalent surface density"
```

**Submodel inputs (used in ODE code):**
```yaml
submodel:
  inputs:
    - name: PD1_total
      value: 10.0
      units: nanomolar
      description: "Total PD-1 concentration in binding assay"
      source_ref: Cheng2013
      value_location: "Methods, SPR"
      value_snippet: "PD-1 immobilized at 10 nM equivalent surface density"

  - name: PDL1_concentration
    value: 100.0
    units: nanomolar
    input_type: experimental_condition
    description: "PD-L1 analyte concentration"
    source_ref: experimental_protocol
    value_location: "Methods"
    value_snippet: "PD-L1 injected at concentrations from 1-100 nM"

  - name: Kd_measured
    value: 8.2
    units: nanomolar
    input_type: direct_parameter
    description: "Equilibrium dissociation constant"
    source_ref: Cheng2013
    value_location: "Table 1"
    value_snippet: "Kd = 8.2 ± 0.4 nM"

  - name: kon_measured
    value: 1.5e5
    units: 1 / (molar * second)
    input_type: direct_parameter
    description: "Association rate constant"
    source_ref: Cheng2013
    value_location: "Table 1"
    value_snippet: "kon = 1.5 × 10⁵ M⁻¹s⁻¹"

  - name: Kd_sd
    value: 0.4
    units: nanomolar
    input_type: direct_parameter
    description: "Standard deviation of Kd measurement"
    source_ref: Cheng2013
    value_location: "Table 1"
    value_snippet: "Kd = 8.2 ± 0.4 nM"
```

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
  median: [8.2]
  ci95: [[7.4, 9.0]]
  units: nanomolar
  sample_size: 3
  sample_size_rationale: "n=3 independent SPR measurements, typical for binding affinity determination"

  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        np.random.seed(42)

        # Sample Kd from reported distribution
        Kd_mean = inputs['Kd_measured'].magnitude
        Kd_sd = inputs['Kd_sd'].magnitude
        n = int(inputs['n_mc_samples'].magnitude)

        samples = np.random.normal(Kd_mean, Kd_sd, n)
        samples = np.maximum(samples, 0.1)  # Kd must be positive

        median_obs = np.array([np.median(samples)]) * ureg.nanomolar
        ci95 = np.percentile(samples, [2.5, 97.5])
        ci95_obs = [[ci95[0] * ureg.nanomolar, ci95[1] * ureg.nanomolar]]

        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}
```

**Observable (convert to fraction bound):**
```python
def compute_observable(t, y, constants, ureg):
    PD1_free = y[0]
    PD1_total = constants['PD1_total']
    fraction_bound = 1 - (PD1_free / PD1_total.magnitude)
    return fraction_bound * ureg.dimensionless
```

**Caveats:**
- "SPR with purified proteins; cell-surface binding may differ due to membrane environment"
- "Human proteins; model uses human parameters"

---

### Example C: Pharmacokinetics (Antibody clearance)

**Parameters:** `CL_nivo`, `V_C_nivo`

**Literature found:** Bajaj et al. (2017) - Population PK of nivolumab
- "Terminal half-life of 26.7 days"
- "Clearance of 9.5 mL/h (228 mL/day)"
- "Central volume of 3.48 L"

**Submodel (Pattern 1 - First-Order Decay):**
```yaml
submodel:
  code: |
    def submodel(t, y, params, inputs):
        C = y[0]  # concentration in central compartment
        CL = params['CL_nivo']  # clearance (mL/day)
        V = params['V_C_nivo']   # central volume (mL)
        k_el = CL / V
        return [-k_el * C]
  pattern: first_order_decay
  identifiability_notes: "CL and V jointly identifiable from concentration time-course; half-life constrains CL/V ratio"
```

**State variables (self-contained):**
```yaml
state_variables:
  - name: nivo_concentration
    units: microgram / milliliter
    initial_value: 30.0
    source_ref: Bajaj2017
    value_location: "Table 2"
    value_snippet: "Cmax of 30.2 μg/mL following 3 mg/kg dose"
```

**Inputs for empirical_data:**
```yaml
empirical_data:
  inputs:
    - name: terminal_half_life
      value: 26.7
      units: day
      input_type: proxy_measurement
      description: "Terminal elimination half-life"
      source_ref: Bajaj2017
      value_location: "Results, p.5"
      value_snippet: "terminal half-life of 26.7 days (95% CI: 19.9-35.8)"
      conversion_formula: "k_el = ln(2) / t_half"

    - name: clearance
      value: 228.0
      units: milliliter / day
      input_type: direct_parameter
      description: "Typical clearance value"
      source_ref: Bajaj2017
      value_location: "Table 3"
      value_snippet: "CL = 9.5 mL/h (CV 40%)"

    - name: clearance_cv
      value: 0.40
      units: dimensionless
      input_type: direct_parameter
      description: "Coefficient of variation for clearance"
      source_ref: Bajaj2017
      value_location: "Table 3"
      value_snippet: "CL = 9.5 mL/h (CV 40%)"
```

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
  median: [228.0]
  ci95: [[109.0, 427.0]]
  units: milliliter / day
  sample_size: 365
  sample_size_rationale: "n=365 patients in population PK analysis, reported in Methods section"

  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        np.random.seed(42)

        # Sample clearance from log-normal distribution (typical for PK parameters)
        CL_mean = inputs['clearance'].magnitude
        CL_cv = inputs['clearance_cv'].magnitude
        n = int(inputs['n_mc_samples'].magnitude)

        # Log-normal: if CV is known, sigma = sqrt(ln(1 + CV^2))
        sigma = np.sqrt(np.log(1 + CL_cv**2))
        mu = np.log(CL_mean) - sigma**2 / 2  # Adjust mean for log-normal

        samples = np.random.lognormal(mu, sigma, n)

        median_obs = np.array([np.median(samples)]) * ureg('milliliter / day')
        ci95 = np.percentile(samples, [2.5, 97.5])
        ci95_obs = [[ci95[0] * ureg('milliliter / day'), ci95[1] * ureg('milliliter / day')]]

        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}
```

**Caveats:**
- "Population PK from melanoma patients; clearance may differ in PDAC"
- "One-compartment approximation; full model uses two-compartment"
- "Linear clearance assumed; target-mediated disposition not captured"

---

### Example D: Cytokine Production (IL-2 secretion)

**Parameters:** `k_IL2_secretion`, `k_IL2_clearance`

**Literature found:** Marchingo et al. (2014) - T cell IL-2 secretion
- "Activated CD8+ T cells secrete 0.5-2 pg/cell/hour"
- "IL-2 half-life in culture: ~2 hours"

**Submodel (Pattern 2 - Production + Decay):**
```yaml
submodel:
  code: |
    def submodel(t, y, params, inputs):
        IL2 = y[0]
        k_sec = params['k_IL2_secretion']  # pg/cell/hour
        k_clear = params['k_IL2_clearance']
        T_cells = inputs['T_cell_count']

        production = k_sec * T_cells
        clearance = k_clear * IL2
        return [production - clearance]
  pattern: production_decay
  identifiability_notes: "At steady-state, only ratio k_sec/k_clear identifiable. Separate measurements of secretion rate and half-life required for independent estimation."
```

**State variables (self-contained):**
```yaml
state_variables:
  - name: IL2_concentration
    units: picogram / milliliter
    initial_value: 0.0
    source_ref: Marchingo2014
    value_location: "Methods"
    value_snippet: "Cells cultured in fresh medium without exogenous IL-2"
```

**Submodel inputs (used in ODE code):**
```yaml
submodel:
  inputs:
    - name: T_cell_count
      value: 100000.0
      units: cell
      description: "Number of T cells in culture well"
      source_ref: Marchingo2014
      value_location: "Methods"
      value_snippet: "10^5 cells per well in 200 μL"
```

**Inputs for empirical_data (used in distribution_code):**
```yaml
empirical_data:
  inputs:
    - name: secretion_rate
      value: 1.0
      units: picogram / cell / hour
      input_type: direct_parameter
      description: "Per-cell IL-2 secretion rate"
      source_ref: Marchingo2014
      value_location: "Figure 2B"
      value_snippet: "secretion rates of 0.5-2 pg/cell/hour depending on stimulation"

    - name: IL2_half_life
      value: 2.0
      units: hour
      input_type: proxy_measurement
      description: "IL-2 half-life in culture"
      source_ref: Marchingo2014
      value_location: "Supplementary"
      value_snippet: "IL-2 half-life approximately 2 hours in culture conditions"
      conversion_formula: "k_clear = ln(2) / t_half"

    - name: secretion_rate_low
      value: 0.5
      units: picogram / cell / hour
      input_type: direct_parameter
      description: "Lower bound of secretion rate range"
      source_ref: Marchingo2014
      value_location: "Figure 2B"
      value_snippet: "secretion rates of 0.5-2 pg/cell/hour depending on stimulation"

    - name: secretion_rate_high
      value: 2.0
      units: picogram / cell / hour
      input_type: direct_parameter
      description: "Upper bound of secretion rate range"
      source_ref: Marchingo2014
      value_location: "Figure 2B"
      value_snippet: "secretion rates of 0.5-2 pg/cell/hour depending on stimulation"
```

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
  median: [1.0]
  ci95: [[0.52, 1.95]]
  units: picogram / cell / hour
  sample_size: 6
  sample_size_rationale: "n=6 biological replicates across different stimulation conditions, pooled to establish range"

  distribution_code: |
    def derive_distribution(inputs, ureg):
        import numpy as np
        np.random.seed(42)

        # Sample uniformly from reported range (literature gives range, not mean±SD)
        low = inputs['secretion_rate_low'].magnitude
        high = inputs['secretion_rate_high'].magnitude
        n = int(inputs['n_mc_samples'].magnitude)

        samples = np.random.uniform(low, high, n)

        median_obs = np.array([np.median(samples)]) * ureg('picogram / cell / hour')
        ci95 = np.percentile(samples, [2.5, 97.5])
        ci95_obs = [[ci95[0] * ureg('picogram / cell / hour'), ci95[1] * ureg('picogram / cell / hour')]]

        return {'median_obs': median_obs, 'ci95_obs': ci95_obs}
```

**Key Study Limitations:**
- "In vitro secretion rates; in vivo consumption by T cells not captured"
- "Half-life in culture; in vivo half-life is shorter (~10 min) due to receptor-mediated uptake"

---

## Required Rationale and Assumptions

Every calibration target MUST include the following narrative fields:

### `key_assumptions` (Required - at least one)
List ALL assumptions made in extracting or interpreting the data:
- Biological equivalence assumptions (e.g., "Mouse proliferation rates assumed similar to human")
- Statistical assumptions (e.g., "Normal distribution assumed for positive-only data")
- Measurement assumptions (e.g., "Pan-CD8 staining includes exhausted cells")

**At least ONE assumption is required.** Even simple extractions involve assumptions.

### `key_study_limitations` (Required - can be empty list)
Document limitations that affect validity or generalizability:
- Sample size limitations (e.g., "n=3, limited statistical power")
- Selection bias (e.g., "Resectable tumors only, excludes advanced cases")
- Measurement method limitations (e.g., "Values estimated from figures, not tabulated")
- Context mismatch impacts (e.g., "Mouse data used for human model")

### `study_interpretation`
Provide overall scientific interpretation:
- What the study measured and why it's relevant to calibration
- How the experimental design maps to model species/parameters
- Key methodological considerations

### `submodel.rationale` (Required for all submodels)
When providing a submodel, explain:
- Why this ODE pattern was chosen (e.g., exponential vs logistic growth)
- How it approximates the relevant full model dynamics
- What simplifications were made and why they're justified

**Example:**
```yaml
rationale: "Exponential growth valid for early expansion before contact inhibition. Single parameter k_pro matches full model's k_CD8_pro for joint inference."
```

### `submodel.observable.rationale` (Optional)
When the observable transformation is non-trivial, explain:
- Why this transformation is appropriate
- Any geometric or biological assumptions in the conversion

**Example (cell count to diameter conversion):**
```yaml
observable:
  code: |
    def compute_observable(t, y, constants, ureg):
        cells = y[0]
        cell_volume = constants['cell_volume']
        volume = cells * cell_volume
        radius = ((3 * volume) / (4 * np.pi)) ** (1/3)
        return (2 * radius).to('micrometer')
  units: micrometer
  rationale: "Spheroid diameter computed from cell count assuming spherical geometry and uniform packing"
```

---

Generate an IsolatedSystemTarget following the Pydantic schema.
