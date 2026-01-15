# Isolated System Target Extraction

Find experimental data to calibrate the specified model parameters and define a Python ODE submodel.

**Parameters to calibrate:** {{PARAMETERS}}
{{#NOTES}}**Notes:** {{NOTES}}{{/NOTES}}

---

## Model Context

{{MODEL_CONTEXT}}

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
│  inputs (LiteratureInput)                                                   │
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
│  calibration_target_estimates                                               │
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
- **Do NOT set `initializes_state`** on any input - there are no state variables to initialize
- Only include inputs needed for the conversion formula (literature values + uncertainty assumptions)
- Do NOT include "initial condition" inputs - those are only for submodel mode

```yaml
submodel: null  # No ODE needed

calibration_target_estimates:
  inputs:
    - name: doubling_time
      value: 8.0
      units: hour
      input_type: proxy_measurement
      conversion_formula: "k = ln(2) / doubling_time"
      # NO initializes_state field!
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

### Example Query Output

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

Choose the simplest ODE structure that captures the experimental data.

### Pattern 1: First-Order Decay
**Use for:** Clearance, death, dissociation, protein degradation

```python
def submodel(t, y, params, inputs):
    X = y[0]
    k = params['k_decay']  # or k_death, k_el, koff
    return [-k * X]
```

**Solution:** X(t) = X₀ × e^(-kt)
**Data needed:** Half-life or decay time-course

---

### Pattern 2: Production + First-Order Decay
**Use for:** Cytokine steady-state, protein turnover, constant secretion

```python
def submodel(t, y, params, inputs):
    C = y[0]
    k_prod = params['k_secretion']
    k_decay = params['k_clearance']
    return [k_prod - k_decay * C]
```

**Solution:** C(t) → k_prod/k_decay at steady-state
**Data needed:** Steady-state concentration + half-life (or turnover rate)

---

### Pattern 3: Exponential Growth
**Use for:** Cell proliferation, viral replication, early tumor growth

```python
def submodel(t, y, params, inputs):
    N = y[0]
    k = params['k_growth']
    return [k * N]
```

**Solution:** N(t) = N₀ × e^(kt)
**Data needed:** Doubling time or fold-expansion over time

---

### Pattern 4: Logistic Growth
**Use for:** Tumor growth to carrying capacity, confluent cultures, spheroid expansion

```python
def submodel(t, y, params, inputs):
    N = y[0]
    k = params['k_growth']
    K = params['carrying_capacity']
    return [k * N * (1 - N / K)]
```

**Solution:** Sigmoid growth to K
**Data needed:** Growth curve with plateau, or growth rate + max size

---

### Pattern 5: Birth-Death Balance
**Use for:** Cell populations with separate proliferation and death

```python
def submodel(t, y, params, inputs):
    N = y[0]
    k_pro = params['k_proliferation']
    k_death = params['k_death']
    return [(k_pro - k_death) * N]
```

**Note:** Only identifiable if both rates measured independently. Otherwise use Pattern 3 with net rate.
**Data needed:** Separate proliferation assay (BrdU/Ki-67) + death assay (Annexin V/caspase)

---

### Pattern 6: Binding Equilibrium
**Use for:** Receptor-ligand, drug-target, antibody-antigen

```python
def submodel(t, y, params, inputs):
    R_free = y[0]  # free receptor
    L = inputs['ligand_concentration']
    R_total = inputs['total_receptor']
    kon = params['kon']
    koff = params['koff']  # or calculate: koff = Kd * kon

    R_bound = R_total - R_free
    dR_free = -kon * R_free * L + koff * R_bound
    return [dR_free]
```

**At equilibrium:** R_bound/R_total = L / (Kd + L)
**Data needed:** Kd (and optionally kon or koff), receptor expression level

---

### Pattern 7: Michaelis-Menten (Saturable Process)
**Use for:** Enzyme kinetics, receptor-mediated clearance, saturable transport

```python
def submodel(t, y, params, inputs):
    S = y[0]  # substrate concentration
    Vmax = params['Vmax']
    Km = params['Km']
    return [-Vmax * S / (Km + S)]
```

**Data needed:** Vmax and Km, or full saturation curve
**Note:** At low [S]: first-order (rate ≈ Vmax/Km × S). At high [S]: zero-order (rate ≈ Vmax).

---

### Pattern 8: Two-Species Interaction
**Use for:** Effector-target killing, predator-prey, competition

```python
def submodel(t, y, params, inputs):
    T = y[0]  # target cells
    E = y[1]  # effector cells
    k_kill = params['k_killing']
    k_growth = params['k_tumor_growth']

    dT = k_growth * T - k_kill * E * T
    dE = 0  # or model effector dynamics
    return [dT, dE]
```

**Data needed:** Killing assay at multiple E:T ratios, or time-course of target elimination
**Note:** k_kill has units 1/(cell × time) or 1/(concentration × time)

---

### Pattern 9: Programmed Expansion
**Use for:** T cell activation where cells undergo fixed number of divisions after activation

```python
def submodel(t, y, params, inputs):
    aT = y[0]  # activated cells in division program
    T = y[1]   # effector cells (completed divisions)

    k_pro = params['k_proliferation']  # rate of completing divisions
    N_div = params['n_divisions']      # number of division generations
    k_death = params['k_death']

    # Activated cells progress through division program
    d_aT = -k_pro / N_div * aT

    # Effector cells produced with 2^N amplification
    d_T = k_pro / N_div * (2**N_div) * aT - k_death * T

    return [d_aT, d_T]
```

**Key insight:** k_pro is the rate of completing the division program, NOT a simple birth rate. One activated cell produces 2^N_div effector cells.
**Data needed:** Division number assay (CFSE), time to complete expansion, fold-expansion

---

### Pattern 10: Generation-Structured
**Use for:** Tracking cell divisions via dye dilution (each division halves fluorescence)

```python
def submodel(t, y, params, inputs):
    # y[i] = cells in generation i (i=0 is undivided)
    k_div = params['k_division']
    k_death = params['k_death']
    n_gens = len(y)

    dydt = []
    for i in range(n_gens):
        # Inflow from previous generation (2x due to division)
        inflow = 2 * k_div * y[i-1] if i > 0 else 0
        # Outflow to next generation
        outflow = k_div * y[i] if i < n_gens - 1 else 0
        # Death
        death = k_death * y[i]

        dydt.append(inflow - outflow - death)

    return dydt
```

**Observable:** Mean division number = Σ(i × y[i]) / Σ(y[i])
**Data needed:** CFSE/CTV dye dilution peaks over time

---

### Pattern 11: Transit Compartment
**Use for:** Approximating a time delay without true delay differential equations

```python
def submodel(t, y, params, inputs):
    # y[0:n] = transit compartments, y[n] = output
    k_tr = params['k_transit']  # n/tau where tau is mean delay
    n_transit = inputs['n_compartments']

    dydt = [0] * (n_transit + 1)

    # Input to first compartment
    dydt[0] = inputs['input_rate'] - k_tr * y[0]

    # Transit compartments
    for i in range(1, n_transit):
        dydt[i] = k_tr * (y[i-1] - y[i])

    # Output compartment
    dydt[n_transit] = k_tr * y[n_transit-1] - params['k_elimination'] * y[n_transit]

    return dydt
```

**Key insight:** Chain of n compartments with rate k_tr = n/τ approximates gamma-distributed delay with mean τ.
**Data needed:** Lag time data, absorption profiles, maturation delays

---

### Choosing the Right Pattern

| Observable Data Type | Recommended Pattern |
|---------------------|---------------------|
| Half-life / single decay curve | Pattern 1 |
| Steady-state + turnover | Pattern 2 |
| Doubling time / fold expansion | Pattern 3 |
| Growth curve with plateau | Pattern 4 |
| Separate birth + death rates | Pattern 5 |
| Kd, kon, koff binding data | Pattern 6 |
| Saturation curve / Vmax, Km | Pattern 7 |
| Killing assay / E:T response | Pattern 8 |
| T cell expansion with fixed divisions | Pattern 9 |
| CFSE/CTV dye dilution | Pattern 10 |
| PK delay (absorption, maturation) | Pattern 11 |

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

Add to `caveats` for every context mismatch:
1. Source context (what was measured)
2. Target context (what model represents)
3. Expected direction of bias
4. Any scaling applied

**Example caveat:** "Proliferation rate from acute LCMV infection in mice; tumor microenvironment likely 5-10× slower due to chronic exhaustion and immunosuppression"

---

## Key Constraints

- **Observable code is optional** - defaults to `return y[0] * ureg(units)`
- **distribution_code returns `median_obs` and `ci95_obs`** (no `iqr_obs`)

---

## Sample Size Requirement

You MUST extract the sample size (n) for each measurement. This is critical
for proper uncertainty quantification and pooling across studies.

**Look for:**
- "n = X" or "N = X" in methods/results sections
- Sample sizes in figure legends (e.g., "n=5 per group")
- Patient/subject counts in study design
- Number of replicates in in vitro experiments
- Number of animals per group in preclinical studies

**For vector-valued data:**
- If sample size differs at each index point, provide as a list: `sample_size: [5, 5, 4, 3]`
- If same for all points, provide single int: `sample_size: 5`

**If sample size is not explicitly reported:**
- Check figure error bars - if SEM is reported, can sometimes back-calculate n from SD/SEM
- Note uncertainty in rationale: "Sample size not explicitly reported; n≈X inferred from methods"
- Use conservative estimate based on study type

**Required fields:**
- `sample_size`: int or List[int] - the numeric value(s)
- `sample_size_rationale`: str - explanation of how sample size was determined

**Example:**
```yaml
calibration_target_estimates:
  median: [750000]
  ci95: [[456000, 1044000]]
  units: cell
  sample_size: 5
  sample_size_rationale: "n=5 replicates per condition, stated in Methods section 2.3"
```

---

## Input Classification (input_type)

Each input must be classified by type:

| Type | Description | source_ref | conversion_formula |
|------|-------------|------------|-------------------|
| `direct_parameter` | Literature reports parameter directly (e.g., "k = 3/day") | Must be literature | Not needed |
| `proxy_measurement` | Requires conversion (e.g., "doubling time = 8h") | Must be literature | **Required** |
| `experimental_condition` | Protocol choice (e.g., seeding density) | Can use `experimental_protocol` | Not needed |

**Example inputs:**
```yaml
# Direct parameter - literature reports the rate directly
- name: proliferation_rate
  value: 3.0
  units: 1/day
  input_type: direct_parameter
  source_ref: DeBoer2001
  # conversion_formula not needed

# Proxy measurement - requires conversion
- name: doubling_time
  value: 8.0
  units: hour
  input_type: proxy_measurement
  source_ref: Smith2020
  conversion_formula: "k_pro = ln(2) / doubling_time"

# Experimental condition - protocol choice
- name: initial_cell_count
  value: 1000.0
  units: cell
  input_type: experimental_condition
  source_ref: experimental_protocol
  # No literature citation needed for protocol choices
```

---

## State Variable Initial Conditions

Each state variable declares which input provides its initial condition via `initial_value_input`:

```yaml
state_variables:
  - name: T_cells
    units: cell
    initial_value_input: initial_cell_count  # Must match an Input.name
  - name: tumor_cells
    units: cell
    initial_value_input: initial_tumor_size
```

The linkage is on the state variable, not the input. This makes the data flow clear.

---

## Submodel Pattern Classification

Classify your submodel using the `pattern` field:

| Pattern | Enum Value | Use For |
|---------|------------|---------|
| dX/dt = -k*X | `first_order_decay` | Clearance, death, dissociation |
| dC/dt = k_prod - k_decay*C | `production_decay` | Cytokine steady-state |
| dN/dt = k*N | `exponential_growth` | Cell proliferation |
| dN/dt = k*N*(1-N/K) | `logistic_growth` | Tumor growth with carrying capacity |
| dN/dt = (k_pro - k_death)*N | `birth_death` | Separate proliferation and death |
| Receptor-ligand binding | `binding_equilibrium` | Kd, kon, koff estimation |
| dS/dt = -Vmax*S/(Km+S) | `michaelis_menten` | Enzyme kinetics |
| Two coupled ODEs | `two_species_interaction` | Effector-target killing |
| Programmed expansion | `programmed_expansion` | T cell clonal expansion |
| Generation-structured | `generation_structured` | CFSE/CTV dye dilution |
| Transit compartments | `transit_compartment` | Delay approximation |
| Other | `custom` | Non-standard patterns |

Also add `identifiability_notes` to document which parameters can be independently estimated:

```yaml
submodel:
  pattern: birth_death
  identifiability_notes: "Only net growth rate (k_pro - k_death) identifiable from this data; individual rates require separate proliferation and death assays"
```

---

## Experimental Context Guidelines

The `experimental_context` describes where and how the data was collected.

**For non-cancer data (e.g., viral infection, healthy tissue):**
- Set `indication: other_disease`
- Set `stage: null` - cancer staging (extent/burden) doesn't apply
- Document what the data actually represents in `context_mismatches`

```yaml
# Example for viral infection data used to inform cancer model
experimental_context:
  species: mouse
  system: animal_in_vivo.syngeneic
  indication: other_disease  # Not cancer
  stage: null  # Cancer staging doesn't apply
  treatment: null  # Or describe the infection protocol in context_mismatches
```

**For cancer data:** Use `indication`, `stage`, and `treatment` as appropriate.

---

## Context Mismatch Documentation

Use `context_mismatches` to explicitly document when experimental data context differs from model context:

```yaml
context_mismatches:
  - dimension: species
    source_context: "Mouse splenocytes from LCMV infection model"
    model_context: "Human tumor-infiltrating lymphocytes in PDAC"
    expected_bias: "Proliferation rates likely 5-10× higher in acute infection vs chronic tumor"
    adjustment_applied: "No adjustment; noted as caveat"

  - dimension: system
    source_context: "In vitro anti-CD3/CD28 stimulation"
    model_context: "Tumor microenvironment"
    expected_bias: "In vitro rates typically higher due to optimal stimulation"
```

**Available dimensions:** `species`, `system`, `indication`, `compartment`, `activation_state`, `treatment`, `protein_source`, `other`

---

## Complete Examples

### Example A: T Cell Expansion (cell count trajectory)

**Parameters:** `k_CD8_pro`

**Literature found:** Smith et al. (2020) - In vitro T cell expansion
- "CD8+ T cells expanded from 100,000 to 750,000 ± 150,000 cells over 72 hours"
- "Anti-CD3/CD28 stimulation with IL-2"

**What we observe:** Cell count at 72 hours
**What we calibrate:** Proliferation rate k_CD8_pro

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
      initial_value_input: initial_T_cells
  parameters:
    - k_CD8_pro
  t_span: [0, 72]
  t_unit: hour
  observable:
    units: cell
    # Default: returns y[0] (cell count)
  identifiability_notes: "Single parameter k_pro identifiable from fold-expansion data"
  rationale: "Exponential growth valid for early expansion before contact inhibition"
```

**Inputs (from literature):**
```yaml
inputs:
  - name: initial_T_cells
    value: 100000.0
    units: cell
    input_type: experimental_condition
    description: "Initial cell seeding density"
    source_ref: Smith2020
    value_location: "Methods, Cell Culture"
    value_snippet: "1×10^5 CD8+ T cells were seeded per well"

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
calibration_target_estimates:
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

**Context mismatches:**
```yaml
context_mismatches:
  - dimension: system
    source_context: "In vitro anti-CD3/CD28 stimulation with IL-2"
    model_context: "Tumor microenvironment"
    expected_bias: "In vitro rates typically 2-5× higher due to optimal stimulation"
```

**Caveats:**
- "In vitro expansion with optimal stimulation; tumor microenvironment rates likely lower"
- "Healthy donor T cells; cancer patient T cells may have reduced proliferative capacity"

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

**State variables:**
```yaml
state_variables:
  - name: PD1_free
    units: nanomolar
    initial_value_input: PD1_total  # Starts fully free
```

**Inputs:**
```yaml
inputs:
  - name: PD1_total
    value: 10.0
    units: nanomolar
    input_type: experimental_condition
    description: "Total PD-1 concentration in binding assay"
    source_ref: experimental_protocol
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
calibration_target_estimates:
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

**Context mismatches:**
```yaml
context_mismatches:
  - dimension: system
    source_context: "SPR with purified recombinant proteins"
    model_context: "Cell-surface receptor-ligand interaction"
    expected_bias: "SPR Kd typically within 2-5× of cell-surface Kd"
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

**State variables:**
```yaml
state_variables:
  - name: nivo_concentration
    units: microgram / milliliter
    initial_value_input: initial_concentration
```

**Inputs:**
```yaml
inputs:
  - name: initial_concentration
    value: 30.0
    units: microgram / milliliter
    input_type: proxy_measurement
    description: "Cmax after 3 mg/kg IV dose - approximates initial concentration"
    source_ref: Bajaj2017
    value_location: "Table 2"
    value_snippet: "Cmax of 30.2 μg/mL following 3 mg/kg dose"
    conversion_formula: "C0 ≈ Cmax for IV bolus"

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
calibration_target_estimates:
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

**Context mismatches:**
```yaml
context_mismatches:
  - dimension: indication
    source_context: "Melanoma patients"
    model_context: "PDAC patients"
    expected_bias: "Clearance may differ ±20% between indications based on tumor burden"
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

**State variables:**
```yaml
state_variables:
  - name: IL2_concentration
    units: picogram / milliliter
    initial_value_input: initial_IL2
```

**Inputs:**
```yaml
inputs:
  - name: initial_IL2
    value: 0.0
    units: picogram / milliliter
    input_type: experimental_condition
    description: "Initial IL-2 concentration (fresh medium)"
    source_ref: experimental_protocol
    value_location: "Methods"
    value_snippet: "Cells cultured in fresh medium without exogenous IL-2"

  - name: T_cell_count
    value: 100000.0
    units: cell
    input_type: experimental_condition
    description: "Number of T cells in culture well"
    source_ref: experimental_protocol
    value_location: "Methods"
    value_snippet: "10^5 cells per well in 200 μL"

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
calibration_target_estimates:
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

**Context mismatches:**
```yaml
context_mismatches:
  - dimension: system
    source_context: "In vitro culture with anti-CD3/CD28 stimulation"
    model_context: "Tumor microenvironment"
    expected_bias: "In vitro secretion may be higher due to optimal stimulation"

  - dimension: protein_source
    source_context: "Secreted IL-2 in culture supernatant"
    model_context: "IL-2 in tumor interstitium"
    expected_bias: "In vivo half-life ~10 min (receptor-mediated uptake) vs ~2 h in culture"
    adjustment_applied: "Will need separate in vivo half-life data for model"
```

**Caveats:**
- "In vitro secretion rates; in vivo consumption by T cells not captured"
- "Half-life in culture; in vivo half-life is shorter (~10 min) due to receptor-mediated uptake"

---

Generate an IsolatedSystemTarget following the Pydantic schema.
