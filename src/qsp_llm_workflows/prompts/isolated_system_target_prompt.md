# Isolated System Target Extraction

Extract calibration data from **in vitro, ex vivo, or preclinical experiments** and define a Python ODE submodel.

**Cancer type:** {{CANCER_TYPE}}
**Observable:** {{OBSERVABLE_DESCRIPTION}}

---

## What is an Isolated System Target?

A calibration target where the experimental system is **isolated from the full physiological context** (cell culture, organoids, animal models). Unlike clinical data where we use the full QSP model, these systems need a **simplified ODE submodel** that captures the relevant dynamics.

**Key insight:** The submodel uses **exact parameter names from the full QSP model**. This enables joint Bayesian inference - when we calibrate the submodel, we're calibrating the same parameters used elsewhere in the full model.

---

## Using Model Query Tools

**CRITICAL: Query the model BEFORE writing your submodel.**

1. **`query_parameters()`** - Get parameter names, values, and units
2. **`query_reactions(compartment?, species?)`** - Get reaction equations that use those parameters
3. **`validate_entity(name, type)`** - Verify a parameter/species exists before using it

**Workflow:**
1. Call `query_parameters()` to find parameters relevant to your observable (growth rates, death rates, etc.)
2. Call `query_reactions()` to see the **actual reaction equations** that use those parameters
3. **Base your submodel ODE on those reactions** - simplify as needed for the isolated system, but preserve the mathematical form (e.g., if the full model uses logistic growth, use logistic growth)
4. Use **exact parameter names** in `submodel.parameters` - typos will fail validation

---

## Key Requirements

### Parameter Names Must Match Exactly
Your `parameters` list must use names exactly as returned by `query_parameters()`. The submodel will be integrated using actual parameter values from the full model.

### Initial Conditions via Inputs
Every state variable needs an initial condition. Use `initializes_state` on an input to link it:

```yaml
inputs:
  - name: initial_cell_count
    value: 1000.0
    units: cell
    initializes_state: spheroid_cells  # Must match a state_variables.name
```

### Observable (Often Not Needed)
`submodel.observable.code` is **optional**. If omitted, defaults to `return y[0] * ureg(units)`.

Only write observable code if you need a transformation (e.g., cell count → diameter):
```python
def compute_observable(t, y, constants, ureg):
    cells = y[0]  # same index as in ODE
    cell_vol = constants['cell_volume']  # Pint Quantity
    volume = cells * cell_vol
    ...
    return diameter.to('micrometer')
```

### Rationale Explains the Approximation
Justify why this reduced ODE captures the experimental system - what's missing from the full model and why that's acceptable for this data.

---

## Validation Checks

- **Parameter names exist** in full model (queried from model structure)
- **Submodel integrates** via `scipy.integrate.solve_ivp` without errors
- **Every state variable** has exactly one input with `initializes_state`
- **Dimensional consistency** - derivatives have units [state]/[time]
- **DOI resolves** and title matches (≥75% similarity)
- **distribution_code** executes and outputs match declared statistics

---

## Model Context

**Species:** {{MODEL_SPECIES}}
**Indication:** {{MODEL_INDICATION}}
**Compartment:** {{MODEL_COMPARTMENT}}
**System:** {{MODEL_SYSTEM}}

{{PRIMARY_SOURCE_TITLE}}

---

Generate an IsolatedSystemTarget following the Pydantic schema.
