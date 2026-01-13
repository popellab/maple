# Isolated System Target Extraction

Extract calibration data from **in vitro or preclinical experiments** and define a Python ODE submodel.

**Cancer type:** {{CANCER_TYPE}}
**Observable:** {{OBSERVABLE_DESCRIPTION}}

---

## Available Tools

Use these to query the full QSP model:

- **query_parameters()** - Get parameter names, values, units. **Use these exact names in your submodel.**
- **query_species(compartment?)** - Get species in compartments
- **query_reactions(compartment?, species?)** - Get reactions involving species

**Before writing submodel_code, call `query_parameters()` to find valid parameter names.**

---

## Key Points

1. Write `submodel_code` as a Python ODE: `def submodel(t, y, params, inputs) -> [dydt]`
2. Use **exact parameter names** from the full model (enables joint inference)
3. Each state variable needs an input with `initializes_state` pointing to it
4. Set `t_span` to cover your experimental time range

---

## Model Context

**Species:** {{MODEL_SPECIES}}
**Indication:** {{MODEL_INDICATION}}
**Compartment:** {{MODEL_COMPARTMENT}}
**System:** {{MODEL_SYSTEM}}

{{PRIMARY_SOURCE_TITLE}}

---

Generate an IsolatedSystemTarget following the Pydantic model schema.
