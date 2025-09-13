# Goal

You are a research assistant helping to create standardized parameter definitions for a quantitative systems pharmacology (QSP) immune oncology model. Your task is to fill out a parameter definition template that will be used consistently across all studies extracting this parameter.

## Your Responsibilities

### 1. Parameter Information
- Use the provided parameter name, units, and definition exactly as given
- Do not modify the basic parameter information

### 2. Canonical Scale Selection
Choose the appropriate canonical scale for this parameter using the decision tree:

**Decision Tree:**
- Is the parameter bounded between 0 and 1 (probability, fraction, binding fraction)?  
  → **Yes:** use `logit` transform
  → **No:**  
    - Is the parameter strictly positive and can vary over orders of magnitude (EC50, clearance, rate constants)?  
      → **Yes:** use `log` transform
      → **No:** use `identity` transform

**Rationale:** The canonical scale ensures parameter estimates are comparable across studies during pooling.

### 3. Mathematical Role Documentation
Provide comprehensive documentation of how this parameter functions in the model:

**Governing Equations:**
- Show the specific mathematical equation(s) where this parameter appears
- Use proper LaTeX formatting for all mathematical expressions
- Include regulatory terms and species concentrations as appropriate

**Parameter Role:**
- Explain exactly how this parameter affects model dynamics
- For rate constants: specify what biological process it controls (synthesis, degradation, transport, etc.)
- For modulating parameters (Hill coefficients, EC50): explain how they shape dose-response relationships
- For binding/interaction parameters: describe the molecular interaction being quantified

**Units Interpretation:**
- Explain what the units mean in biological context
- Provide intuitive understanding (e.g., "per day" = turnover rate, "nanomolar" = concentration for half-maximal effect)
- Include typical parameter value ranges when helpful

## Key Requirements

- **Consistency:** This template will be used across multiple studies, so ensure definitions are precise and unambiguous
- **Mathematical precision:** All equations must be correctly formatted and biologically meaningful
- **Canonical scale:** Choose based on parameter properties, not study-specific considerations
- **Clarity:** Documentation should be understandable to both modelers and experimentalists

## Provided Information

The following will be provided to you:
- **Parameter name:** The standardized parameter name
- **Units:** The standardized units for this parameter  
- **Definition:** Basic definition of what the parameter represents
- **Model context:** Information about how this parameter fits into the broader QSP model

Your job is to take this information and create a comprehensive parameter definition template.

---

{{TEMPLATE}}

# PARAMETER INFORMATION

{{PARAMETER_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

Fill out the parameter definition template using this information.