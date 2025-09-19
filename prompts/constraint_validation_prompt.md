# Goal

You are a research assistant helping to extract and formalize constraint validation tests for quantitative systems pharmacology (QSP) models from scientific literature.
Your task is to create **MATLAB unit test-style constraint definitions** that encode biological/clinical expectations as executable validation tests.

{{EXISTING_CONSTRAINTS}}

For this constraint, you must:

---

## Constraint Identification & Formalization
1. **Biological rationale:** Identify clear biological or clinical expectations that can be tested against model simulations.
2. **Reference data extraction:** Extract specific quantitative thresholds, ranges, or relationships from literature.
3. **Testable assertions:** Convert biological expectations into precise mathematical conditions that can be programmatically verified.

---

## MATLAB Test Implementation
4. **Model setup code:** Write MATLAB code that:
   - Loads the base model and parameter files using provided variables (`parameter_file`, `base_model_file`)
   - Configures dosing schedules using `schedule_dosing()` helper function
   - Sets up any required initial conditions or patient-specific parameters
   - Constructs the SimBiology model ready for simulation

5. **Simulation and validation code:** Write MATLAB code that:
   - Runs SimBiology simulations with appropriate time vectors
   - Extracts relevant model outputs using `select()` function
   - Implements specific validation logic with `assert()` statements
   - Provides clear error messages when constraints are violated

6. **Code requirements:**
   - Use standard SimBiology functions: `sbiosimulate()`, `select()`, `interp1()`, `trapz()`, `corr()`
   - Assume `startup.m` has been run and paths are configured
   - Use generic variable names (`parameter_file`, `base_model_file`) for reusability
   - Include appropriate patient weight and dosing parameters when relevant
   - Handle time-series data interpolation and analysis correctly

---

## Literature Source Documentation
7. **Reference extraction:** Document the specific literature sources that support each constraint:
   - Precise citation information with DOI when available
   - Exact figure/table locations where reference data appears
   - Text snippets describing the biological expectation or measured data
   - Derivation code (R/MATLAB) showing how reference values were processed

8. **Data derivation:** When reference data requires processing:
   - Provide executable R or MATLAB code in `derivation_code` sections
   - Show how raw literature values were converted to constraint thresholds
   - Document any assumptions or interpolations made
   - Include confidence intervals or uncertainty bounds when available

---

## Constraint Validation Strategy
9. **Appropriate constraint types:** Choose validation approaches that match the biological question:
   - **Envelope constraints:** For parameters that should stay within literature-reported ranges
   - **Threshold constraints:** For binary biological outcomes (e.g., efficacy thresholds)
   - **Correlation constraints:** For relationships between model variables
   - **AUC/exposure constraints:** For pharmacokinetic validation
   - **Population constraints:** For parameter sensitivity analysis across multiple samples

10. **Robust validation logic:** Design tests that are:
    - Numerically stable (handle edge cases, interpolation)
    - Biologically meaningful (test clinically relevant time points)
    - Appropriately tolerant (account for model/data uncertainty)
    - Well-documented (clear error messages for debugging)

---

## Quality & Reproducibility
11. **Literature verification:** Ensure all citations are real and accessible.
12. **Code testing:** Verify MATLAB code syntax and SimBiology function usage.
13. **Biological plausibility:** Check that constraint thresholds align with known biology.
14. **Error handling:** Include appropriate error messages and boundary condition checks.

---

## Template Structure
The constraint definition should follow this YAML structure:
- `id`: Unique identifier slug for the constraint
- `description`: Clear narrative of the biological/clinical expectation
- `model_setup_code`: MATLAB code for model construction and configuration
- `simulation_and_checks_code`: MATLAB code for simulation and validation
- `sources`: Bibliographic metadata with derivation code for reference data

---

## Provided Template
{{TEMPLATE}}

## Example
{{EXAMPLES}}

# CONSTRAINT INFORMATION

{{CONSTRAINT_INFO}}

## MODEL_CONTEXT:
{{MODEL_CONTEXT}}

Fill out the YAML constraint validation template for this biological expectation.

**IMPORTANT: Format your response as follows:**
```yaml
# Your complete YAML constraint validation here
# Fill out all sections of the template above
```

Make sure to wrap your entire YAML response in ```yaml code block tags as shown above.