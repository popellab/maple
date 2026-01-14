#!/usr/bin/env python3
"""
Isolated system target model for in vitro/preclinical calibration data.

Extends CalibrationTarget with a Python submodel that approximates the full QSP model
dynamics for the isolated experimental system. The submodel uses the same parameter
names as the full model, enabling joint inference across multiple calibration targets.
"""

import warnings
from typing import List, Optional

from pydantic import Field, ValidationInfo, model_validator

from qsp_llm_workflows.core.calibration.calibration_target_models import (
    CalibrationTarget,
)
from qsp_llm_workflows.core.calibration.enums import Indication, System
from qsp_llm_workflows.core.calibration.exceptions import (
    CalibrationTargetValidationError,
    CodeStructureError,
    DimensionalityMismatchError,
    HardcodedConstantError,
)
from qsp_llm_workflows.core.calibration.observable import Submodel
from qsp_llm_workflows.core.calibration.shared_models import ContextMismatch


class IsolatedSystemTarget(CalibrationTarget):
    """
    A calibration target from an isolated/in vitro/preclinical experimental system.

    Supports two modes:

    1. **Direct conversion** (submodel=None): For simple analytical relationships
       - distribution_code computes the parameter directly from literature values
       - Example: k_pro = ln(2) / doubling_time
       - No ODE simulation needed

    2. **Submodel-based** (submodel provided): For complex dynamics
       - submodel.code defines an ODE that can be integrated
       - Bayesian inference finds parameters that make submodel output ≈ data
       - Use when dynamics don't have analytical solutions

    Choose direct conversion when:
    - Simple algebraic formula relates literature value to parameter
    - Examples: doubling time, half-life, Kd from binding assay

    Choose submodel when:
    - Multiple interacting parameters to estimate jointly
    - Nonlinear dynamics (logistic growth, saturation kinetics)
    - Time-course data requiring ODE fitting

    Example (submodel for logistic growth):
        submodel = Submodel(
            code='''
            def submodel(t, y, params, inputs):
                S = y[0]
                k = params['k_C1_growth']
                C_max = params['C_max']
                return [k * S * (1 - S / C_max)]
            ''',
            state_variables=[
                SubmodelStateVariable(name='spheroid_cells', units='cell')
            ],
            parameters=['k_C1_growth', 'C_max'],
            t_span=[0, 14],
            t_unit='day',
            observable=SubmodelObservable(units='micrometer', ...),
            rationale='Logistic growth captures contact inhibition...'
        )
    """

    # Override observable with submodel
    # Note: We exclude the parent's observable field
    observable: None = Field(
        default=None,
        exclude=True,
        description="Not used for IsolatedSystemTarget - use submodel instead",
    )

    submodel: Optional[Submodel] = Field(
        default=None,
        description=(
            "Optional isolated submodel for complex dynamics requiring ODE simulation.\n"
            "Set to None for direct conversion cases where distribution_code computes "
            "parameters via simple analytical formulas (e.g., k = ln(2) / t_double).\n"
            "Provide a Submodel when dynamics don't have analytical solutions and "
            "require Bayesian inference over ODE simulations."
        ),
    )

    context_mismatches: List[ContextMismatch] = Field(
        default_factory=list,
        description=(
            "Structured documentation of context mismatches between experimental data and model.\n"
            "Use this to explicitly document when the experimental context differs from the model context,\n"
            "along with expected bias direction and any adjustments applied.\n\n"
            "Example: Species mismatch (mouse data for human model), system mismatch "
            "(in vitro data for in vivo model), activation state mismatch (activated vs exhausted)."
        ),
    )

    @model_validator(mode="after")
    def validate_t_span(self) -> "IsolatedSystemTarget":
        """Validate t_span is valid (t_start < t_end, both non-negative)."""
        if self.submodel is None:
            return self
        t_start, t_end = self.submodel.t_span
        if t_start < 0:
            raise ValueError(f"submodel.t_span[0] must be non-negative, got {t_start}")
        if t_end <= t_start:
            raise ValueError(f"submodel.t_span[1] must be > t_span[0], got {self.submodel.t_span}")
        return self

    @model_validator(mode="after")
    def validate_t_unit(self) -> "IsolatedSystemTarget":
        """Validate t_unit is a valid Pint time unit."""
        if self.submodel is None:
            return self
        from qsp_llm_workflows.core.unit_registry import ureg

        try:
            t_quantity = 1.0 * ureg(self.submodel.t_unit)
        except Exception as e:
            raise ValueError(
                f"submodel.t_unit '{self.submodel.t_unit}' is not a valid Pint unit: {e}"
            )

        # Check it has time dimensionality
        if t_quantity.dimensionality != ureg.day.dimensionality:
            raise ValueError(
                f"submodel.t_unit '{self.submodel.t_unit}' must have time dimensionality, "
                f"got {t_quantity.dimensionality}"
            )

        return self

    @model_validator(mode="after")
    def validate_submodel_code(self) -> "IsolatedSystemTarget":
        """Validate submodel.code syntax and signature using unified CodeValidator."""
        if self.submodel is None:
            return self

        from qsp_llm_workflows.core.calibration.code_validator import (
            CodeType,
            validate_code_block,
        )

        result = validate_code_block(
            self.submodel.code,
            CodeType.SUBMODEL,
            check_hardcoded=False,  # Submodel code doesn't typically use ureg directly
            check_execution=False,  # We have separate validators for execution
        )

        if not result.passed:
            errors = result.get_errors()
            if errors:
                raise ValueError(errors[0].message)

        return self

    @model_validator(mode="after")
    def validate_state_variables(self) -> "IsolatedSystemTarget":
        """Validate state_variables is not empty."""
        if self.submodel is None:
            return self
        if len(self.submodel.state_variables) == 0:
            raise ValueError("At least one state variable is required in submodel.state_variables")
        return self

    @model_validator(mode="after")
    def validate_state_variable_source_refs(self) -> "IsolatedSystemTarget":
        """
        Validate that state variable source_refs point to defined sources.

        Each SubmodelStateVariable.source_ref must match a source_tag in
        primary_data_source or secondary_data_sources.
        """
        if self.submodel is None:
            return self

        # Build set of valid source tags
        valid_tags = {self.primary_data_source.source_tag}
        valid_tags.update(s.source_tag for s in self.secondary_data_sources)

        for sv in self.submodel.state_variables:
            if sv.source_ref not in valid_tags:
                raise ValueError(
                    f"State variable '{sv.name}' has source_ref='{sv.source_ref}' "
                    f"which is not a valid source tag.\n"
                    f"Valid source tags: {sorted(valid_tags)}\n"
                    f"Add the source to primary_data_source or secondary_data_sources."
                )

        return self

    @model_validator(mode="after")
    def validate_submodel_inputs_source_refs(self) -> "IsolatedSystemTarget":
        """
        Validate that submodel.inputs source_refs point to defined sources.
        """
        if self.submodel is None or not self.submodel.inputs:
            return self

        # Build set of valid source tags
        valid_tags = {self.primary_data_source.source_tag}
        valid_tags.update(s.source_tag for s in self.secondary_data_sources)

        for inp in self.submodel.inputs:
            if inp.source_ref not in valid_tags:
                raise ValueError(
                    f"submodel.inputs '{inp.name}' has source_ref='{inp.source_ref}' "
                    f"which is not a valid source tag.\n"
                    f"Valid source tags: {sorted(valid_tags)}\n"
                    f"Add the source to primary_data_source or secondary_data_sources."
                )

        return self

    @model_validator(mode="after")
    def validate_cancer_fields_for_non_cancer_indication(self) -> "IsolatedSystemTarget":
        """
        Warn if cancer-specific fields are populated for non-cancer indications.

        When indication is 'other_disease' or not a cancer type, fields like
        stage.extent, stage.burden shouldn't be populated with cancer values.
        """
        ctx = self.experimental_context

        # Only warn for non-cancer indications
        if ctx.indication != Indication.OTHER_DISEASE:
            return self

        # Check for cancer-specific stage fields
        if ctx.stage is not None:
            warnings.warn(
                f"experimental_context.stage is set (extent={ctx.stage.extent.value}, "
                f"burden={ctx.stage.burden.value}) but indication='other_disease'.\n"
                f"Cancer staging (extent/burden) doesn't apply to non-cancer contexts.\n"
                f"Consider setting stage to null for non-cancer data.",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def validate_system_type(self) -> "IsolatedSystemTarget":
        """Warn if system type doesn't match isolated/preclinical context."""
        system = self.experimental_context.system
        valid_systems = {
            System.IN_VITRO_ORGANOID,
            System.IN_VITRO_PRIMARY_CELLS,
            System.IN_VITRO_CELL_LINE,
            System.EX_VIVO_FRESH,
            System.EX_VIVO_CULTURED,
            System.ANIMAL_IN_VIVO_ORTHOTOPIC,
            System.ANIMAL_IN_VIVO_SUBCUTANEOUS,
            System.ANIMAL_IN_VIVO_PDX,
            System.ANIMAL_IN_VIVO_GEM,
            System.ANIMAL_IN_VIVO_SYNGENEIC,
        }

        if system not in valid_systems:
            warnings.warn(
                f"IsolatedSystemTarget typically uses in vitro, ex vivo, or preclinical "
                f"systems, but got '{system.value}'.",
                UserWarning,
            )

        return self

    def _require_model_structure(self, info: ValidationInfo):
        """Helper to require model_structure in context."""
        if not info.context or "model_structure" not in info.context:
            raise ValueError(
                "model_structure is required in validation context.\n"
                "Pass context={'model_structure': ModelStructure(...), 'species_units': {...}}"
            )
        return info.context["model_structure"]

    @model_validator(mode="after")
    def validate_parameters_exist(self, info: ValidationInfo) -> "IsolatedSystemTarget":
        """
        Validate that all parameters listed in submodel.parameters exist in the full model.

        Requires context:
            model_structure: ModelStructure instance with parameter definitions
        """
        if self.submodel is None:
            return self
        model_structure = self._require_model_structure(info)

        # Get valid parameter names from model
        valid_params = {p.name for p in model_structure.parameters}

        unknown = set(self.submodel.parameters) - valid_params
        if unknown:
            raise ValueError(
                f"Unknown parameters in submodel.parameters: {sorted(unknown)}\n"
                f"These must match parameter names in the full QSP model.\n"
                f"Use the model query service to find valid parameter names."
            )

        return self

    @model_validator(mode="after")
    def validate_submodel_integrates(self, info: ValidationInfo) -> "IsolatedSystemTarget":
        """
        Validate that submodel.code can be integrated as an ODE system.

        Tests actual integration using scipy.integrate.solve_ivp to catch:
        - Runtime errors in the ODE function
        - Numerical instabilities (NaN, Inf)
        - Wrong return shape

        Requires context:
            model_structure: ModelStructure instance with parameter values
        """
        if self.submodel is None:
            return self
        from scipy.integrate import solve_ivp
        import numpy as np

        model_structure = self._require_model_structure(info)

        # Compile and extract the function
        try:
            local_scope: dict = {}
            exec(self.submodel.code, local_scope)
            submodel_fn = local_scope.get("submodel")
            if submodel_fn is None:
                raise CodeStructureError("submodel function not found after exec")
        except SyntaxError as e:
            raise CodeStructureError(f"submodel.code syntax error: {e}")
        except Exception as e:
            raise CodeStructureError(f"Failed to compile submodel.code: {e}")

        # Build params from model structure (use actual values if available)
        param_lookup = {p.name: p for p in model_structure.parameters}
        params = {}
        for param_name in self.submodel.parameters:
            if param_name in param_lookup and param_lookup[param_name].value is not None:
                params[param_name] = param_lookup[param_name].value
            else:
                params[param_name] = 1.0  # Fallback

        # Build inputs dict from submodel.inputs (experimental conditions)
        inputs = {}
        for inp in self.submodel.inputs:
            # Get scalar value (first element if vector)
            if isinstance(inp.value, list):
                val = inp.value[0]
            else:
                val = inp.value
            inputs[inp.name] = val

        # Build y0 directly from state variables' initial_value (now self-contained)
        n_states = len(self.submodel.state_variables)
        y0 = [sv.initial_value for sv in self.submodel.state_variables]

        # Create ODE function for solve_ivp (signature: f(t, y))
        def ode_func(t, y):
            return submodel_fn(t, list(y), params, inputs)

        # Test single evaluation first
        try:
            dydt = ode_func(0.0, y0)
        except Exception as e:
            raise CalibrationTargetValidationError(
                f"submodel.code failed to execute: {e}\n"
                f"Check that all params and inputs are accessed correctly."
            )

        # Check return shape
        if not hasattr(dydt, "__len__"):
            raise CodeStructureError(
                f"submodel must return a list/array of derivatives, got {type(dydt).__name__}"
            )
        if len(dydt) != n_states:
            raise CodeStructureError(
                f"submodel returned {len(dydt)} derivatives but has {n_states} state variables."
            )

        # Test actual integration over full t_span
        try:
            sol = solve_ivp(ode_func, list(self.submodel.t_span), y0, method="RK45")
        except Exception as e:
            raise CalibrationTargetValidationError(
                f"submodel.code failed during ODE integration: {e}\n"
                f"The ODE system may be stiff or have numerical issues."
            )

        if not sol.success:
            raise CalibrationTargetValidationError(
                f"ODE integration failed: {sol.message}\n"
                f"Check for numerical instabilities in the submodel."
            )

        # Check for NaN or Inf in solution
        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise CalibrationTargetValidationError(
                "ODE integration produced NaN or Inf values.\n"
                "Check for division by zero or exponential blowup in the submodel."
            )

        return self

    @model_validator(mode="after")
    def validate_dimensional_consistency(self, info: ValidationInfo) -> "IsolatedSystemTarget":
        """
        Validate that the submodel ODE is dimensionally consistent using Pint.

        Checks that d(state)/dt has units of [state_units]/[time].

        Requires context:
            model_structure: ModelStructure instance with parameter units
        """
        if self.submodel is None:
            return self
        from qsp_llm_workflows.core.unit_registry import ureg

        model_structure = self._require_model_structure(info)

        # Build parameter units lookup
        param_units = {p.name: p.units for p in model_structure.parameters}

        # Build params dict with Pint quantities
        params_pint = {}
        for param_name in self.submodel.parameters:
            units_str = param_units.get(param_name, "dimensionless")
            try:
                params_pint[param_name] = 1.0 * ureg(units_str)
            except Exception:
                params_pint[param_name] = 1.0 * ureg.dimensionless

        # Build state vector with Pint quantities
        y_pint = []
        for sv in self.submodel.state_variables:
            try:
                y_pint.append(1.0 * ureg(sv.units))
            except Exception:
                y_pint.append(1.0 * ureg.dimensionless)

        # Build inputs with Pint quantities for dimensional analysis from submodel.inputs
        inputs_pint = {}
        for inp in self.submodel.inputs:
            try:
                if isinstance(inp.value, list):
                    inputs_pint[inp.name] = inp.value[0] * ureg(inp.units)
                else:
                    inputs_pint[inp.name] = inp.value * ureg(inp.units)
            except Exception:
                # If unit parsing fails, fall back to dimensionless
                val = inp.value[0] if isinstance(inp.value, list) else inp.value
                inputs_pint[inp.name] = val * ureg.dimensionless

        # Execute submodel with Pint quantities
        try:
            local_scope: dict = {"ureg": ureg}
            exec(self.submodel.code, local_scope)
            submodel_fn = local_scope["submodel"]

            dydt = submodel_fn(1.0, y_pint, params_pint, inputs_pint)
        except Exception as e:
            # Dimensional analysis failed - this might be expected if
            # the submodel doesn't handle Pint quantities gracefully
            warnings.warn(
                f"Could not perform dimensional analysis on submodel: {e}\n"
                f"Ensure parameter and state units are correctly specified.",
                UserWarning,
            )
            return self

        # Check each derivative has correct dimensions: [state_units] / [time]
        time_unit = ureg(self.submodel.t_unit)
        for i, (deriv, sv) in enumerate(zip(dydt, self.submodel.state_variables)):
            if not hasattr(deriv, "dimensionality"):
                continue  # Skip if not a Pint quantity

            try:
                expected_units = ureg(sv.units) / time_unit
                if deriv.dimensionality != expected_units.dimensionality:
                    raise DimensionalityMismatchError(
                        f"State variable '{sv.name}' derivative has wrong dimensions:\n"
                        f"  Expected: {expected_units.dimensionality} ({sv.units}/{self.submodel.t_unit})\n"
                        f"  Got: {deriv.dimensionality}\n"
                        f"Check parameter units and ODE structure."
                    )
            except DimensionalityMismatchError:
                raise
            except Exception:
                # Unit parsing failed, skip
                pass

        return self

    @model_validator(mode="after")
    def validate_observable_code(self, info: ValidationInfo) -> "IsolatedSystemTarget":
        """
        Validate that submodel.observable.code executes correctly.

        If code is None, uses default: return y[0] * ureg(units).

        Checks:
        - Function signature is compute_observable(t, y, constants, ureg) (if code provided)
        - Code executes with mock integrated state
        - Returns a Pint Quantity with correct units

        Requires context:
            model_structure: ModelStructure instance with parameter values
        """
        if self.submodel is None:
            return self
        from scipy.integrate import solve_ivp
        from qsp_llm_workflows.core.unit_registry import ureg
        from qsp_llm_workflows.core.calibration.code_validator import (
            CodeType,
            validate_code_block,
        )

        model_structure = self._require_model_structure(info)

        # If code is provided, validate syntax and signature using CodeValidator
        if self.submodel.observable.code is not None:
            result = validate_code_block(
                self.submodel.observable.code,
                CodeType.SUBMODEL_OBSERVABLE,
                check_hardcoded=False,  # Handled by separate validator
                check_execution=False,  # We do custom execution below
            )

            if not result.passed:
                errors = result.get_errors()
                if errors:
                    raise CodeStructureError(errors[0].message)

        # Integrate ODE to get state at t_end
        param_lookup = {p.name: p for p in model_structure.parameters}
        params = {}
        for param_name in self.submodel.parameters:
            if param_name in param_lookup and param_lookup[param_name].value is not None:
                params[param_name] = param_lookup[param_name].value
            else:
                params[param_name] = 1.0

        # Build inputs from submodel.inputs
        inputs = {}
        for inp in self.submodel.inputs:
            val = inp.value[0] if isinstance(inp.value, list) else inp.value
            inputs[inp.name] = val

        # Build y0 directly from state variables' initial_value (now self-contained)
        y0 = [sv.initial_value for sv in self.submodel.state_variables]

        # Compile and run ODE
        local_scope: dict = {}
        exec(self.submodel.code, local_scope)
        submodel_fn = local_scope["submodel"]

        def ode_func(t, y):
            return submodel_fn(t, list(y), params, inputs)

        sol = solve_ivp(ode_func, list(self.submodel.t_span), y0, method="RK45")

        # Get final state as list (same format as y in ODE)
        y_final = list(sol.y[:, -1])
        t_final = sol.t[-1]

        # Build constants dict with Pint quantities
        constants = {}
        for const in self.submodel.observable.constants:
            try:
                constants[const.name] = const.value * ureg(const.units)
            except Exception as e:
                raise CalibrationTargetValidationError(
                    f"Invalid units '{const.units}' for constant '{const.name}': {e}"
                )

        # Get observable function (custom or default)
        if self.submodel.observable.code is not None:
            # Compile custom observable code
            try:
                obs_scope: dict = {}
                exec(self.submodel.observable.code, obs_scope)
                compute_obs = obs_scope["compute_observable"]
            except Exception as e:
                raise CodeStructureError(f"Failed to compile observable code: {e}")
        else:
            # Default: return y[0] with declared units
            obs_units = self.submodel.observable.units

            def compute_obs(t, y, constants, ureg):
                return y[0] * ureg(obs_units)

        try:
            result = compute_obs(t_final, y_final, constants, ureg)
        except Exception as e:
            raise CalibrationTargetValidationError(
                f"submodel.observable.code failed to execute: {e}\n"
                f"Check that y indices and constants are accessed correctly."
            )

        # Check result is a Pint Quantity
        if not hasattr(result, "units"):
            raise CalibrationTargetValidationError(
                f"compute_observable must return a Pint Quantity, got {type(result).__name__}\n"
                "Ensure the return value has units attached (e.g., return value * ureg.micrometer)"
            )

        # Check units match declared observable units
        try:
            expected_units = ureg(self.submodel.observable.units)
            if result.dimensionality != expected_units.dimensionality:
                raise DimensionalityMismatchError(
                    f"Observable output has wrong dimensions:\n"
                    f"  Expected: {expected_units.dimensionality} ({self.submodel.observable.units})\n"
                    f"  Got: {result.dimensionality}\n"
                    f"Check observable code and constants."
                )
        except DimensionalityMismatchError:
            raise
        except Exception as e:
            raise CalibrationTargetValidationError(
                f"Invalid observable units '{self.submodel.observable.units}': {e}"
            )

        return self

    @model_validator(mode="after")
    def validate_no_hardcoded_constants_in_submodel(self) -> "IsolatedSystemTarget":
        """
        Validator (ERROR): Flag hardcoded numbers with units in submodel code.

        Uses AST-based detection to find numeric constants multiplied by ureg units.

        In submodel.code (ODE), all numeric constants should come from:
        - The params dict (parameters from the full model)
        - The inputs dict (experimental conditions)

        In submodel.observable.code (if provided), all numeric constants with units
        must be declared in submodel.observable.constants and accessed via the
        constants dict.

        Allowed inline numbers:
        - Universal mathematical constants (π, e)
        - Statistical percentiles (2.5, 25, 50, 75, 97.5)
        - Small integers (0, 1, 2, 3, 4, 5)
        - Common fractions (0.5, 0.25, 0.75)
        - Common conversion factors (100, 1000)
        """
        if self.submodel is None:
            return self

        from qsp_llm_workflows.core.calibration.code_validator import find_hardcoded_constants

        all_violations = []

        # Check submodel.observable.code if provided
        if self.submodel.observable.code is not None:
            violations = find_hardcoded_constants(self.submodel.observable.code)
            for value, line, col, context in violations:
                all_violations.append((value, context, "submodel.observable.code", line))

        if all_violations:
            violation_strs = [
                f"  • {v[0]} in '{v[1]}' ({v[2]}, line {v[3]})" for v in all_violations[:5]
            ]
            raise HardcodedConstantError(
                "Hardcoded numeric constants with units found in submodel code:\n"
                + "\n".join(violation_strs)
                + (f"\n  ... and {len(all_violations) - 5} more" if len(all_violations) > 5 else "")
                + "\n\nFor submodel.observable.code, declare constants in submodel.observable.constants "
                + "and access via the constants dict.\n"
                + "Example fix:\n"
                + "  1. Add to submodel.observable.constants:\n"
                + "     - name: cell_volume\n"
                + "       value: 1766.0\n"
                + "       units: micrometer**3\n"
                + "       biological_basis: 'PDAC cell ~15 μm diameter → V = 4/3×π×(7.5)³'\n"
                + "       source_ref: modeling_assumption\n"
                + "  2. Use in code: cell_vol = constants['cell_volume']"
            )

        return self

    def get_parameters_used(self) -> List[str]:
        """Get the list of parameter names used in the submodel (empty if no submodel)."""
        if self.submodel is None:
            return []
        return list(self.submodel.parameters)


__all__ = ["IsolatedSystemTarget"]
