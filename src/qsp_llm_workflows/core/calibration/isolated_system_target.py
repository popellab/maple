#!/usr/bin/env python3
"""
Isolated system target model for in vitro/preclinical calibration data.

Extends CalibrationTarget with a Python submodel that approximates the full QSP model
dynamics for the isolated experimental system. The submodel uses the same parameter
names as the full model, enabling joint inference across multiple calibration targets.
"""

import ast
import warnings
from typing import List

from pydantic import Field, ValidationInfo, model_validator

from qsp_llm_workflows.core.calibration.calibration_target_models import (
    CalibrationTarget,
)
from qsp_llm_workflows.core.calibration.enums import System
from qsp_llm_workflows.core.calibration.exceptions import (
    CalibrationTargetValidationError,
    CodeStructureError,
    DimensionalityMismatchError,
)
from qsp_llm_workflows.core.calibration.observable import Submodel


class IsolatedSystemTarget(CalibrationTarget):
    """
    A calibration target from an isolated/in vitro/preclinical experimental system.

    Inherits calibration data fields from CalibrationTarget but REPLACES the observable
    field with a submodel that defines both the ODE dynamics AND how to compute the observable.

    Key differences from CalibrationTarget:
    - Uses `submodel` instead of `observable` (submodel includes nested observable)
    - The submodel.code defines an ODE that can be integrated independently
    - The submodel.observable transforms the integrated state to the experimental measurement

    Example:
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
            observable=SubmodelObservable(
                code='''
                def compute_observable(t, y, constants, ureg):
                    cells = y[0]  # same index as in ODE
                    cell_vol = constants['cell_volume']
                    ...
                    return diameter.to('micrometer')
                ''',
                units='micrometer',
                constants=[...]
            ),
            rationale='Logistic growth captures spheroid expansion dynamics...'
        )
    """

    # Override observable with submodel
    # Note: We exclude the parent's observable field
    observable: None = Field(
        default=None,
        exclude=True,
        description="Not used for IsolatedSystemTarget - use submodel instead",
    )

    submodel: Submodel = Field(
        description=(
            "Isolated submodel definition including ODE code, state variables, "
            "parameters from full model, and how to compute the observable from state."
        )
    )

    @model_validator(mode="after")
    def validate_t_span(self) -> "IsolatedSystemTarget":
        """Validate t_span is valid (t_start < t_end, both non-negative)."""
        t_start, t_end = self.submodel.t_span
        if t_start < 0:
            raise ValueError(f"submodel.t_span[0] must be non-negative, got {t_start}")
        if t_end <= t_start:
            raise ValueError(f"submodel.t_span[1] must be > t_span[0], got {self.submodel.t_span}")
        return self

    @model_validator(mode="after")
    def validate_t_unit(self) -> "IsolatedSystemTarget":
        """Validate t_unit is a valid Pint time unit."""
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
        """Validate submodel.code syntax and signature."""
        try:
            tree = ast.parse(self.submodel.code)
        except SyntaxError as e:
            raise ValueError(f"submodel.code syntax error: {e}")

        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "submodel":
                func_def = node
                break

        if not func_def:
            raise ValueError("submodel.code must define a function named 'submodel'")

        args = [arg.arg for arg in func_def.args.args]
        if args != ["t", "y", "params", "inputs"]:
            raise ValueError(
                f"submodel signature must be (t, y, params, inputs), got ({', '.join(args)})"
            )

        return self

    @model_validator(mode="after")
    def validate_state_variables(self) -> "IsolatedSystemTarget":
        """Validate state_variables is not empty."""
        if len(self.submodel.state_variables) == 0:
            raise ValueError("At least one state variable is required in submodel.state_variables")
        return self

    @model_validator(mode="after")
    def validate_initial_conditions_specified(self) -> "IsolatedSystemTarget":
        """
        Validate that every state variable has an input specifying its initial condition.

        Each state variable must have exactly one input with initializes_state pointing to it.
        """
        # Build list of state variable names
        state_var_names = [sv.name for sv in self.submodel.state_variables]

        # Build mapping from state variable to input(s) that initialize it
        state_to_inputs: dict[str, list[str]] = {sv: [] for sv in state_var_names}

        for inp in self.calibration_target_estimates.inputs:
            if inp.initializes_state is not None:
                if inp.initializes_state in state_to_inputs:
                    state_to_inputs[inp.initializes_state].append(inp.name)
                else:
                    raise ValueError(
                        f"Input '{inp.name}' has initializes_state='{inp.initializes_state}' "
                        f"which is not a valid state variable.\n"
                        f"Valid state variables: {state_var_names}"
                    )

        # Check each state variable has exactly one initial condition
        missing = [sv for sv, inps in state_to_inputs.items() if len(inps) == 0]
        if missing:
            raise ValueError(
                f"State variables missing initial conditions: {missing}\n"
                f"Each state variable must have an input with initializes_state set to it."
            )

        duplicates = {sv: inps for sv, inps in state_to_inputs.items() if len(inps) > 1}
        if duplicates:
            raise ValueError(
                f"State variables with multiple initial conditions: {duplicates}\n"
                f"Each state variable must have exactly one input providing its initial condition."
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

        # Build inputs dict and initial conditions from calibration_target_estimates.inputs
        inputs = {}
        state_var_names = [sv.name for sv in self.submodel.state_variables]
        initial_conditions: dict[str, float] = {}

        for inp in self.calibration_target_estimates.inputs:
            # Get scalar value (first element if vector)
            if isinstance(inp.value, list):
                val = inp.value[0]
            else:
                val = inp.value

            inputs[inp.name] = val

            # Track initial conditions explicitly
            if inp.initializes_state is not None:
                initial_conditions[inp.initializes_state] = val

        # Build y0 in state_variables order
        n_states = len(self.submodel.state_variables)
        y0 = [initial_conditions[sv] for sv in state_var_names]

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

        # Build inputs (as floats - inputs are experimental values)
        inputs_mock = {}
        for inp in self.calibration_target_estimates.inputs:
            if isinstance(inp.value, list):
                inputs_mock[inp.name] = inp.value[0]
            else:
                inputs_mock[inp.name] = inp.value

        # Execute submodel with Pint quantities
        try:
            local_scope: dict = {"ureg": ureg}
            exec(self.submodel.code, local_scope)
            submodel_fn = local_scope["submodel"]

            dydt = submodel_fn(1.0, y_pint, params_pint, inputs_mock)
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
        from scipy.integrate import solve_ivp
        from qsp_llm_workflows.core.unit_registry import ureg

        model_structure = self._require_model_structure(info)

        # If code is provided, validate signature
        if self.submodel.observable.code is not None:
            try:
                tree = ast.parse(self.submodel.observable.code)
            except SyntaxError as e:
                raise CodeStructureError(f"submodel.observable.code syntax error: {e}")

            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "compute_observable":
                    func_def = node
                    break

            if not func_def:
                raise CodeStructureError(
                    "submodel.observable.code must define a function named 'compute_observable'"
                )

            args = [arg.arg for arg in func_def.args.args]
            if args != ["t", "y", "constants", "ureg"]:
                raise CodeStructureError(
                    f"compute_observable signature must be (t, y, constants, ureg), "
                    f"got ({', '.join(args)})"
                )

        # Integrate ODE to get state at t_end
        param_lookup = {p.name: p for p in model_structure.parameters}
        params = {}
        for param_name in self.submodel.parameters:
            if param_name in param_lookup and param_lookup[param_name].value is not None:
                params[param_name] = param_lookup[param_name].value
            else:
                params[param_name] = 1.0

        inputs = {}
        state_var_names = [sv.name for sv in self.submodel.state_variables]
        initial_conditions: dict[str, float] = {}

        for inp in self.calibration_target_estimates.inputs:
            val = inp.value[0] if isinstance(inp.value, list) else inp.value
            inputs[inp.name] = val
            if inp.initializes_state is not None:
                initial_conditions[inp.initializes_state] = val

        y0 = [initial_conditions[sv] for sv in state_var_names]

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

    def get_parameters_used(self) -> List[str]:
        """Get the list of parameter names used in the submodel."""
        return list(self.submodel.parameters)


__all__ = ["IsolatedSystemTarget"]
