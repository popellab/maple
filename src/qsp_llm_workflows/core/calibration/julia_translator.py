#!/usr/bin/env python3
"""
Translator from SubmodelTarget YAML to Julia Turing.jl inference code.

This module provides tools to:
1. Load and validate YAML targets
2. Execute observation_code to get {value, sd, sd_uncertain}
3. Map model types to Julia ODE functions
4. Generate complete Julia scripts for Bayesian inference

Usage:
    from qsp_llm_workflows.core.calibration.julia_translator import JuliaTranslator

    # Create translator with model structure for unit validation
    translator = JuliaTranslator.from_model_structure_file("model_structure.json")
    julia_code = translator.generate_script("path/to/target.yaml")

    # Or for joint inference from multiple targets:
    from qsp_llm_workflows.core.calibration.julia_translator import JointInferenceBuilder

    builder = JointInferenceBuilder.from_model_structure_file("model_structure.json")
    julia_code = builder.build_from_files(["target1.yaml", "target2.yaml"])
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from qsp_llm_workflows.core.calibration.submodel_target import (
    AlgebraicModel,
    BatchAccumulationModel,
    CustomODEModel,
    ExponentialGrowthModel,
    FirstOrderDecayModel,
    InputRef,
    ReferenceRef,
    LogisticModel,
    MichaelisMentenModel,
    PriorDistribution,
    SaturationModel,
    SteadyStateConcentrationModel,
    SteadyStateDensityModel,
    SteadyStateFractionModel,
    SteadyStateProliferationIndexModel,
    SteadyStateRatioModel,
    SubmodelTarget,
    TwoStateModel,
)
from qsp_llm_workflows.core.model_structure import ModelStructure


# =============================================================================
# STEADY-STATE MODEL HELPERS
# =============================================================================


def _resolve_role(role, param_names: list[str]) -> str:
    """Resolve a ParameterRole to a Julia expression.

    - ReferenceRef -> look up in reference_values dict
    - InputRef -> look up in inputs dict
    - str matching a param name -> use directly as variable
    - str numeric literal (e.g., "1.0") -> use as constant
    - str other -> assume input name
    """
    if isinstance(role, ReferenceRef):
        return f'reference_values["{role.reference_ref}"]'
    if isinstance(role, InputRef):
        return f'inputs["{role.input_ref}"]'
    if isinstance(role, str):
        if role in param_names:
            return f'params["{role}"]'
        try:
            float(role)
            return role
        except ValueError:
            return f'inputs["{role}"]'
    return str(role)


def _generate_inputs_dict_from_target(target: "SubmodelTarget") -> str:
    """Generate Julia Dict literal for inputs from target inputs."""
    items = [f'"{inp.name}" => {inp.value}' for inp in target.inputs]
    return "Dict(" + ", ".join(items) + ")"


def _collect_reference_refs(model) -> set[str]:
    """Collect all ReferenceRef names used in a forward model's fields."""
    refs = set()
    for field_name in model.model_fields:
        value = getattr(model, field_name)
        if isinstance(value, ReferenceRef):
            refs.add(value.reference_ref)
    return refs


def _load_reference_values(yaml_path: str) -> dict[str, float]:
    """Load reference values from YAML and return name -> value mapping."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return {v["name"]: float(v["value"]) for v in data["values"]}


def _generate_reference_values_dict(model, ref_db: dict[str, float]) -> Optional[str]:
    """Generate Julia Dict literal for reference values used by the model.

    Returns None if no reference values are needed.
    """
    refs = _collect_reference_refs(model)
    if not refs:
        return None
    items = []
    for name in sorted(refs):
        if name not in ref_db:
            raise ValueError(
                f"Reference value '{name}' not found in reference database. "
                f"Available: {sorted(ref_db.keys())}"
            )
        items.append(f'"{name}" => {ref_db[name]}')
    return "Dict(" + ", ".join(items) + ")"


def _generate_steady_state_compute(
    model,
    func_name: str,
    param_names: list[str],
    inputs_code: str,
    params_dict: str,
    ref_values_code: Optional[str] = None,
) -> Optional[str]:
    """Generate Julia compute + simulate functions for steady-state model types."""
    r = lambda role: _resolve_role(role, param_names)  # noqa: E731

    # Determine if compute function needs reference_values argument
    has_refs = ref_values_code is not None
    compute_sig = "params, inputs, reference_values" if has_refs else "params, inputs"

    ucf = r(model.unit_conversion_factor)

    if isinstance(model, SteadyStateDensityModel):
        body = (
            f"    target_rate = {r(model.target_rate)}\n"
            f"    ucf = {ucf}\n"
            f"    source_pool = {r(model.source_pool)}\n"
            f"    loss_rate = {r(model.loss_rate)}\n"
            f"    section_vol = {r(model.section_volume_factor)}\n"
            f"    eff = {r(model.recruitment_efficiency)}\n"
            f"    excl = {r(model.exclusion_fraction)}\n"
            f"    return target_rate * ucf * source_pool * eff * (1 - excl) / loss_rate * section_vol"
        )
    elif isinstance(model, SteadyStateFractionModel):
        body = (
            f"    target_rate = {r(model.target_rate)}\n"
            f"    ucf = {ucf}\n"
            f"    loss_rate = {r(model.loss_rate)}\n"
            f"    parent_density = {r(model.parent_density)}\n"
            f"    drive = {r(model.drive_factor)}\n"
            f"    return target_rate * ucf * drive / (loss_rate * parent_density)"
        )
    elif isinstance(model, SteadyStateConcentrationModel):
        body = (
            f"    sec_rate = {r(model.secretion_rate)}\n"
            f"    ucf = {ucf}\n"
            f"    source = {r(model.source_count)}\n"
            f"    clearance = {r(model.clearance_rate)}\n"
            f"    vol = {r(model.distribution_volume)}\n"
            f"    return sec_rate * ucf * source / (clearance * vol)"
        )
    elif isinstance(model, SteadyStateRatioModel):
        body = (
            f"    r_num = {r(model.rate_numerator)}\n"
            f"    ucf = {ucf}\n"
            f"    r_den = {r(model.rate_denominator)}\n"
            f"    d_num = {r(model.drive_numerator)}\n"
            f"    d_den = {r(model.drive_denominator)}\n"
            f"    return r_num * ucf * d_num / (r_den * d_den)"
        )
    elif isinstance(model, SteadyStateProliferationIndexModel):
        body = (
            f"    k_prolif = {r(model.proliferation_rate)} * {ucf}\n"
            f"    t_vis = {r(model.visible_duration)}\n"
            f"    k_loss = {r(model.loss_rate)}\n"
            f"    return k_prolif * t_vis / (k_prolif * t_vis + k_loss)"
        )
    elif isinstance(model, BatchAccumulationModel):
        body = (
            f"    sec_rate = {r(model.secretion_rate)}\n"
            f"    cells = {r(model.cell_count)}\n"
            f"    t_inc = {r(model.incubation_time)}\n"
            f"    mw = {r(model.molecular_weight)}\n"
            f"    ucf = {ucf}\n"
            f"    vol = {r(model.medium_volume)}\n"
            f"    return sec_rate * cells * t_inc * mw * ucf / vol"
        )
    else:
        return None

    param_sig = ", ".join(param_names)
    ref_line = f"\n    reference_values = {ref_values_code}" if has_refs else ""
    compute_call = (
        f"compute_{func_name}(params, inputs, reference_values)"
        if has_refs
        else f"compute_{func_name}(params, inputs)"
    )
    return f"""
# Forward model ({model.type})
function compute_{func_name}({compute_sig})
{body}
end

function simulate_{func_name}({param_sig})
    params = {params_dict}
    inputs = {inputs_code}{ref_line}
    return {compute_call}
end
"""


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ObservationData:
    """Extracted observation data for a single measurement point."""

    median: float
    sigma: float
    eval_time: Optional[float]  # None for direct mode
    units: str
    sd_uncertain: bool = False  # If True, put a prior on sigma
    likelihood_distribution: str = "normal"  # "normal" or "lognormal"


@dataclass
class TargetObservations:
    """All observations for a target, ready for Julia translation."""

    target_id: str
    observations: list[ObservationData]
    t_span: Optional[tuple[float, float]]
    t_unit: Optional[str]
    initial_conditions: list[float]
    state_var_names: list[str]
    parameter_names: list[str]


# =============================================================================
# OBSERVATION EXTRACTOR
# =============================================================================


class ObservationExtractor:
    """Extract observation data from target inputs and observation_code."""

    def extract(self, target: SubmodelTarget) -> TargetObservations:
        """
        Extract all observation data from a target.

        For each error_model entry:
        - Executes observation_code to get {value, sd, sd_uncertain, n}
        """
        observations = []

        for measurement in target.calibration.measurements:
            # Build inputs dict for observation_code
            inputs_dict = self._build_inputs_dict(target, measurement.uses_inputs)

            # Execute observation_code to get {value, sd, ...}
            obs_result = self._exec_observation_code(
                measurement.observation_code,
                inputs_dict,
                measurement.sample_size,
            )

            # Extract value (observation point estimate)
            value = obs_result.get("value")
            if value is None:
                raise ValueError(
                    f"observation_code for '{measurement.name}' did not return 'value'"
                )
            median = float(value)

            # Extract sd (measurement uncertainty)
            sd_val = obs_result.get("sd")
            if sd_val is None:
                raise ValueError(f"observation_code for '{measurement.name}' did not return 'sd'")
            sigma = float(sd_val)

            sd_uncertain = obs_result.get("sd_uncertain", False)

            # Get evaluation time (for ODE models)
            eval_time = None
            if measurement.evaluation_points:
                eval_time = measurement.evaluation_points[0]

            # Get likelihood distribution type (normal or lognormal)
            likelihood_distribution = "normal"
            if measurement.likelihood and measurement.likelihood.distribution:
                likelihood_distribution = measurement.likelihood.distribution.lower()

            observations.append(
                ObservationData(
                    median=median,
                    sigma=sigma,
                    eval_time=eval_time,
                    units=measurement.units,
                    sd_uncertain=sd_uncertain,
                    likelihood_distribution=likelihood_distribution,
                )
            )

        # Extract time span and initial conditions
        t_span = None
        t_unit = None
        initial_conditions = []
        state_var_names = []

        if target.calibration.independent_variable:
            iv = target.calibration.independent_variable
            if iv.span:
                t_span = (iv.span[0], iv.span[1])
            t_unit = iv.units

        if target.calibration.state_variables:
            for sv in target.calibration.state_variables:
                state_var_names.append(sv.name)
                # Get initial condition value
                ic = sv.initial_condition
                if hasattr(ic, "value"):
                    initial_conditions.append(ic.value)
                elif hasattr(ic, "input_ref"):
                    # Look up input by name
                    for inp in target.inputs:
                        if inp.name == ic.input_ref:
                            initial_conditions.append(inp.value)
                            break

        parameter_names = [p.name for p in target.calibration.parameters]

        return TargetObservations(
            target_id=target.target_id,
            observations=observations,
            t_span=t_span,
            t_unit=t_unit,
            initial_conditions=initial_conditions,
            state_var_names=state_var_names,
            parameter_names=parameter_names,
        )

    def _build_inputs_dict(self, target: SubmodelTarget, input_names: list[str]) -> dict:
        """Build inputs dict with plain float values for observation_code."""
        inputs_dict = {}
        for inp in target.inputs:
            if inp.name in input_names or not input_names:
                inputs_dict[inp.name] = inp.value
        return inputs_dict

    def _exec_observation_code(self, code: str, inputs: dict, sample_size: Optional[int]) -> dict:
        """
        Execute observation_code and return result dict.

        The code should define: def derive_observation(inputs, sample_size) -> dict
        Required return keys: 'value', 'sd'
        Optional return keys: 'sd_uncertain', 'n'
        """
        namespace = {"inputs": inputs, "np": np}
        exec(code, namespace)

        if "derive_observation" not in namespace:
            raise ValueError("observation_code must define 'derive_observation'")

        return namespace["derive_observation"](inputs, sample_size)


# =============================================================================
# JULIA ODE MAPPER
# =============================================================================


class JuliaODEMapper:
    """Map model types to Julia ODE function code."""

    # Template for each model type
    ODE_TEMPLATES = {
        "exponential_growth": """
function {func_name}_ode!(du, u, p, t)
    {rate_param} = p[1]
    du[1] = {rate_param} * u[1]
end
""",
        "first_order_decay": """
function {func_name}_ode!(du, u, p, t)
    {rate_param} = p[1]
    du[1] = -{rate_param} * u[1]
end
""",
        "two_state": """
function {func_name}_ode!(du, u, p, t)
    {forward_rate} = p[1]
    du[1] = -{forward_rate} * u[1]  # A -> B
    du[2] = {forward_rate} * u[1]   # A -> B
end
""",
        "saturation": """
function {func_name}_ode!(du, u, p, t)
    {rate_param} = p[1]
    du[1] = {rate_param} * (1.0 - u[1])
end
""",
        "logistic": """
function {func_name}_ode!(du, u, p, t)
    {rate_param}, {capacity_param} = p
    du[1] = {rate_param} * u[1] * (1.0 - u[1] / {capacity_param})
end
""",
        "michaelis_menten": """
function {func_name}_ode!(du, u, p, t)
    {vmax_param}, {km_param} = p
    du[1] = -{vmax_param} * u[1] / ({km_param} + u[1])
end
""",
    }

    def generate_ode_function(self, target: SubmodelTarget) -> Optional[str]:
        """Generate Julia ODE function for a target's model.

        Returns None for models with analytical solutions (first_order_decay,
        exponential_growth) since these don't need numerical ODE solvers.
        """
        model = target.calibration.model
        func_name = self._sanitize_name(target.target_id)

        # Handle custom models - use code_julia directly
        if isinstance(model, CustomODEModel):
            return model.code_julia

        # Handle algebraic models - no ODE needed, uses code_julia compute function
        if isinstance(model, AlgebraicModel):
            return None

        # Steady-state types have auto-generated compute functions, no ODE needed
        if isinstance(
            model,
            (
                SteadyStateDensityModel,
                SteadyStateFractionModel,
                SteadyStateConcentrationModel,
                SteadyStateRatioModel,
                SteadyStateProliferationIndexModel,
                BatchAccumulationModel,
            ),
        ):
            return None

        # Skip ODE generation for models with analytical solutions
        if isinstance(model, (FirstOrderDecayModel, ExponentialGrowthModel)):
            return None

        # Get template for model type
        model_type = model.type
        if model_type not in self.ODE_TEMPLATES:
            raise ValueError(f"Unsupported model type: {model_type}")

        template = self.ODE_TEMPLATES[model_type]

        # Get parameter names for substitution
        params = {}
        if isinstance(model, (ExponentialGrowthModel, FirstOrderDecayModel, SaturationModel)):
            rate_param = model.rate_constant
            if isinstance(rate_param, str):
                params["rate_param"] = rate_param
            else:
                params["rate_param"] = rate_param.input_ref

        elif isinstance(model, TwoStateModel):
            forward_rate = model.forward_rate
            if isinstance(forward_rate, str):
                params["forward_rate"] = forward_rate
            else:
                params["forward_rate"] = forward_rate.input_ref

        elif isinstance(model, LogisticModel):
            rate_param = model.rate_constant
            capacity_param = model.carrying_capacity
            params["rate_param"] = (
                rate_param if isinstance(rate_param, str) else rate_param.input_ref
            )
            params["capacity_param"] = (
                capacity_param if isinstance(capacity_param, str) else capacity_param.input_ref
            )

        elif isinstance(model, MichaelisMentenModel):
            params["vmax_param"] = (
                model.vmax if isinstance(model.vmax, str) else model.vmax.input_ref
            )
            params["km_param"] = model.km if isinstance(model.km, str) else model.km.input_ref

        params["func_name"] = func_name
        return template.format(**params)

    def _sanitize_name(self, name: str) -> str:
        """Convert target_id to valid Julia function name."""
        return name.replace("-", "_").replace(".", "_")


# =============================================================================
# JULIA CODE GENERATOR
# =============================================================================


class JuliaCodeGenerator:
    """Generate complete Julia Turing.jl inference script."""

    HEADER = """# Auto-generated Julia calibration script
# Source: {source_file}
# Generated by: qsp_llm_workflows.julia_translator

using Turing
using DifferentialEquations
using Distributions
using Random

Random.seed!(42)

"""

    def __init__(self, reference_db: Optional[dict[str, float]] = None):
        self.extractor = ObservationExtractor()
        self.mapper = JuliaODEMapper()
        self.reference_db = reference_db or {}

    def generate_single_target(self, target: SubmodelTarget, source_file: str = "unknown") -> str:
        """Generate Julia code for a single target."""
        sections = []

        # Header
        sections.append(self.HEADER.format(source_file=source_file))

        # Extract observations
        obs_data = self.extractor.extract(target)

        # Generate constants
        sections.append(self._generate_constants(target, obs_data))

        # Generate ODE function (if needed)
        ode_code = self.mapper.generate_ode_function(target)
        if ode_code:
            sections.append("# ODE Model")
            sections.append(ode_code)

        # Generate simulate function
        sim_code = self._generate_simulate_function(target, obs_data)
        if sim_code:
            sections.append("# Simulation Function")
            sections.append(sim_code)

        # Generate Turing model
        sections.append("# Turing Model")
        sections.append(self._generate_turing_model(target, obs_data))

        # Generate sampling code
        sections.append("# Run Inference")
        sections.append(self._generate_sampling_code(target, obs_data))

        return "\n".join(sections)

    def _generate_constants(self, target: SubmodelTarget, obs_data: TargetObservations) -> str:
        """Generate observed data constants."""
        lines = ["# Observed Data Constants"]
        prefix = self._const_prefix(target.target_id)

        for i, obs in enumerate(obs_data.observations):
            suffix = f"_{i+1}" if len(obs_data.observations) > 1 else ""
            lines.append(f"const {prefix}_OBS_MEDIAN{suffix} = {obs.median}")
            lines.append(f"const {prefix}_OBS_SIGMA{suffix} = {obs.sigma}")
            if obs.eval_time is not None:
                lines.append(f"const {prefix}_T_EVAL{suffix} = {obs.eval_time}")

        if obs_data.t_span:
            lines.append(f"const {prefix}_T_SPAN = {obs_data.t_span}")

        if obs_data.initial_conditions:
            if len(obs_data.initial_conditions) == 1:
                lines.append(f"const {prefix}_Y0 = {obs_data.initial_conditions[0]}")
            else:
                lines.append(f"const {prefix}_Y0 = {obs_data.initial_conditions}")

        lines.append("")
        return "\n".join(lines)

    def _generate_simulate_function(
        self, target: SubmodelTarget, obs_data: TargetObservations
    ) -> Optional[str]:
        """Generate Julia simulate function."""
        model = target.calibration.model

        func_name = self.mapper._sanitize_name(target.target_id)
        prefix = self._const_prefix(target.target_id)
        param_names = obs_data.parameter_names

        # Build parameter signature
        if len(param_names) == 1:
            param_sig = param_names[0]
        else:
            param_sig = ", ".join(param_names)

        # Handle algebraic models - use code_julia compute function directly
        if isinstance(model, AlgebraicModel):
            if not model.code_julia:
                return None

            # Build inputs dict for compute function
            inputs_code = self._generate_inputs_dict(target)

            # Build params dict
            params_dict = "Dict(" + ", ".join(f'"{p}" => {p}' for p in param_names) + ")"

            # Rename compute -> compute_<func_name> to avoid collisions in combined scripts
            code_julia = model.code_julia.replace(
                "function compute(", f"function compute_{func_name}("
            )

            code = f"""
# Forward model (algebraic)
{code_julia}

function simulate_{func_name}({param_sig})
    params = {params_dict}
    inputs = {inputs_code}
    return compute_{func_name}(params, inputs)
end
"""
            return code

        # Handle steady-state typed models - auto-generate compute functions
        if isinstance(
            model,
            (
                SteadyStateDensityModel,
                SteadyStateFractionModel,
                SteadyStateConcentrationModel,
                SteadyStateRatioModel,
                SteadyStateProliferationIndexModel,
                BatchAccumulationModel,
            ),
        ):
            inputs_code = _generate_inputs_dict_from_target(target)
            params_dict = "Dict(" + ", ".join(f'"{p}" => {p}' for p in param_names) + ")"
            ref_values_code = (
                _generate_reference_values_dict(model, self.reference_db)
                if self.reference_db
                else None
            )
            return _generate_steady_state_compute(
                model, func_name, param_names, inputs_code, params_dict, ref_values_code
            )

        # Analytical solutions for simple ODE types (avoid numerical solver overhead)
        n_obs = len(obs_data.observations)

        if isinstance(model, FirstOrderDecayModel):
            rate_param = (
                model.rate_constant
                if isinstance(model.rate_constant, str)
                else model.rate_constant.input_ref
            )
            if n_obs == 1:
                return f"""
# Analytical solution for first-order decay: du/dt = -k*u => u(t) = u0 * exp(-k*t)
function simulate_{func_name}({param_sig})
    t_final = {prefix}_T_SPAN[2]
    y0 = {prefix}_Y0
    return y0 * exp(-{rate_param} * t_final)
end
"""
            else:
                eval_times = f"[{', '.join(f'{prefix}_T_EVAL_{i+1}' for i in range(n_obs))}]"
                return f"""
# Analytical solution for first-order decay: du/dt = -k*u => u(t) = u0 * exp(-k*t)
function simulate_{func_name}({param_sig})
    y0 = {prefix}_Y0
    return [y0 * exp(-{rate_param} * t) for t in {eval_times}]
end
"""

        if isinstance(model, ExponentialGrowthModel):
            rate_param = (
                model.rate_constant
                if isinstance(model.rate_constant, str)
                else model.rate_constant.input_ref
            )
            if n_obs == 1:
                return f"""
# Analytical solution for exponential growth: du/dt = k*u => u(t) = u0 * exp(k*t)
function simulate_{func_name}({param_sig})
    t_final = {prefix}_T_SPAN[2]
    y0 = {prefix}_Y0
    return y0 * exp({rate_param} * t_final)
end
"""
            else:
                eval_times = f"[{', '.join(f'{prefix}_T_EVAL_{i+1}' for i in range(n_obs))}]"
                return f"""
# Analytical solution for exponential growth: du/dt = k*u => u(t) = u0 * exp(k*t)
function simulate_{func_name}({param_sig})
    y0 = {prefix}_Y0
    return [y0 * exp({rate_param} * t) for t in {eval_times}]
end
"""

        # ODE models (numerical solver)
        param_vec = f"[{param_sig}]"

        # Build evaluation times
        if len(obs_data.observations) == 1:
            eval_times = f"[{prefix}_T_SPAN[2]]"
            return_expr = "sol[1, end]"
        else:
            eval_times = f"[{', '.join(f'{prefix}_T_EVAL_{i+1}' for i in range(len(obs_data.observations)))}]"
            return_expr = f"[sol(t)[1] for t in {eval_times}]"

        # Handle observable transformation for two_state
        if isinstance(model, TwoStateModel):
            return_expr = "sol[2, end] / (sol[1, end] + sol[2, end])"

        # Get ODE function name - for custom models, extract from code_julia
        if isinstance(model, CustomODEModel):
            import re

            match = re.search(r"function\s+([\w!]+)\s*\(", model.code_julia)
            if match:
                ode_func_name = match.group(1)
            else:
                ode_func_name = f"{func_name}_ode!"
        else:
            ode_func_name = f"{func_name}_ode!"

        code = f"""
function simulate_{func_name}({param_sig})
    prob = ODEProblem({ode_func_name}, [{prefix}_Y0...], {prefix}_T_SPAN, {param_vec})
    sol = solve(prob, AutoTsit5(Rosenbrock23());
                saveat={eval_times},
                abstol=1e-8, reltol=1e-6, maxiters=1e6)
    return {return_expr}
end
"""
        return code

    def _generate_inputs_dict(self, target: SubmodelTarget) -> str:
        """Generate Julia Dict for inputs from target inputs."""
        return _generate_inputs_dict_from_target(target)

    def _generate_turing_model(self, target: SubmodelTarget, obs_data: TargetObservations) -> str:
        """Generate Turing @model block."""
        model = target.calibration.model
        func_name = self.mapper._sanitize_name(target.target_id)

        # Check if this is a multi-output model (forward model returns dict)
        # Only AlgebraicModels can return dictionaries; ODE models return scalars
        multi_output_keys = []
        for em in target.calibration.error_model:
            if em.observable and em.observable.state_variables:
                multi_output_keys.append(em.observable.state_variables[0])
            else:
                multi_output_keys.append(None)
        is_multi_output = isinstance(model, AlgebraicModel) and any(
            k is not None for k in multi_output_keys
        )

        lines = ["@model function calibrate_{}(obs, sigma)".format(func_name)]

        # Generate priors
        lines.append("    # Priors")
        for param in target.calibration.parameters:
            prior_code = self._generate_prior(param)
            lines.append(f"    {prior_code}")

        lines.append("")
        lines.append("    # Forward model")

        param_names = [p.name for p in target.calibration.parameters]
        param_call = ", ".join(param_names)
        lines.append(f"    result = simulate_{func_name}({param_call})")

        lines.append("")
        lines.append("    # Likelihood")

        if is_multi_output:
            # Multi-output model: extract each observable from the dict
            for i, (em, key) in enumerate(zip(target.calibration.error_model, multi_output_keys)):
                if key:
                    lines.append(f'    pred_{i+1} = result["{key}"]')
                else:
                    lines.append(f"    pred_{i+1} = result")

            # Generate likelihood statements for each observation
            for i, em in enumerate(target.calibration.error_model):
                dist = em.likelihood.distribution if em.likelihood else "normal"
                if dist == "lognormal":
                    lines.append(f"    obs[{i+1}] ~ LogNormal(log(pred_{i+1}), sigma[{i+1}])")
                else:
                    lines.append(f"    obs[{i+1}] ~ Normal(pred_{i+1}, sigma[{i+1}])")
        elif len(obs_data.observations) == 1:
            obs = obs_data.observations[0]
            if obs.likelihood_distribution == "lognormal":
                lines.append("    obs ~ LogNormal(log(result), sigma)")
            else:
                lines.append("    obs ~ Normal(result, sigma)")
        else:
            # Multiple observations without multi-output keys
            is_algebraic = isinstance(model, AlgebraicModel)
            lines.append("    for i in eachindex(obs)")
            if obs_data.observations[0].likelihood_distribution == "lognormal":
                pred_expr = "result" if is_algebraic else "result[i]"
                lines.append(f"        obs[i] ~ LogNormal(log({pred_expr}), sigma[i])")
            else:
                pred_expr = "result" if is_algebraic else "result[i]"
                lines.append(f"        obs[i] ~ Normal({pred_expr}, sigma[i])")
            lines.append("    end")

        lines.append("")
        lines.append(
            f"    return ({', '.join(f'{p.name}={p.name}' for p in target.calibration.parameters)})"
        )
        lines.append("end")

        return "\n".join(lines)

    def _generate_prior(self, param) -> str:
        """Generate prior declaration for a parameter."""
        if param.prior:
            prior = param.prior
            if prior.distribution == PriorDistribution.LOGNORMAL:
                return f"{param.name} ~ LogNormal({prior.mu}, {prior.sigma})"
            elif prior.distribution == PriorDistribution.NORMAL:
                return f"{param.name} ~ Normal({prior.mu}, {prior.sigma})"
            elif prior.distribution == PriorDistribution.UNIFORM:
                return f"{param.name} ~ Uniform({prior.lower}, {prior.upper})"
            elif prior.distribution == PriorDistribution.HALF_NORMAL:
                return f"{param.name} ~ truncated(Normal(0, {prior.sigma}), 0, Inf)"
        else:
            # Default: wide log-normal prior
            return f"{param.name} ~ LogNormal(0.0, 2.0)  # Default prior"

    def _generate_sampling_code(self, target: SubmodelTarget, obs_data: TargetObservations) -> str:
        """Generate NUTS sampling code."""
        func_name = self.mapper._sanitize_name(target.target_id)
        prefix = self._const_prefix(target.target_id)

        if len(obs_data.observations) == 1:
            obs_arg = f"{prefix}_OBS_MEDIAN"
            sigma_arg = f"{prefix}_OBS_SIGMA"
        else:
            obs_arg = f"[{', '.join(f'{prefix}_OBS_MEDIAN_{i+1}' for i in range(len(obs_data.observations)))}]"
            sigma_arg = f"[{', '.join(f'{prefix}_OBS_SIGMA_{i+1}' for i in range(len(obs_data.observations)))}]"

        # Build posterior median output for each parameter
        param_lines = []
        for param in target.calibration.parameters:
            param_lines.append(
                f"let samples = chain[:{param.name}][:]\n"
                f"    med = round(median(samples), sigdigits=4)\n"
                f"    ci_lo = round(quantile(samples, 0.05), sigdigits=4)\n"
                f"    ci_hi = round(quantile(samples, 0.95), sigdigits=4)\n"
                f'    println("  {param.name}: $med ({param.units})  [90% CI: $ci_lo - $ci_hi]")\n'
                f"end"
            )
        posterior_output = "\n".join(param_lines)

        return f"""
model = calibrate_{func_name}({obs_arg}, {sigma_arg})

println("Sampling...")
chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4; progress=true)

println("\\nPosterior Summary:")
display(chain)

println("\\n" * "=" ^ 60)
println("POSTERIOR MEDIAN AND 90% CI")
println("=" ^ 60)
{posterior_output}
"""

    def _const_prefix(self, target_id: str) -> str:
        """Generate constant name prefix from target_id."""
        # Extract meaningful part: psc_proliferation_PDAC_deriv001 -> PROLIF
        parts = target_id.split("_")
        if "proliferation" in target_id:
            return "PROLIF"
        elif "death" in target_id:
            return "DEATH"
        elif "activation" in target_id:
            return "ACTIV"
        elif "recruitment" in target_id and "const" in target_id:
            return "CONST"
        elif "recruitment" in target_id and "encounter" in target_id:
            return "RECRUIT"
        else:
            return parts[1].upper()[:6] if len(parts) > 1 else "TARGET"


# =============================================================================
# MAIN TRANSLATOR
# =============================================================================


class JuliaTranslator:
    """Main translator class for YAML to Julia conversion."""

    def __init__(
        self,
        model_structure: ModelStructure,
        reference_db: Optional[dict[str, float]] = None,
    ):
        """
        Initialize translator with model structure for validation.

        Args:
            model_structure: ModelStructure instance containing parameter definitions
                            with expected units. Required for unit validation.
            reference_db: Optional dict of reference value name -> numeric value,
                         loaded from reference_values.yaml.
        """
        self.model_structure = model_structure
        self.generator = JuliaCodeGenerator()
        self.reference_db = reference_db or {}

    @classmethod
    def from_model_structure_file(
        cls,
        model_structure_path: str,
        reference_values_path: Optional[str] = None,
    ) -> "JuliaTranslator":
        """
        Create translator from a model_structure.json file.

        Args:
            model_structure_path: Path to model_structure.json file
            reference_values_path: Optional path to reference_values.yaml.
                Auto-discovers reference_values.yaml next to model_structure.json if not specified.

        Returns:
            JuliaTranslator instance
        """
        model_structure = ModelStructure.from_json(model_structure_path)
        if reference_values_path is None:
            auto_path = Path(model_structure_path).parent / "reference_values.yaml"
            if auto_path.exists():
                reference_values_path = str(auto_path)
        ref_db = _load_reference_values(reference_values_path) if reference_values_path else None
        return cls(model_structure, reference_db=ref_db)

    def load_target(self, yaml_path: str) -> SubmodelTarget:
        """Load and validate a target from YAML."""
        path = Path(yaml_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return SubmodelTarget.model_validate(
            data,
            context={"model_structure": self.model_structure, "reference_db": self.reference_db},
        )

    def generate_script(self, yaml_path: str) -> str:
        """Generate Julia script from a single YAML target."""
        target = self.load_target(yaml_path)
        return self.generator.generate_single_target(target, yaml_path)

    def generate_to_file(self, yaml_path: str, output_path: str) -> None:
        """Generate Julia script and write to file."""
        code = self.generate_script(yaml_path)
        with open(output_path, "w") as f:
            f.write(code)
        print(f"Generated: {output_path}")


# =============================================================================
# JOINT INFERENCE BUILDER
# =============================================================================


@dataclass
class ParameterInfo:
    """Aggregated info about a parameter across all targets."""

    name: str
    units: str
    prior: Optional[any]  # First encountered Prior object
    targets: list[str]  # Which targets constrain this parameter


@dataclass
class TargetInfo:
    """Info about a single target for joint inference."""

    target_id: str
    target: SubmodelTarget
    observations: TargetObservations
    parameter_names: list[str]


class JointInferenceBuilder:
    """
    Build joint inference script from multiple YAML targets.

    Automatically infers parameter sharing - parameters with the same name
    across different targets are treated as shared in the joint model.
    """

    def __init__(
        self,
        model_structure: ModelStructure,
        reference_db: Optional[dict[str, float]] = None,
    ):
        """
        Initialize builder with model structure for validation.

        Args:
            model_structure: ModelStructure instance containing parameter definitions
                            with expected units. Required for unit validation.
            reference_db: Optional dict of reference value name -> numeric value,
                         loaded from reference_values.yaml. Required when targets
                         use ReferenceRef in forward model fields.
        """
        self.model_structure = model_structure
        self.extractor = ObservationExtractor()
        self.mapper = JuliaODEMapper()
        self.reference_db = reference_db or {}

    @classmethod
    def from_model_structure_file(
        cls,
        model_structure_path: str,
        reference_values_path: Optional[str] = None,
    ) -> "JointInferenceBuilder":
        """
        Create builder from a model_structure.json file.

        Args:
            model_structure_path: Path to model_structure.json file
            reference_values_path: Optional path to reference_values.yaml

        Returns:
            JointInferenceBuilder instance
        """
        model_structure = ModelStructure.from_json(model_structure_path)
        # Auto-discover reference_values.yaml next to model_structure.json if not specified
        if reference_values_path is None:
            auto_path = Path(model_structure_path).parent / "reference_values.yaml"
            if auto_path.exists():
                reference_values_path = str(auto_path)
        ref_db = _load_reference_values(reference_values_path) if reference_values_path else None
        return cls(model_structure, reference_db=ref_db)

    def build_from_files(self, yaml_paths: list[str], force_fixed_sigma: bool = False) -> str:
        """
        Build joint inference Julia script from multiple YAML files.

        Args:
            yaml_paths: List of paths to YAML target files
            force_fixed_sigma: If True, treat all sigmas as fixed (not sampled),
                              regardless of sd_uncertain in measurement_error_code.
                              This reduces model dimensionality and speeds up sampling.

        Returns:
            Complete Julia script for joint Bayesian inference
        """
        # Load all targets
        targets_info = []
        for path in yaml_paths:
            target = self._load_target(path)
            obs_data = self.extractor.extract(target)

            # Force fixed sigma if requested
            if force_fixed_sigma:
                for obs in obs_data.observations:
                    obs.sd_uncertain = False

            targets_info.append(
                TargetInfo(
                    target_id=target.target_id,
                    target=target,
                    observations=obs_data,
                    parameter_names=[p.name for p in target.calibration.parameters],
                )
            )

        # Collect unique parameters (shared by name)
        parameters = self._collect_parameters(targets_info)

        # Generate script
        return self._generate_joint_script(targets_info, parameters)

    def build_single_all(self, yaml_paths: list[str]) -> str:
        """
        Build combined single-target inference script from multiple YAML files.

        Each target is run independently (no parameter sharing) to detect conflicts.
        Reuses the same DATA dict and simulate functions as joint inference to avoid
        const name collisions and duplicate function definitions.
        """
        # Load all targets (same as joint)
        targets_info = []
        for path in yaml_paths:
            target = self._load_target(path)
            obs_data = self.extractor.extract(target)
            for obs in obs_data.observations:
                obs.sd_uncertain = False
            targets_info.append(
                TargetInfo(
                    target_id=target.target_id,
                    target=target,
                    observations=obs_data,
                    parameter_names=[p.name for p in target.calibration.parameters],
                )
            )

        return self._generate_single_all_script(targets_info)

    def _generate_single_all_script(self, targets_info: list[TargetInfo]) -> str:
        """Generate combined single-target inference script."""
        sections = []

        # Header
        target_list = "\n".join(f"#   - {ti.target_id}" for ti in targets_info)
        sections.append(
            f"""# Combined Single-Target Inference Script
# Each target is run independently to detect conflicts between targets
# Generated by: qsp_llm_workflows.julia_translator
#
# Targets:
{target_list}

using Turing
using DifferentialEquations
using Distributions
using Statistics
using Random

Random.seed!(42)

# Results storage
results = Dict{{String, NamedTuple}}()
all_chains = Dict{{String, Any}}()
"""
        )

        # Reuse DATA dict from joint builder
        sections.append("# " + "=" * 70)
        sections.append("# OBSERVED DATA")
        sections.append("# " + "=" * 70)
        sections.append(self._generate_data_dict(targets_info))

        # Reuse ODE functions from joint builder
        sections.append("\n# " + "=" * 70)
        sections.append("# ODE SUBMODELS")
        sections.append("# " + "=" * 70)
        for ti in targets_info:
            ode_code = self.mapper.generate_ode_function(ti.target)
            if ode_code:
                sections.append(f"\n# {ti.target_id}")
                sections.append(ode_code)

        # Reuse simulate functions from joint builder
        sections.append("\n# " + "=" * 70)
        sections.append("# SIMULATION FUNCTIONS")
        sections.append("# " + "=" * 70)
        for ti in targets_info:
            sim_code = self._generate_simulate_function_dict(ti)
            if sim_code:
                sections.append(sim_code)

        # Per-target Turing models (independent, not joint)
        sections.append("\n# " + "=" * 70)
        sections.append("# SINGLE-TARGET TURING MODELS")
        sections.append("# " + "=" * 70)
        for ti in targets_info:
            sections.append(self._generate_single_turing_model(ti))

        # Per-target sampling
        sections.append("\n# " + "=" * 70)
        sections.append("# RUN INDEPENDENT INFERENCE")
        sections.append("# " + "=" * 70)
        for ti in targets_info:
            sections.append(self._generate_single_sampling(ti))

        # Summary
        sections.append(
            """
println("\\n" * "=" ^ 70)
println("SUMMARY: ALL SINGLE-TARGET POSTERIORS")
println("=" ^ 70)
for (tid, r) in sort(collect(results))
    println("$(tid): $(r.param) = $(round(r.median, sigdigits=4)) [$(round(r.ci_lo, sigdigits=4)) - $(round(r.ci_hi, sigdigits=4))] $(r.units)")
end
"""
        )

        # JSON saving -- samples keyed by target_id then parameter name
        # Build list of (target_id, [param_names]) for the Julia code
        target_params = []
        for ti in targets_info:
            pnames = [p.name for p in ti.target.calibration.parameters]
            target_params.append((ti.target_id, pnames))

        json_lines = [
            """
# ======================================================================
# SAVE RESULTS TO JSON FOR COMPARISON
# ======================================================================
using JSON

single_results = Dict{String, Any}()
single_samples = Dict{String, Any}()
"""
        ]

        for tid, pnames in target_params:
            json_lines.append(f'single_results["{tid}"] = Dict()')
            json_lines.append(f'single_samples["{tid}"] = Dict()')
            for pname in pnames:
                json_lines.append(
                    f"""let samples = vec(all_chains["{tid}"][:{pname}])
    single_results["{tid}"]["{pname}"] = Dict(
        "median" => median(samples),
        "mean" => mean(samples),
        "std" => std(samples),
        "ci_05" => quantile(samples, 0.05),
        "ci_95" => quantile(samples, 0.95),
        "ci_025" => quantile(samples, 0.025),
        "ci_975" => quantile(samples, 0.975),
        "units" => "{next(p.units for p in next(t for t in targets_info if t.target_id == tid).target.calibration.parameters if p.name == pname)}"
    )
    single_samples["{tid}"]["{pname}"] = samples
end"""
                )

        json_lines.append(
            """
open("single_inference_results.json", "w") do f
    JSON.print(f, single_results, 2)
end
println("\\nSaved: single_inference_results.json")

open("single_posterior_samples.json", "w") do f
    JSON.print(f, single_samples, 2)
end
println("Saved: single_posterior_samples.json")
"""
        )

        sections.append("\n".join(json_lines))

        return "\n".join(sections)

    def _generate_single_turing_model(self, ti: TargetInfo) -> str:
        """Generate independent Turing model for a single target."""
        model = ti.target.calibration.model
        func_name = self.mapper._sanitize_name(ti.target_id)
        n_obs = len(ti.observations.observations)

        lines = [f"\n# {ti.target_id}"]
        lines.append(f"@model function calibrate_{func_name}(data)")

        # Priors
        lines.append("    # Priors")
        for param in ti.target.calibration.parameters:
            prior_code = self._generate_prior(param.name, param.prior)
            lines.append(f"    {prior_code}")

        # Forward model
        lines.append("")
        lines.append("    # Forward model")
        if isinstance(model, AlgebraicModel):
            param_dict_items = ", ".join(f'"{p}" => {p}' for p in ti.parameter_names)
            lines.append(f"    params = Dict({param_dict_items})")
            lines.append(f'    inputs = data["{ti.target_id}"].inputs')
            lines.append(f"    pred = compute_{func_name}(params, inputs)")
        else:
            param_call = ", ".join(ti.parameter_names)
            lines.append(f"    pred = simulate_{func_name}({param_call})")

        # Likelihood
        lines.append("")
        lines.append("    # Likelihood")
        for i, obs in enumerate(ti.observations.observations):
            idx = i + 1
            sigma_expr = f'data["{ti.target_id}"].obs_sigma[{idx}]'
            if isinstance(model, AlgebraicModel) or n_obs == 1:
                pred_expr = "pred"
            else:
                pred_expr = f"pred[{idx}]"

            if obs.likelihood_distribution == "lognormal":
                lines.append(
                    f'    data["{ti.target_id}"].obs_median[{idx}] ~ LogNormal(log({pred_expr}), {sigma_expr})'
                )
            else:
                lines.append(
                    f'    data["{ti.target_id}"].obs_median[{idx}] ~ Normal({pred_expr}, {sigma_expr})'
                )

        # Return
        param_returns = ", ".join(f"{p.name}={p.name}" for p in ti.target.calibration.parameters)
        lines.append(f"\n    return ({param_returns})")
        lines.append("end")

        return "\n".join(lines)

    def _generate_single_sampling(self, ti: TargetInfo) -> str:
        """Generate sampling code for a single target."""
        func_name = self.mapper._sanitize_name(ti.target_id)
        tid = ti.target_id

        lines = ['\nprintln("\\n" * "=" ^ 70)']
        lines.append(f'println("RUNNING: {tid}")')
        lines.append('println("=" ^ 70)')
        lines.append(f"model_{func_name} = calibrate_{func_name}(DATA)")
        lines.append(
            f"chain_{func_name} = sample(model_{func_name}, NUTS(0.65), 1000; progress=false)"
        )
        lines.append(f'all_chains["{tid}"] = chain_{func_name}')

        for param in ti.target.calibration.parameters:
            lines.append(
                f"""
let samples = chain_{func_name}[:{param.name}][:]
    med = median(samples)
    ci_lo = quantile(samples, 0.05)
    ci_hi = quantile(samples, 0.95)
    results["{tid}"] = (param="{param.name}", median=med, ci_lo=ci_lo, ci_hi=ci_hi, units="{param.units}")
    println("  {param.name}: $(round(med, sigdigits=4)) ({param.units})  [90% CI: $(round(ci_lo, sigdigits=4)) - $(round(ci_hi, sigdigits=4))]")
end"""
            )

        return "\n".join(lines)

    def _load_target(self, yaml_path: str) -> SubmodelTarget:
        """Load and validate a target from YAML."""
        path = Path(yaml_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return SubmodelTarget.model_validate(
            data,
            context={"model_structure": self.model_structure, "reference_db": self.reference_db},
        )

    def _collect_parameters(self, targets_info: list[TargetInfo]) -> dict[str, ParameterInfo]:
        """
        Collect unique parameters across all targets.

        Parameters with the same name are merged - widest prior (largest sigma) is used.
        """
        parameters: dict[str, ParameterInfo] = {}

        for ti in targets_info:
            for param in ti.target.calibration.parameters:
                if param.name not in parameters:
                    parameters[param.name] = ParameterInfo(
                        name=param.name,
                        units=param.units,
                        prior=param.prior,
                        targets=[ti.target_id],
                    )
                else:
                    # Parameter already exists - add this target to list
                    parameters[param.name].targets.append(ti.target_id)
                    # Use widest prior (largest sigma for lognormal/normal)
                    existing_prior = parameters[param.name].prior
                    new_prior = param.prior
                    if existing_prior and new_prior:
                        if (
                            existing_prior.distribution == new_prior.distribution
                            and existing_prior.sigma is not None
                            and new_prior.sigma is not None
                        ):
                            if new_prior.sigma > existing_prior.sigma:
                                parameters[param.name].prior = new_prior

        return parameters

    def _generate_joint_script(
        self,
        targets_info: list[TargetInfo],
        parameters: dict[str, ParameterInfo],
    ) -> str:
        """Generate complete joint inference Julia script."""
        sections = []

        # Header
        sections.append(self._generate_header(targets_info))

        # Observed data as Dict keyed by target_id
        sections.append("# " + "=" * 70)
        sections.append("# OBSERVED DATA (Dict keyed by target_id)")
        sections.append("# " + "=" * 70)
        sections.append(self._generate_data_dict(targets_info))

        # ODE functions
        sections.append("\n# " + "=" * 70)
        sections.append("# ODE SUBMODELS")
        sections.append("# " + "=" * 70)
        for ti in targets_info:
            ode_code = self.mapper.generate_ode_function(ti.target)
            if ode_code:
                sections.append(f"\n# {ti.target_id}")
                sections.append(ode_code)

        # Simulate functions
        sections.append("\n# " + "=" * 70)
        sections.append("# SIMULATION FUNCTIONS")
        sections.append("# " + "=" * 70)
        for ti in targets_info:
            sim_code = self._generate_simulate_function_dict(ti)
            if sim_code:
                sections.append(sim_code)

        # Joint Turing model
        sections.append("\n# " + "=" * 70)
        sections.append("# JOINT TURING MODEL")
        sections.append("# " + "=" * 70)
        sections.append(self._generate_joint_turing_model_dict(targets_info, parameters))

        # Sampling code
        sections.append("\n# " + "=" * 70)
        sections.append("# RUN INFERENCE")
        sections.append("# " + "=" * 70)
        sections.append(self._generate_joint_sampling_code_dict(targets_info, parameters))

        return "\n".join(sections)

    def _generate_header(self, targets_info: list[TargetInfo]) -> str:
        """Generate script header."""
        target_list = "\n".join(f"#   - {ti.target_id}" for ti in targets_info)
        return f"""# Auto-generated Joint Inference Script
# Generated by: qsp_llm_workflows.julia_translator
#
# Targets:
{target_list}

using Turing
using DifferentialEquations
using Distributions
using StatsPlots
using Plots: mm
using Random

Random.seed!(42)
"""

    def _generate_target_constants(self, ti: TargetInfo) -> str:
        """Generate observed data constants for a target."""
        lines = [f"\n# {ti.target_id}"]
        prefix = self._const_prefix(ti.target_id)

        for i, obs in enumerate(ti.observations.observations):
            suffix = f"_{i+1}" if len(ti.observations.observations) > 1 else ""
            lines.append(f"const {prefix}_OBS_MEDIAN{suffix} = {obs.median}")
            lines.append(f"const {prefix}_OBS_SIGMA{suffix} = {obs.sigma}")
            # Add eval time constants for multi-observation targets
            if obs.eval_time is not None and len(ti.observations.observations) > 1:
                lines.append(f"const {prefix}_T_EVAL{suffix} = {obs.eval_time}")

        if ti.observations.t_span:
            lines.append(f"const {prefix}_T_SPAN = {ti.observations.t_span}")

        if ti.observations.initial_conditions:
            if len(ti.observations.initial_conditions) == 1:
                lines.append(f"const {prefix}_Y0 = {ti.observations.initial_conditions[0]}")
            else:
                lines.append(f"const {prefix}_Y0 = {ti.observations.initial_conditions}")

        return "\n".join(lines)

    def _generate_simulate_function(self, ti: TargetInfo) -> Optional[str]:
        """Generate simulate function for a target."""
        model = ti.target.calibration.model
        func_name = self.mapper._sanitize_name(ti.target_id)
        prefix = self._const_prefix(ti.target_id)
        param_names = ti.parameter_names

        # Build parameter signature
        param_sig = ", ".join(param_names)
        param_vec = f"[{param_sig}]"

        # Build evaluation expression
        n_obs = len(ti.observations.observations)
        if n_obs == 1:
            eval_times = f"[{prefix}_T_SPAN[2]]"
            if isinstance(model, TwoStateModel):
                return_expr = "sol[2, end] / (sol[1, end] + sol[2, end])"
            else:
                return_expr = "sol[1, end]"
        else:
            eval_times = f"[{', '.join(f'{prefix}_T_EVAL_{i+1}' for i in range(n_obs))}]"
            if isinstance(model, TwoStateModel):
                return_expr = f"[sol(t)[2] / (sol(t)[1] + sol(t)[2]) for t in {eval_times}]"
            else:
                return_expr = f"[sol(t)[1] for t in {eval_times}]"

        # Get ODE function name
        if isinstance(model, CustomODEModel):
            import re

            match = re.search(r"function\s+([\w!]+)\s*\(", model.code_julia)
            ode_func_name = match.group(1) if match else f"{func_name}_ode!"
        else:
            ode_func_name = f"{func_name}_ode!"

        return f"""
function simulate_{func_name}({param_sig})
    prob = ODEProblem({ode_func_name}, [{prefix}_Y0...], {prefix}_T_SPAN, {param_vec})
    sol = solve(prob, AutoTsit5(Rosenbrock23());
                saveat={eval_times},
                abstol=1e-8, reltol=1e-6, maxiters=1e6)
    return {return_expr}
end
"""

    def _generate_joint_turing_model(
        self,
        targets_info: list[TargetInfo],
        parameters: dict[str, ParameterInfo],
    ) -> str:
        """Generate joint Turing model with shared parameters."""
        lines = []

        # Build observation arguments
        obs_args = []
        for ti in targets_info:
            prefix = self._const_prefix(ti.target_id).lower()
            n_obs = len(ti.observations.observations)
            if n_obs == 1:
                obs_args.append(f"{prefix}_obs")
                obs_args.append(f"{prefix}_sigma")
            else:
                obs_args.append(f"{prefix}_obs")
                obs_args.append(f"{prefix}_sigmas")

        lines.append(f"@model function joint_calibration({', '.join(obs_args)})")

        # Priors - one per unique parameter
        lines.append("    # =========================================")
        lines.append("    # PRIORS (shared across targets)")
        lines.append("    # =========================================")
        for pname, pinfo in parameters.items():
            prior_code = self._generate_prior(pname, pinfo.prior)
            targets_str = ", ".join(pinfo.targets)
            lines.append(f"    {prior_code}  # Used by: {targets_str}")

        # Likelihoods - one section per target
        lines.append("")
        lines.append("    # =========================================")
        lines.append("    # LIKELIHOODS")
        lines.append("    # =========================================")

        for ti in targets_info:
            lines.append(f"\n    # {ti.target_id}")
            prefix = self._const_prefix(ti.target_id).lower()

            # ODE or algebraic mode
            func_name = self.mapper._sanitize_name(ti.target_id)
            param_call = ", ".join(ti.parameter_names)
            lines.append(f"    pred_{prefix} = simulate_{func_name}({param_call})")

            n_obs = len(ti.observations.observations)
            if n_obs == 1:
                lines.append(f"    {prefix}_obs ~ Normal(pred_{prefix}, {prefix}_sigma)")
            else:
                lines.append(f"    for i in eachindex({prefix}_obs)")
                lines.append(
                    f"        {prefix}_obs[i] ~ Normal(pred_{prefix}[i], {prefix}_sigmas[i])"
                )
                lines.append("    end")

        # Return statement
        lines.append("")
        param_returns = ", ".join(f"{p}={p}" for p in parameters.keys())
        lines.append(f"    return ({param_returns})")
        lines.append("end")

        return "\n".join(lines)

    def _generate_joint_sampling_code(
        self,
        targets_info: list[TargetInfo],
        parameters: dict[str, ParameterInfo],
    ) -> str:
        """Generate sampling code for joint model."""
        lines = []

        # Build observation arguments for model call
        obs_args = []
        for ti in targets_info:
            prefix = self._const_prefix(ti.target_id)
            n_obs = len(ti.observations.observations)
            if n_obs == 1:
                obs_args.append(f"{prefix}_OBS_MEDIAN")
                obs_args.append(f"{prefix}_OBS_SIGMA")
            else:
                medians = ", ".join(f"{prefix}_OBS_MEDIAN_{i+1}" for i in range(n_obs))
                sigmas = ", ".join(f"{prefix}_OBS_SIGMA_{i+1}" for i in range(n_obs))
                obs_args.append(f"[{medians}]")
                obs_args.append(f"[{sigmas}]")

        lines.append(
            f"""
model = joint_calibration({', '.join(obs_args)})

println("=" ^ 60)
println("JOINT BAYESIAN CALIBRATION")
println("=" ^ 60)
println("\\nParameters to estimate:")"""
        )

        for pname, pinfo in parameters.items():
            targets_str = ", ".join(pinfo.targets)
            lines.append(f'println("  {pname} ({pinfo.units}) - constrained by: {targets_str}")')

        # Build Julia symbol list for params
        param_symbols = ", ".join(f":{p}" for p in parameters.keys())

        lines.append(
            f"""
println("\\nSampling with NUTS...")
chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4; progress=true)

println("\\n" * "=" ^ 60)
println("POSTERIOR SUMMARY")
println("=" ^ 60)
display(chain)

# Convergence diagnostics
println("\\nConvergence diagnostics:")
params = [{param_symbols}]
for p in params
    rhat_val = rhat(chain[p])[1]
    ess_val = ess(chain[p])[1]
    status = rhat_val < 1.01 ? "✓" : "⚠"
    println("  $p: R-hat=$(round(rhat_val, digits=3)) $status, ESS=$(round(ess_val, digits=0))")
end

println("\\n" * "=" ^ 60)
println("POSTERIOR MEDIANS")
println("=" ^ 60)"""
        )

        for pname, pinfo in parameters.items():
            lines.append(
                f'println("  {pname}: $(round(median(chain[:{pname}][:]), sigdigits=4)) {pinfo.units}")'
            )

        return "\n".join(lines)

    def _generate_data_dict(self, targets_info: list[TargetInfo]) -> str:
        """Generate observed data as a Julia Dict keyed by target_id."""
        lines = ["\nconst DATA = Dict("]

        for i, ti in enumerate(targets_info):
            comma = "," if i < len(targets_info) - 1 else ""
            obs = ti.observations
            model = ti.target.calibration.model

            # Build observation arrays (convert numpy to plain Python floats)
            medians = [float(o.median) for o in obs.observations]
            sigmas = [float(o.sigma) for o in obs.observations]
            sd_uncertain = [o.sd_uncertain for o in obs.observations]
            likelihood_dists = [o.likelihood_distribution for o in obs.observations]
            # Convert Python booleans to Julia booleans (true/false)
            sd_uncertain_julia = "[" + ", ".join(str(b).lower() for b in sd_uncertain) + "]"
            likelihood_dists_julia = "[" + ", ".join(f'"{d}"' for d in likelihood_dists) + "]"
            eval_times = [float(o.eval_time) for o in obs.observations if o.eval_time is not None]

            lines.append(f'    "{ti.target_id}" => (')
            lines.append(f"        obs_median = {medians},")
            lines.append(f"        obs_sigma = {sigmas},")
            lines.append(f"        sd_uncertain = {sd_uncertain_julia},")
            lines.append(f"        likelihood = {likelihood_dists_julia},")

            if obs.t_span:
                lines.append(f"        t_span = {obs.t_span},")
            if eval_times:
                lines.append(f"        t_eval = {eval_times},")
            if obs.initial_conditions:
                lines.append(f"        y0 = {obs.initial_conditions},")

            # For algebraic models, include inputs dict
            if isinstance(model, AlgebraicModel):
                inputs_dict_items = []
                for inp in ti.target.inputs:
                    val = inp.value
                    if isinstance(val, float) and val == int(val):
                        val = int(val)
                    inputs_dict_items.append(f'"{inp.name}" => {val}')
                lines.append(f"        inputs = Dict({', '.join(inputs_dict_items)}),")

            lines.append(f"    ){comma}")

        lines.append(")\n")
        return "\n".join(lines)

    def _generate_simulate_function_dict(self, ti: TargetInfo) -> Optional[str]:
        """Generate simulate function that reads from DATA dict.

        Uses analytical solutions for simple ODE types (first_order_decay,
        exponential_growth) to avoid numerical solver overhead in the sampler.
        """
        model = ti.target.calibration.model
        func_name = self.mapper._sanitize_name(ti.target_id)
        param_names = ti.parameter_names
        param_sig = ", ".join(param_names)

        # Handle algebraic models - use code_julia compute function directly
        if isinstance(model, AlgebraicModel):
            if not model.code_julia:
                return None
            # Rename the compute function to be unique per target
            code = model.code_julia
            # Replace "function compute(" with "function compute_<func_name>("
            code = code.replace("function compute(", f"function compute_{func_name}(")
            return code

        # Handle steady-state typed models - auto-generate compute functions
        if isinstance(
            model,
            (
                SteadyStateDensityModel,
                SteadyStateFractionModel,
                SteadyStateConcentrationModel,
                SteadyStateRatioModel,
                SteadyStateProliferationIndexModel,
                BatchAccumulationModel,
            ),
        ):
            inputs_code = _generate_inputs_dict_from_target(ti.target)
            params_dict = "Dict(" + ", ".join(f'"{p}" => {p}' for p in param_names) + ")"
            ref_values_code = (
                _generate_reference_values_dict(model, self.reference_db)
                if self.reference_db
                else None
            )
            return _generate_steady_state_compute(
                model, func_name, param_names, inputs_code, params_dict, ref_values_code
            )

        # Check for analytical solutions (simple ODE types)
        n_obs = len(ti.observations.observations)

        if isinstance(model, FirstOrderDecayModel):
            # du/dt = -k*u => u(t) = u₀ * exp(-k*t)
            rate_param = (
                model.rate_constant
                if isinstance(model.rate_constant, str)
                else model.rate_constant.input_ref
            )
            if n_obs == 1:
                return f"""
# Analytical solution for first-order decay: du/dt = -k*u => u(t) = u₀ * exp(-k*t)
function simulate_{func_name}({param_sig})
    d = DATA["{ti.target_id}"]
    t_final = d.t_span[2]
    y0 = d.y0[1]
    return y0 * exp(-{rate_param} * t_final)
end
"""
            else:
                return f"""
# Analytical solution for first-order decay: du/dt = -k*u => u(t) = u₀ * exp(-k*t)
function simulate_{func_name}({param_sig})
    d = DATA["{ti.target_id}"]
    y0 = d.y0[1]
    return [y0 * exp(-{rate_param} * t) for t in d.t_eval]
end
"""

        if isinstance(model, ExponentialGrowthModel):
            # du/dt = k*u => u(t) = u₀ * exp(k*t)
            rate_param = (
                model.rate_constant
                if isinstance(model.rate_constant, str)
                else model.rate_constant.input_ref
            )
            if n_obs == 1:
                return f"""
# Analytical solution for exponential growth: du/dt = k*u => u(t) = u₀ * exp(k*t)
function simulate_{func_name}({param_sig})
    d = DATA["{ti.target_id}"]
    t_final = d.t_span[2]
    y0 = d.y0[1]
    return y0 * exp({rate_param} * t_final)
end
"""
            else:
                return f"""
# Analytical solution for exponential growth: du/dt = k*u => u(t) = u₀ * exp(k*t)
function simulate_{func_name}({param_sig})
    d = DATA["{ti.target_id}"]
    y0 = d.y0[1]
    return [y0 * exp({rate_param} * t) for t in d.t_eval]
end
"""

        # Fall back to numerical ODE solver for complex models
        param_vec = f"[{param_sig}]"

        # Build return expression based on model type and observation count
        if n_obs == 1:
            if isinstance(model, TwoStateModel):
                return_expr = "sol[2, end] / (sol[1, end] + sol[2, end])"
            else:
                return_expr = "sol[1, end]"
            saveat_expr = f'[DATA["{ti.target_id}"].t_span[2]]'
        else:
            if isinstance(model, TwoStateModel):
                return_expr = (
                    f'[sol(t)[2] / (sol(t)[1] + sol(t)[2]) for t in DATA["{ti.target_id}"].t_eval]'
                )
            else:
                return_expr = f'[sol(t)[1] for t in DATA["{ti.target_id}"].t_eval]'
            saveat_expr = f'DATA["{ti.target_id}"].t_eval'

        # Get ODE function name
        if isinstance(model, CustomODEModel):
            import re

            match = re.search(r"function\s+([\w!]+)\s*\(", model.code_julia)
            ode_func_name = match.group(1) if match else f"{func_name}_ode!"
        else:
            ode_func_name = f"{func_name}_ode!"

        return f"""
function simulate_{func_name}({param_sig})
    d = DATA["{ti.target_id}"]
    prob = ODEProblem({ode_func_name}, d.y0, d.t_span, {param_vec})
    sol = solve(prob, AutoTsit5(Rosenbrock23());
                saveat={saveat_expr},
                abstol=1e-8, reltol=1e-6, maxiters=1e6)
    return {return_expr}
end
"""

    def _generate_joint_turing_model_dict(
        self,
        targets_info: list[TargetInfo],
        parameters: dict[str, ParameterInfo],
    ) -> str:
        """Generate joint Turing model using DATA dict."""
        lines = []
        lines.append("@model function joint_calibration(data)")

        # Priors for model parameters
        lines.append("    # =========================================")
        lines.append("    # PRIORS (shared across targets)")
        lines.append("    # =========================================")
        for pname, pinfo in parameters.items():
            prior_code = self._generate_prior(pname, pinfo.prior)
            targets_str = ", ".join(pinfo.targets)
            lines.append(f"    {prior_code}  # Used by: {targets_str}")

        # Priors for uncertain sigmas (hierarchical error model)
        sigma_params = []
        for ti in targets_info:
            for i, obs in enumerate(ti.observations.observations):
                if obs.sd_uncertain:
                    func_name = self.mapper._sanitize_name(ti.target_id)
                    sigma_name = (
                        f"sigma_{func_name}"
                        if len(ti.observations.observations) == 1
                        else f"sigma_{func_name}_{i+1}"
                    )
                    # Half-normal prior centered on the estimated sigma
                    lines.append(
                        f'    {sigma_name} ~ truncated(Normal(data["{ti.target_id}"].obs_sigma[{i+1}], data["{ti.target_id}"].obs_sigma[{i+1}]), 0, Inf)'
                    )
                    sigma_params.append(sigma_name)

        # Likelihoods
        lines.append("")
        lines.append("    # =========================================")
        lines.append("    # LIKELIHOODS")
        lines.append("    # =========================================")

        for ti in targets_info:
            tid = ti.target_id
            lines.append(f"\n    # {tid}")
            model = ti.target.calibration.model
            n_obs = len(ti.observations.observations)
            func_name = self.mapper._sanitize_name(tid)

            if isinstance(model, AlgebraicModel):
                # Algebraic model - call compute function with params dict and inputs dict
                param_dict_items = ", ".join(f'"{p}" => {p}' for p in ti.parameter_names)
                lines.append(f"    params_{func_name} = Dict({param_dict_items})")
                lines.append(f'    inputs_{func_name} = data["{tid}"].inputs')
                lines.append(
                    f"    pred_{func_name} = compute_{func_name}(params_{func_name}, inputs_{func_name})"
                )
            else:
                param_call = ", ".join(ti.parameter_names)
                lines.append(f"    pred_{func_name} = simulate_{func_name}({param_call})")

            # Generate likelihood with appropriate sigma (fixed or inferred) and distribution
            for i, obs in enumerate(ti.observations.observations):
                idx = i + 1
                if obs.sd_uncertain:
                    sigma_name = f"sigma_{func_name}" if n_obs == 1 else f"sigma_{func_name}_{idx}"
                    sigma_expr = sigma_name
                else:
                    sigma_expr = f'data["{tid}"].obs_sigma[{idx}]'

                # Algebraic models return a scalar; ODE models may return arrays for multiple eval points
                if isinstance(model, AlgebraicModel) or n_obs == 1:
                    pred_expr = f"pred_{func_name}"
                else:
                    pred_expr = f"pred_{func_name}[{idx}]"

                # Use LogNormal likelihood when specified (sigma is in log-space)
                if obs.likelihood_distribution == "lognormal":
                    # LogNormal(μ, σ) where μ = log(pred), so median = pred
                    lines.append(
                        f'    data["{tid}"].obs_median[{idx}] ~ LogNormal(log({pred_expr}), {sigma_expr})'
                    )
                else:
                    # Normal likelihood (default)
                    lines.append(
                        f'    data["{tid}"].obs_median[{idx}] ~ Normal({pred_expr}, {sigma_expr})'
                    )

        # Return
        lines.append("")
        param_returns = ", ".join(f"{p}={p}" for p in parameters.keys())
        if sigma_params:
            sigma_returns = ", ".join(f"{s}={s}" for s in sigma_params)
            lines.append(f"    return ({param_returns}, {sigma_returns})")
        else:
            lines.append(f"    return ({param_returns})")
        lines.append("end")

        return "\n".join(lines)

    def _generate_joint_sampling_code_dict(
        self,
        targets_info: list[TargetInfo],
        parameters: dict[str, ParameterInfo],
    ) -> str:
        """Generate sampling code using DATA dict."""
        lines = []

        # Collect sigma parameters for targets with uncertain sd
        sigma_params = []
        for ti in targets_info:
            for i, obs in enumerate(ti.observations.observations):
                if obs.sd_uncertain:
                    func_name = self.mapper._sanitize_name(ti.target_id)
                    sigma_name = (
                        f"sigma_{func_name}"
                        if len(ti.observations.observations) == 1
                        else f"sigma_{func_name}_{i+1}"
                    )
                    sigma_params.append(sigma_name)

        lines.append(
            """
model = joint_calibration(DATA)

println("=" ^ 60)
println("JOINT BAYESIAN CALIBRATION")
println("=" ^ 60)
println("\\nTargets:")"""
        )

        for ti in targets_info:
            lines.append(f'println("  - {ti.target_id}")')

        lines.append('println("\\nParameters to estimate:")')
        for pname, pinfo in parameters.items():
            targets_str = ", ".join(pinfo.targets)
            lines.append(f'println("  {pname} ({pinfo.units}) - constrained by: {targets_str}")')

        # Combine model params and sigma params for diagnostics
        all_param_symbols = ", ".join(f":{p}" for p in parameters.keys())
        if sigma_params:
            sigma_symbols = ", ".join(f":{s}" for s in sigma_params)
            all_param_symbols = f"{all_param_symbols}, {sigma_symbols}"

        lines.append(
            f"""
println("\\nSampling with NUTS...")
chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4; progress=true)

println("\\n" * "=" ^ 60)
println("POSTERIOR SUMMARY")
println("=" ^ 60)
display(chain)

# Convergence diagnostics
println("\\nConvergence diagnostics:")
params = [{all_param_symbols}]
for p in params
    rhat_val = rhat(chain[p])[1]
    ess_val = ess(chain[p])[1]
    status = rhat_val < 1.01 ? "✓" : "⚠"
    println("  $p: R-hat=$(round(rhat_val, digits=3)) $status, ESS=$(round(ess_val, digits=0))")
end

println("\\n" * "=" ^ 60)
println("POSTERIOR MEDIANS AND 90% CI")
println("=" ^ 60)"""
        )

        for pname, pinfo in parameters.items():
            lines.append(
                f"""let samples = chain[:{pname}][:]
    med = round(median(samples), sigdigits=4)
    ci_lo = round(quantile(samples, 0.05), sigdigits=4)
    ci_hi = round(quantile(samples, 0.95), sigdigits=4)
    println("  {pname}: $med ({pinfo.units})  [90% CI: $ci_lo - $ci_hi]")
end"""
            )

        # Add plotting code with priors
        param_names_list = ", ".join(f":{p}" for p in parameters.keys())
        param_labels_list = ", ".join(f'"{p}"' for p in parameters.keys())

        # Build priors dict entries
        prior_entries = []
        for pname, pinfo in parameters.items():
            prior = pinfo.prior
            if prior:
                if prior.distribution == PriorDistribution.LOGNORMAL:
                    prior_entries.append(f"    :{pname} => LogNormal({prior.mu}, {prior.sigma})")
                elif prior.distribution == PriorDistribution.NORMAL:
                    prior_entries.append(f"    :{pname} => Normal({prior.mu}, {prior.sigma})")
                elif prior.distribution == PriorDistribution.UNIFORM:
                    prior_entries.append(f"    :{pname} => Uniform({prior.lower}, {prior.upper})")
                elif prior.distribution == PriorDistribution.HALF_NORMAL:
                    prior_entries.append(
                        f"    :{pname} => truncated(Normal(0, {prior.sigma}), 0, Inf)"
                    )
                else:
                    prior_entries.append(f"    :{pname} => LogNormal(0.0, 2.0)")
            else:
                prior_entries.append(f"    :{pname} => LogNormal(0.0, 2.0)")
        priors_dict = ",\n".join(prior_entries)

        n_params = len(parameters)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        lines.append(
            f"""
# Plot marginal posteriors with priors
println("\\nPlotting marginal posteriors...")

param_names = [{param_names_list}]
param_labels = [{param_labels_list}]

priors = Dict(
{priors_dict}
)

plots = []
for (i, pname) in enumerate(param_names)
    samples = vec(chain[pname])
    med = median(samples)
    ci_lo = quantile(samples, 0.05)
    ci_hi = quantile(samples, 0.95)

    # X range based on posterior samples
    xmin, xmax = quantile(samples, 0.001), quantile(samples, 0.999)
    xrange = range(xmin, xmax, length=200)

    # Prior density
    prior_dist = priors[pname]
    prior_pdf = pdf.(prior_dist, xrange)

    # Start with prior (gray, underneath)
    plt = plot(xrange, prior_pdf;
        fill=true, fillalpha=0.2, color=:gray, linewidth=1, linecolor=:gray,
        xlabel=param_labels[i], ylabel="",
        label="Prior", legend=false, grid=false, framestyle=:box,
        xrotation=45)

    # Overlay posterior density
    density!(plt, samples;
        fill=true, fillalpha=0.4, color=:steelblue, linewidth=2,
        label="Posterior")

    # Add median and CI lines
    vline!(plt, [med]; color=:red, linewidth=2, linestyle=:solid, label="")
    vline!(plt, [ci_lo, ci_hi]; color=:red, linewidth=1, linestyle=:dash, label="")

    push!(plots, plt)
end

p = plot(plots...; layout=({n_rows}, {n_cols}), size=(1200, 1000), margin=8mm)
savefig(p, "posterior_marginals.png")
println("Saved: posterior_marginals.png")

# ======================================================================
# SAVE RESULTS TO JSON FOR AUTOMATED PROCESSING
# ======================================================================
using JSON

results = Dict(
    "n_chains" => 4,
    "n_samples" => 1000,
    "parameters" => Dict()
)

# Store raw samples for plotting
posterior_samples = Dict{{String, Vector{{Float64}}}}()

for pname in param_names
    samples = vec(chain[pname])
    rhat_val = rhat(chain[pname])[1]
    ess_bulk = ess(chain[pname])[1]
    ess_tail = ess(chain[pname]; kind=:tail)[1]

    # Store raw samples
    posterior_samples[string(pname)] = samples

    results["parameters"][string(pname)] = Dict(
        "median" => median(samples),
        "mean" => mean(samples),
        "std" => std(samples),
        "ci_05" => quantile(samples, 0.05),
        "ci_95" => quantile(samples, 0.95),
        "ci_025" => quantile(samples, 0.025),
        "ci_975" => quantile(samples, 0.975),
        "rhat" => rhat_val,
        "ess_bulk" => ess_bulk,
        "ess_tail" => ess_tail
    )
end

open("inference_results.json", "w") do f
    JSON.print(f, results, 2)
end
println("\\nSaved: inference_results.json")

# Save raw posterior samples for plotting
open("posterior_samples.json", "w") do f
    JSON.print(f, posterior_samples, 2)
end
println("Saved: posterior_samples.json")
"""
        )

        return "\n".join(lines)

    def _generate_prior(self, param_name: str, prior) -> str:
        """Generate prior declaration."""
        if prior:
            if prior.distribution == PriorDistribution.LOGNORMAL:
                return f"{param_name} ~ LogNormal({prior.mu}, {prior.sigma})"
            elif prior.distribution == PriorDistribution.NORMAL:
                return f"{param_name} ~ Normal({prior.mu}, {prior.sigma})"
            elif prior.distribution == PriorDistribution.UNIFORM:
                return f"{param_name} ~ Uniform({prior.lower}, {prior.upper})"
            elif prior.distribution == PriorDistribution.HALF_NORMAL:
                return f"{param_name} ~ truncated(Normal(0, {prior.sigma}), 0, Inf)"
        # Default
        return f"{param_name} ~ LogNormal(0.0, 2.0)"

    def _const_prefix(self, target_id: str) -> str:
        """Generate constant name prefix from target_id."""
        if "proliferation" in target_id:
            return "PROLIF"
        elif "death" in target_id:
            return "DEATH"
        elif "activation" in target_id:
            return "ACTIV"
        elif "recruitment" in target_id and "const" in target_id:
            return "CONST"
        elif "recruitment" in target_id and "encounter" in target_id:
            return "RECRUIT"
        else:
            parts = target_id.split("_")
            return parts[1].upper()[:6] if len(parts) > 1 else "TARGET"


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """CLI entry point for translator."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single target:  python -m ... --model-structure <path> <yaml_file> [output.jl]")
        print(
            "  Joint inference: python -m ... --joint --model-structure <path> <yaml1> <yaml2> ... [--output output.jl] [--fixed-sigma]"
        )
        print(
            "  All singles:    python -m ... --single-all --model-structure <path> <yaml1> <yaml2> ... [--output output.jl]"
        )
        print("")
        print("Options:")
        print("  --model-structure <path>  Path to model_structure.json (required)")
        print("  --fixed-sigma             Treat all measurement sigmas as fixed (not sampled).")
        print("                            Reduces model dimensionality and speeds up sampling.")
        print(
            "  --single-all              Run each target as independent single-target inference in one script."
        )
        print(
            "  --reference-values <path> Path to reference_values.yaml (for targets using ReferenceRef)"
        )
        sys.exit(1)

    args = sys.argv[1:]

    # Check for model-structure flag (required)
    model_structure_path = None
    if "--model-structure" in args:
        idx = args.index("--model-structure")
        model_structure_path = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    if not model_structure_path:
        print("Error: --model-structure <path> is required")
        sys.exit(1)

    # Check for reference-values flag (optional)
    reference_values_path = None
    if "--reference-values" in args:
        idx = args.index("--reference-values")
        reference_values_path = args[idx + 1]
        args = args[:idx] + args[idx + 2 :]

    # Check for single-all mode (run each target independently in one script)
    if "--single-all" in args:
        args.remove("--single-all")

        # Check for output flag
        output_path = None
        if "--output" in args:
            idx = args.index("--output")
            output_path = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

        yaml_paths = args
        builder = JointInferenceBuilder.from_model_structure_file(
            model_structure_path, reference_values_path=reference_values_path
        )
        code = builder.build_single_all(yaml_paths)

        if output_path:
            with open(output_path, "w") as f:
                f.write(code)
            print(f"Generated combined single-target script: {output_path}")
        else:
            print(code)

        sys.exit(0)

    # Check for joint mode
    if "--joint" in args:
        args.remove("--joint")

        # Check for fixed-sigma flag
        force_fixed_sigma = False
        if "--fixed-sigma" in args:
            args.remove("--fixed-sigma")
            force_fixed_sigma = True

        # Check for output flag
        output_path = None
        if "--output" in args:
            idx = args.index("--output")
            output_path = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

        yaml_paths = args
        builder = JointInferenceBuilder.from_model_structure_file(
            model_structure_path, reference_values_path=reference_values_path
        )
        code = builder.build_from_files(yaml_paths, force_fixed_sigma=force_fixed_sigma)

        if output_path:
            with open(output_path, "w") as f:
                f.write(code)
            print(f"Generated joint inference script: {output_path}")
        else:
            print(code)
    else:
        # Single target mode
        yaml_path = args[0]
        output_path = args[1] if len(args) > 1 else None

        translator = JuliaTranslator.from_model_structure_file(
            model_structure_path, reference_values_path=reference_values_path
        )

        if output_path:
            translator.generate_to_file(yaml_path, output_path)
        else:
            print(translator.generate_script(yaml_path))


if __name__ == "__main__":
    main()
