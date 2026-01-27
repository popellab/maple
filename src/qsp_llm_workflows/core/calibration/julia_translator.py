#!/usr/bin/env python3
"""
Translator from SubmodelTarget YAML to Julia Turing.jl inference code.

This module provides tools to:
1. Load and validate YAML targets
2. Execute distribution_code to extract observations
3. Map model types to Julia ODE functions
4. Generate complete Julia scripts for Bayesian inference

Usage:
    from qsp_llm_workflows.core.calibration.julia_translator import JuliaTranslator

    translator = JuliaTranslator()
    julia_code = translator.generate_script("path/to/target.yaml")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from qsp_llm_workflows.core.calibration.submodel_target import (
    CustomModel,
    DirectConversionModel,
    ExponentialGrowthModel,
    FirstOrderDecayModel,
    InputRole,
    LogisticModel,
    MichaelisMentenModel,
    PriorDistribution,
    SaturationModel,
    SubmodelTarget,
    TwoStateModel,
)
from qsp_llm_workflows.core.unit_registry import ureg


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
    """Extract observation data by running distribution_code."""

    def extract(self, target: SubmodelTarget) -> TargetObservations:
        """
        Extract all observation data from a target.

        Runs distribution_code for each measurement to get median/CI95,
        then converts CI95 to sigma.

        Filters out observations where the corresponding input has
        role=initial_condition (these are used for IC, not likelihood).
        """
        observations = []

        for measurement in target.calibration.measurements:
            # Build inputs dict for distribution_code
            inputs_dict = self._build_inputs_dict(target, measurement.uses_inputs)

            if measurement.distribution_code:
                # Execute distribution_code
                result = self._exec_distribution_code(measurement.distribution_code, inputs_dict)
                # Support both formats:
                # New format: median, ci95_lower, ci95_upper (scalars or lists)
                # Old format: median (list), ci95 (list of [lo, hi] pairs)
                if "ci95_lower" in result:
                    # New format
                    median_val = result["median"]
                    ci95_lo = result["ci95_lower"]
                    ci95_hi = result["ci95_upper"]
                    # Normalize to lists
                    medians = [median_val] if not isinstance(median_val, list) else median_val
                    ci95_lo = [ci95_lo] if not isinstance(ci95_lo, list) else ci95_lo
                    ci95_hi = [ci95_hi] if not isinstance(ci95_hi, list) else ci95_hi
                    ci95s = list(zip(ci95_lo, ci95_hi))
                else:
                    # Old format (for backwards compatibility)
                    medians = result["median"]
                    ci95s = result["ci95"]
                    if not isinstance(medians, list):
                        medians = [medians]
                    if not isinstance(ci95s[0], (list, tuple)):
                        ci95s = [ci95s]
                units = result.get("units", measurement.units)
            else:
                # No distribution_code - use inputs directly
                # This handles simple cases where input IS the observation
                medians = [inputs_dict[measurement.uses_inputs[0]].magnitude]
                ci95s = [(medians[0] * 0.9, medians[0] * 1.1)]  # Assume 10% uncertainty
                units = measurement.units

            # Determine which observations to include based on input roles
            # Group inputs by their evaluation point index (heuristic: inputs come in pairs
            # of mean/sd for each timepoint)
            eval_points = measurement.evaluation_points or [None]
            n_points = len(eval_points) if eval_points[0] is not None else 1

            # Check if any inputs have role=initial_condition
            # If so, identify which evaluation point they correspond to
            ic_indices = set()
            target_inputs = [inp for inp in target.inputs if inp.name in measurement.uses_inputs]

            # Heuristic: inputs with role=initial_condition correspond to first evaluation point(s)
            for inp in target_inputs:
                if inp.role == InputRole.INITIAL_CONDITION:
                    # Try to determine which eval point this corresponds to
                    # If input name contains "day1" or "week4" etc, match to first point
                    if any(x in inp.name.lower() for x in ["day1", "week4", "_1_", "_1"]):
                        ic_indices.add(0)
                    elif any(x in inp.name.lower() for x in ["day2", "week8", "_2_", "_2"]):
                        ic_indices.add(1)
                    # Default: assume first point if IC role is set
                    elif n_points > 1:
                        ic_indices.add(0)

            # Convert CI95 to sigma and create ObservationData, filtering IC points
            for i, (median, ci95) in enumerate(zip(medians, ci95s)):
                if i in ic_indices:
                    continue  # Skip initial condition observations

                sigma = self._ci95_to_sigma(ci95)
                eval_time = eval_points[i] if i < len(eval_points) else eval_points[-1]
                observations.append(
                    ObservationData(
                        median=median,
                        sigma=sigma,
                        eval_time=eval_time,
                        units=units,
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
        """Build inputs dict with Pint quantities for distribution_code."""
        inputs_dict = {}
        for inp in target.inputs:
            if inp.name in input_names or not input_names:
                inputs_dict[inp.name] = inp.value * ureg(inp.units)
        return inputs_dict

    def _exec_distribution_code(self, code: str, inputs: dict) -> dict:
        """Execute distribution_code and return result dict."""
        # Create execution namespace
        namespace = {"inputs": inputs, "ureg": ureg, "np": np}

        # Execute the code to define the function
        exec(code, namespace)

        # Call derive_distribution
        if "derive_distribution" not in namespace:
            raise ValueError("distribution_code must define 'derive_distribution'")

        return namespace["derive_distribution"](inputs, ureg)

    def _ci95_to_sigma(self, ci95: list[float]) -> float:
        """Convert 95% CI to standard deviation (assuming normal)."""
        return (ci95[1] - ci95[0]) / (2 * 1.96)


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
        """Generate Julia ODE function for a target's model."""
        model = target.calibration.model
        func_name = self._sanitize_name(target.target_id)

        # Handle custom models - use code_julia directly
        if isinstance(model, CustomModel):
            return model.code_julia

        # Handle direct conversion - no ODE needed
        if isinstance(model, DirectConversionModel):
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

    def __init__(self):
        self.extractor = ObservationExtractor()
        self.mapper = JuliaODEMapper()

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

        # No simulate function for direct conversion
        if isinstance(model, DirectConversionModel):
            return None

        func_name = self.mapper._sanitize_name(target.target_id)
        prefix = self._const_prefix(target.target_id)
        param_names = obs_data.parameter_names

        # Build parameter unpacking
        if len(param_names) == 1:
            param_sig = param_names[0]
            param_vec = f"[{param_names[0]}]"
        else:
            param_sig = ", ".join(param_names)
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
        if isinstance(model, CustomModel):
            # Extract function name from code_julia (e.g., "function foo_ode!(du, u, p, t)")
            # Note: Julia function names can include ! so we use [\w!]+ pattern
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

    def _generate_turing_model(self, target: SubmodelTarget, obs_data: TargetObservations) -> str:
        """Generate Turing @model block."""
        model = target.calibration.model
        func_name = self.mapper._sanitize_name(target.target_id)

        lines = ["@model function calibrate_{}(obs, sigma)".format(func_name)]

        # Generate priors
        lines.append("    # Priors")
        for param in target.calibration.parameters:
            prior_code = self._generate_prior(param)
            lines.append(f"    {prior_code}")

        lines.append("")
        lines.append("    # Likelihood")

        # Direct conversion mode
        if isinstance(model, DirectConversionModel):
            param_name = target.calibration.parameters[0].name
            lines.append(f"    obs ~ Normal({param_name}, sigma)")
        else:
            # ODE mode
            param_names = [p.name for p in target.calibration.parameters]
            param_call = ", ".join(param_names)
            lines.append(f"    pred = simulate_{func_name}({param_call})")

            if len(obs_data.observations) == 1:
                lines.append("    obs ~ Normal(pred, sigma)")
            else:
                lines.append("    for i in eachindex(obs)")
                lines.append("        obs[i] ~ Normal(pred[i], sigma[i])")
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

        return f"""
model = calibrate_{func_name}({obs_arg}, {sigma_arg})

println("Sampling...")
chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4; progress=true)

println("\\nPosterior Summary:")
display(chain)
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

    def __init__(self):
        self.generator = JuliaCodeGenerator()

    def load_target(self, yaml_path: str) -> SubmodelTarget:
        """Load and validate a target from YAML."""
        path = Path(yaml_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return SubmodelTarget.model_validate(data)

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

    def __init__(self):
        self.extractor = ObservationExtractor()
        self.mapper = JuliaODEMapper()

    def build_from_files(self, yaml_paths: list[str]) -> str:
        """
        Build joint inference Julia script from multiple YAML files.

        Args:
            yaml_paths: List of paths to YAML target files

        Returns:
            Complete Julia script for joint Bayesian inference
        """
        # Load all targets
        targets_info = []
        for path in yaml_paths:
            target = self._load_target(path)
            obs_data = self.extractor.extract(target)
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

    def _load_target(self, yaml_path: str) -> SubmodelTarget:
        """Load and validate a target from YAML."""
        path = Path(yaml_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return SubmodelTarget.model_validate(data)

    def _collect_parameters(self, targets_info: list[TargetInfo]) -> dict[str, ParameterInfo]:
        """
        Collect unique parameters across all targets.

        Parameters with the same name are merged - first encountered prior is used.
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
                    # Use first prior encountered (or could merge/validate)

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

        # No simulate function for direct conversion
        if isinstance(model, DirectConversionModel):
            return None

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
        if isinstance(model, CustomModel):
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
            model = ti.target.calibration.model

            if isinstance(model, DirectConversionModel):
                # Direct mode - parameter IS the observable
                param_name = ti.parameter_names[0]
                lines.append(f"    {prefix}_obs ~ Normal({param_name}, {prefix}_sigma)")
            else:
                # ODE mode
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

            # Build observation arrays (convert numpy to plain Python floats)
            medians = [float(o.median) for o in obs.observations]
            sigmas = [float(o.sigma) for o in obs.observations]
            eval_times = [float(o.eval_time) for o in obs.observations if o.eval_time is not None]

            lines.append(f'    "{ti.target_id}" => (')
            lines.append(f"        obs_median = {medians},")
            lines.append(f"        obs_sigma = {sigmas},")

            if obs.t_span:
                lines.append(f"        t_span = {obs.t_span},")
            if eval_times:
                lines.append(f"        t_eval = {eval_times},")
            if obs.initial_conditions:
                lines.append(f"        y0 = {obs.initial_conditions},")

            lines.append(f"    ){comma}")

        lines.append(")\n")
        return "\n".join(lines)

    def _generate_simulate_function_dict(self, ti: TargetInfo) -> Optional[str]:
        """Generate simulate function that reads from DATA dict."""
        model = ti.target.calibration.model

        # No simulate function for direct conversion
        if isinstance(model, DirectConversionModel):
            return None

        func_name = self.mapper._sanitize_name(ti.target_id)
        param_names = ti.parameter_names
        param_sig = ", ".join(param_names)
        param_vec = f"[{param_sig}]"

        # Build return expression based on model type and observation count
        n_obs = len(ti.observations.observations)
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
        if isinstance(model, CustomModel):
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

        # Priors
        lines.append("    # =========================================")
        lines.append("    # PRIORS (shared across targets)")
        lines.append("    # =========================================")
        for pname, pinfo in parameters.items():
            prior_code = self._generate_prior(pname, pinfo.prior)
            targets_str = ", ".join(pinfo.targets)
            lines.append(f"    {prior_code}  # Used by: {targets_str}")

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

            if isinstance(model, DirectConversionModel):
                param_name = ti.parameter_names[0]
                lines.append(
                    f'    data["{tid}"].obs_median[1] ~ Normal({param_name}, data["{tid}"].obs_sigma[1])'
                )
            else:
                func_name = self.mapper._sanitize_name(tid)
                param_call = ", ".join(ti.parameter_names)
                lines.append(f"    pred_{func_name} = simulate_{func_name}({param_call})")

                if n_obs == 1:
                    lines.append(
                        f'    data["{tid}"].obs_median[1] ~ Normal(pred_{func_name}, data["{tid}"].obs_sigma[1])'
                    )
                else:
                    lines.append(f"    for i in 1:{n_obs}")
                    lines.append(
                        f'        data["{tid}"].obs_median[i] ~ Normal(pred_{func_name}[i], data["{tid}"].obs_sigma[i])'
                    )
                    lines.append("    end")

        # Return
        lines.append("")
        param_returns = ", ".join(f"{p}={p}" for p in parameters.keys())
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
        print("  Single target:  python -m ... <yaml_file> [output.jl]")
        print("  Joint inference: python -m ... --joint <yaml1> <yaml2> ... [--output output.jl]")
        sys.exit(1)

    args = sys.argv[1:]

    # Check for joint mode
    if "--joint" in args:
        args.remove("--joint")

        # Check for output flag
        output_path = None
        if "--output" in args:
            idx = args.index("--output")
            output_path = args[idx + 1]
            args = args[:idx] + args[idx + 2 :]

        yaml_paths = args
        builder = JointInferenceBuilder()
        code = builder.build_from_files(yaml_paths)

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

        translator = JuliaTranslator()

        if output_path:
            translator.generate_to_file(yaml_path, output_path)
        else:
            print(translator.generate_script(yaml_path))


if __name__ == "__main__":
    main()
