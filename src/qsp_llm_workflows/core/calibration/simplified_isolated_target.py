#!/usr/bin/env python3
"""
Simplified isolated system target models for calibration data extraction.

This module implements the simplified schema that separates:
- `inputs`: What was extracted from papers (with full provenance)
- `calibration`: How to use those inputs for inference
"""

from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# ENUMS
# =============================================================================


class InputType(str, Enum):
    """Type of input extracted from literature."""

    DIRECT_MEASUREMENT = "direct_measurement"  # Value reported directly in source
    PROXY_MEASUREMENT = "proxy_measurement"  # Requires conversion (e.g., doubling time → rate)
    EXPERIMENTAL_CONDITION = "experimental_condition"  # Protocol choice from source
    INFERRED_ESTIMATE = "inferred_estimate"  # Value interpreted from qualitative text in source
    ASSUMED_VALUE = "assumed_value"  # Value assumed from domain knowledge, not in source


class InputRole(str, Enum):
    """Role of input in calibration - clarifies how each input is used."""

    INITIAL_CONDITION = "initial_condition"  # Used as IC for ODE integration
    TARGET = "target"  # Used as calibration target (likelihood term)
    FIXED_PARAMETER = "fixed_parameter"  # Fixed value in model (not estimated)
    AUXILIARY = "auxiliary"  # Contextual/supporting data, not directly used in inference


class ExtractionMethod(str, Enum):
    """Method used to extract the value."""

    MANUAL = "manual"
    WEBPLOTDIGITIZER = "webplotdigitizer"
    DIGITIZER = "digitizer"
    OTHER = "other"


class ObservableType(str, Enum):
    """Type of observable transformation."""

    IDENTITY = "identity"  # Return first state variable directly
    CUSTOM = "custom"  # User-provided code for any other transformation


class CurveType(str, Enum):
    """Curve type for direct_fit models."""

    HILL = "hill"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


# =============================================================================
# UNCERTAINTY
# =============================================================================


class Uncertainty(BaseModel):
    """Uncertainty specification - either CI95 or SD."""

    ci95: Optional[List[float]] = Field(
        default=None,
        description="95% confidence interval [lower, upper]",
    )
    sd: Optional[float] = Field(
        default=None,
        description="Standard deviation",
    )

    @model_validator(mode="after")
    def validate_ci95_ordering(self) -> "Uncertainty":
        """Validate that ci95[0] < ci95[1]."""
        if self.ci95 is not None:
            if len(self.ci95) != 2:
                raise ValueError(f"ci95 must have exactly 2 values, got {len(self.ci95)}")
            if self.ci95[0] >= self.ci95[1]:
                raise ValueError(f"ci95[0] must be < ci95[1], got {self.ci95}")
        return self


# =============================================================================
# INPUTS
# =============================================================================


class Input(BaseModel):
    """
    A value extracted from literature with full provenance.

    Inputs are referenced by name from calibration.measurements and
    calibration.state_variables.initial_condition.
    """

    name: str = Field(description="Unique identifier for this input (used in references)")
    value: float = Field(description="Extracted numeric value")
    units: str = Field(description="Units of the value")
    uncertainty: Optional[Uncertainty] = Field(
        default=None,
        description="Uncertainty in the measurement",
    )
    n: Optional[int] = Field(
        default=None,
        description="Sample size",
    )
    input_type: InputType = Field(
        description="Type of input: direct_measurement, proxy_measurement, or experimental_condition"
    )
    role: Optional[InputRole] = Field(
        default=None,
        description="Role in calibration: initial_condition, target, fixed_parameter, or auxiliary. "
        "Clarifies how this input is used in inference.",
    )
    source_ref: str = Field(
        description="Reference to source_tag in primary_data_source or secondary_data_sources"
    )
    source_location: str = Field(
        description="Location within the source (e.g., 'Figure 3B, day 28')"
    )
    extraction_method: Optional[ExtractionMethod] = Field(
        default=None,
        description="Method used to extract the value",
    )
    value_snippet: Optional[str] = Field(
        default=None,
        description="Exact text from paper containing the value (for validation)",
    )


# =============================================================================
# CALIBRATION - PRIORS
# =============================================================================


class PriorDistribution(str, Enum):
    """Supported prior distribution types for Bayesian inference."""

    LOGNORMAL = "lognormal"  # For positive parameters (rates, densities)
    NORMAL = "normal"  # For unconstrained parameters
    UNIFORM = "uniform"  # For bounded parameters
    HALF_NORMAL = "half_normal"  # For positive parameters with mode at 0


class Prior(BaseModel):
    """
    Prior distribution specification for Bayesian inference.

    For lognormal: parameter ~ LogNormal(mu, sigma) where mu = log(median)
    For normal: parameter ~ Normal(mu, sigma)
    For uniform: parameter ~ Uniform(lower, upper)
    For half_normal: parameter ~ HalfNormal(sigma)
    """

    distribution: PriorDistribution = Field(description="Prior distribution type")
    mu: Optional[float] = Field(
        default=None,
        description="Location parameter (log-scale for lognormal, mean for normal)",
    )
    sigma: Optional[float] = Field(
        default=None,
        description="Scale parameter (log-scale SD for lognormal, SD for normal/half_normal)",
    )
    lower: Optional[float] = Field(
        default=None,
        description="Lower bound (for uniform)",
    )
    upper: Optional[float] = Field(
        default=None,
        description="Upper bound (for uniform)",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Justification for prior choice",
    )

    @model_validator(mode="after")
    def validate_prior_params(self) -> "Prior":
        """Validate that required parameters are provided for each distribution type."""
        dist = self.distribution
        errors = []

        if dist == PriorDistribution.LOGNORMAL:
            if self.mu is None:
                errors.append("lognormal prior requires mu (log-scale location)")
            if self.sigma is None:
                errors.append("lognormal prior requires sigma (log-scale SD)")
        elif dist == PriorDistribution.NORMAL:
            if self.mu is None:
                errors.append("normal prior requires mu (mean)")
            if self.sigma is None:
                errors.append("normal prior requires sigma (SD)")
        elif dist == PriorDistribution.UNIFORM:
            if self.lower is None:
                errors.append("uniform prior requires lower bound")
            if self.upper is None:
                errors.append("uniform prior requires upper bound")
            if self.lower is not None and self.upper is not None:
                if self.lower >= self.upper:
                    errors.append(f"uniform lower ({self.lower}) must be < upper ({self.upper})")
        elif dist == PriorDistribution.HALF_NORMAL:
            if self.sigma is None:
                errors.append("half_normal prior requires sigma")

        if errors:
            raise ValueError(f"Prior validation errors: {'; '.join(errors)}")

        return self


# =============================================================================
# CALIBRATION - PARAMETERS
# =============================================================================


class Parameter(BaseModel):
    """A parameter to be estimated during inference."""

    name: str = Field(description="Parameter name from the full QSP model")
    units: str = Field(description="Parameter units")
    prior: Optional[Prior] = Field(
        default=None,
        description="Prior distribution for Bayesian inference. If not specified, "
        "translator will use heuristics based on parameter units and expected range.",
    )


# =============================================================================
# CALIBRATION - STATE VARIABLES
# =============================================================================


class FixedInitialCondition(BaseModel):
    """Initial condition with a fixed/normalized value."""

    value: float = Field(description="Fixed initial value")
    rationale: str = Field(description="Why this value was chosen")


class InputRefInitialCondition(BaseModel):
    """Initial condition referencing a measured input."""

    input_ref: str = Field(description="Name of input to use as initial condition")
    rationale: str = Field(description="Why this input is used as IC")


class StateVariable(BaseModel):
    """A state variable in the ODE system."""

    name: str = Field(description="State variable name")
    units: str = Field(description="State variable units")
    initial_condition: Union[FixedInitialCondition, InputRefInitialCondition] = Field(
        description="Initial condition: either fixed value or reference to input"
    )


# =============================================================================
# CALIBRATION - MODEL
# =============================================================================


class InputRef(BaseModel):
    """Reference to an input for fixed parameter values."""

    input_ref: str = Field(description="Name of input to use as fixed value")


# ParameterRole can be either a string (parameter name to estimate) or InputRef (fixed from data)
ParameterRole = Union[str, InputRef]


# =============================================================================
# CALIBRATION - MODEL TYPES (each with specific required parameters)
# =============================================================================


class BaseModelSpec(BaseModel):
    """Base class for all model specifications."""

    data_rationale: str = Field(
        description="Why this model type fits the experimental data (assay design, conditions, readout)"
    )
    submodel_rationale: str = Field(
        description="Why this is a valid submodel of the full QSP model (which reactions, what assumptions)"
    )


class FirstOrderDecayModel(BaseModelSpec):
    """First-order decay: dy/dt = -k * y"""

    type: Literal["first_order_decay"] = "first_order_decay"
    rate_constant: ParameterRole = Field(description="Rate constant parameter name or input_ref")


class ExponentialGrowthModel(BaseModelSpec):
    """Exponential growth: dy/dt = k * y"""

    type: Literal["exponential_growth"] = "exponential_growth"
    rate_constant: ParameterRole = Field(description="Rate constant parameter name or input_ref")


class LogisticModel(BaseModelSpec):
    """Logistic growth: dy/dt = k * y * (1 - y/K)"""

    type: Literal["logistic"] = "logistic"
    rate_constant: ParameterRole = Field(
        description="Growth rate constant parameter name or input_ref"
    )
    carrying_capacity: ParameterRole = Field(
        description="Carrying capacity parameter name or input_ref"
    )


class MichaelisMentenModel(BaseModelSpec):
    """Michaelis-Menten kinetics: dy/dt = -Vmax * y / (Km + y)"""

    type: Literal["michaelis_menten"] = "michaelis_menten"
    vmax: ParameterRole = Field(description="Maximum rate parameter name or input_ref")
    km: ParameterRole = Field(description="Michaelis constant parameter name or input_ref")


class TwoStateModel(BaseModelSpec):
    """Two-state transition: A → B with first-order kinetics.

    State variables: [A, B] where dA/dt = -k*A, dB/dt = +k*A
    Useful for activation, differentiation, or state transition dynamics.
    """

    type: Literal["two_state"] = "two_state"
    forward_rate: ParameterRole = Field(
        description="Forward transition rate constant (A → B) parameter name or input_ref"
    )


class SaturationModel(BaseModelSpec):
    """First-order approach to saturation: dy/dt = k * (1 - y)

    State variable y approaches 1 asymptotically from below.
    Useful for recruitment, filling, or saturation dynamics where y is a
    dimensionless fraction (0 to 1) of some carrying capacity.
    """

    type: Literal["saturation"] = "saturation"
    rate_constant: ParameterRole = Field(
        description="Approach rate constant parameter name or input_ref"
    )


class DirectConversionModel(BaseModelSpec):
    """Direct analytical conversion (no ODE): e.g., k = ln(2) / t_half"""

    type: Literal["direct_conversion"] = "direct_conversion"
    formula: str = Field(description="Analytical formula (e.g., 'k = ln(2) / doubling_time')")


class DirectFitModel(BaseModelSpec):
    """Direct curve fitting (no ODE): e.g., Hill equation for IC50"""

    type: Literal["direct_fit"] = "direct_fit"
    curve: CurveType = Field(description="Curve type to fit (hill, linear, exponential)")


class CustomModel(BaseModelSpec):
    """Custom ODE with user-provided code"""

    type: Literal["custom"] = "custom"
    code: str = Field(
        description="Python ODE function. Signature: def ode(t, y, params, inputs) -> dict"
    )
    code_julia: str = Field(
        description="Julia ODE function for inference. "
        "Signature: function ode!(du, u, p, t) where du is modified in-place.",
    )


# Discriminated union of all model types
Model = Annotated[
    Union[
        FirstOrderDecayModel,
        ExponentialGrowthModel,
        LogisticModel,
        MichaelisMentenModel,
        TwoStateModel,
        SaturationModel,
        DirectConversionModel,
        DirectFitModel,
        CustomModel,
    ],
    Field(discriminator="type"),
]


# =============================================================================
# CALIBRATION - INDEPENDENT VARIABLE
# =============================================================================


class IndependentVariable(BaseModel):
    """The independent variable (what was varied in the experiment)."""

    name: str = Field(description="Variable name: time, dose, concentration, ET_ratio, or custom")
    units: str = Field(description="Units of the independent variable")
    span: Optional[List[float]] = Field(
        default=None,
        description="Range [start, end] for ODE integration or curve evaluation",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Why this span was chosen",
    )


# =============================================================================
# CALIBRATION - OBSERVABLE
# =============================================================================


class Observable(BaseModel):
    """
    Transform ODE state variable(s) into the measured quantity.

    For type-based observables, the transformation is implicit.
    For custom observables, code provides the transformation function.
    """

    type: ObservableType = Field(description="Observable type")
    state_variables: List[str] = Field(description="State variable(s) used in the observable")
    rationale: Optional[str] = Field(
        default=None,
        description="Why this observable type was chosen",
    )

    # For custom
    code: Optional[str] = Field(
        default=None,
        description="Python function for custom observables. Signature: def compute(t, y, y_start) -> float",
    )


# =============================================================================
# CALIBRATION - LIKELIHOOD
# =============================================================================


class Likelihood(BaseModel):
    """Likelihood specification for fitting."""

    distribution: str = Field(description="Distribution type (normal, lognormal, beta, etc.)")
    rationale: Optional[str] = Field(
        default=None,
        description="Why this distribution was chosen",
    )


# =============================================================================
# CALIBRATION - MEASUREMENT
# =============================================================================


class Measurement(BaseModel):
    """
    A measurement to be used for calibration.

    References inputs by name and specifies evaluation points for comparison.
    evaluation_points units are inherited from independent_variable.units.
    """

    name: str = Field(description="Measurement name")
    observable: Optional[Observable] = Field(
        default=None,
        description="How to compute the observable from state variables",
    )
    units: str = Field(description="Units of the measurement")
    uses_inputs: List[str] = Field(description="Names of inputs that feed this measurement")
    evaluation_points: List[float] = Field(
        description="Points at which to evaluate the model (units from independent_variable)"
    )
    sample_size: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="Sample size (single int or list matching evaluation_points)",
    )
    sample_size_rationale: Optional[str] = Field(
        default=None,
        description="Rationale for sample size, especially if assumed or uncertain",
    )
    distribution_code: Optional[str] = Field(
        default=None,
        description="Python code for uncertainty propagation. Signature: def derive_distribution(inputs, ureg) -> dict",
    )
    likelihood: Likelihood = Field(description="Likelihood specification")


# =============================================================================
# CALIBRATION (TOP-LEVEL)
# =============================================================================


class Calibration(BaseModel):
    """
    Everything needed for inference code generation.

    Contains parameters, state variables, model, independent variable, and measurements.
    """

    parameters: List[Parameter] = Field(description="Parameters to estimate during inference")
    state_variables: Optional[List[StateVariable]] = Field(
        default=None,
        description="State variables for ODE-based models",
    )
    model: Model = Field(description="Mathematical model specification")
    independent_variable: Optional[IndependentVariable] = Field(
        default=None,
        description="Independent variable (required for ODE models)",
    )
    measurements: List[Measurement] = Field(
        description="Measurements for calibration",
    )
    identifiability_notes: str = Field(
        description="Discussion of parameter identifiability: which parameters are constrained, "
        "which are correlated, what additional data would be needed"
    )


# =============================================================================
# DATA SOURCES
# =============================================================================


class PrimaryDataSource(BaseModel):
    """Primary literature data source. DOI is required."""

    doi: str = Field(min_length=1, description="DOI of the source (required)")
    title: Optional[str] = Field(default=None, description="Title of the source")
    authors: Optional[List[str]] = Field(default=None, description="Author list")
    year: Optional[int] = Field(default=None, description="Publication year")
    source_tag: str = Field(description="Short identifier for referencing (e.g., 'Smith2023')")


class SecondaryDataSource(BaseModel):
    """Secondary literature data source. Requires either DOI or URL."""

    doi: Optional[str] = Field(default=None, description="DOI of the source")
    url: Optional[str] = Field(default=None, description="URL if no DOI available")
    title: Optional[str] = Field(default=None, description="Title of the source")
    authors: Optional[List[str]] = Field(default=None, description="Author list")
    year: Optional[int] = Field(default=None, description="Publication year")
    source_tag: str = Field(description="Short identifier for referencing (e.g., 'Smith2023')")
    contribution: Optional[str] = Field(
        default=None,
        description="What this source contributed",
    )

    @model_validator(mode="after")
    def validate_doi_or_url(self) -> "SecondaryDataSource":
        """Ensure at least one of doi or url is provided."""
        if not self.doi and not self.url:
            raise ValueError(f"Secondary source '{self.source_tag}' must have either doi or url")
        return self


# =============================================================================
# EXPERIMENTAL CONTEXT
# =============================================================================


class CellLine(BaseModel):
    """Cell line information."""

    name: str = Field(description="Cell line name")
    species: Optional[str] = Field(default=None, description="Species of origin")
    tissue_origin: Optional[str] = Field(default=None, description="Tissue of origin")
    cell_type: Optional[str] = Field(default=None, description="Cell type")


class CellType(BaseModel):
    """Cell type information (for primary cells)."""

    name: str = Field(description="Cell type name")
    phenotype: Optional[str] = Field(default=None, description="Cell phenotype")
    isolation_method: Optional[str] = Field(default=None, description="How cells were isolated")


class CultureConditions(BaseModel):
    """Cell culture conditions."""

    substrate: Optional[str] = Field(default=None, description="Culture substrate")
    serum_concentration: Optional[float] = Field(default=None, description="Serum concentration")
    serum_concentration_units: Optional[str] = Field(
        default=None, description="Units for serum concentration"
    )
    oxygen_level: Optional[str] = Field(
        default=None, description="Oxygen level (normoxic, hypoxic, etc.)"
    )
    culture_type: Optional[str] = Field(
        default=None, description="Culture type (2d_monolayer, 3d_spheroid, etc.)"
    )


class ExperimentalContext(BaseModel):
    """Experimental context for the calibration target."""

    species: str = Field(description="Species (human, mouse, rat, etc.)")
    system: str = Field(
        description="Experimental system (in_vitro_primary_cells, in_vitro_immortalized, etc.)"
    )
    cell_lines: Optional[List[CellLine]] = Field(default=None, description="Cell lines used")
    cell_types: Optional[List[CellType]] = Field(
        default=None, description="Cell types used (for primary cells)"
    )
    culture_conditions: Optional[CultureConditions] = Field(
        default=None, description="Culture conditions"
    )
    indication: Optional[str] = Field(default=None, description="Disease indication (PDAC, etc.)")


# =============================================================================
# SIMPLIFIED ISOLATED TARGET (TOP-LEVEL)
# =============================================================================


class SimplifiedIsolatedTarget(BaseModel):
    """
    Simplified isolated system target for calibration data extraction.

    Separates:
    - `inputs`: What was extracted from papers (with full provenance)
    - `calibration`: How to use those inputs for inference
    """

    target_id: str = Field(description="Unique identifier for this target")

    # Extracted data with provenance
    inputs: List[Input] = Field(
        description="Values extracted from literature with full provenance",
    )

    # Calibration specification
    calibration: Calibration = Field(description="Everything needed for inference code generation")

    # Experimental context
    experimental_context: ExperimentalContext = Field(
        description="Experimental context for the target"
    )

    # Study interpretation
    study_interpretation: str = Field(
        description="Scientific interpretation of how the study data informs the model"
    )
    key_assumptions: List[str] = Field(
        description="Key assumptions made in using this data",
    )
    key_study_limitations: Optional[List[str]] = Field(
        default=None,
        description="Known limitations of the study",
    )

    # Data sources
    primary_data_source: PrimaryDataSource = Field(description="Primary literature source")
    secondary_data_sources: Optional[List[SecondaryDataSource]] = Field(
        default=None,
        description="Additional literature sources",
    )

    # -------------------------------------------------------------------------
    # VALIDATORS
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_input_refs(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that all input references point to existing inputs.

        Checks:
        - measurement.uses_inputs references
        - state_variable.initial_condition.input_ref references
        - model parameter_role InputRef references
        """
        input_names = {inp.name for inp in self.inputs}
        errors = []

        # Check uses_inputs in measurements
        for measurement in self.calibration.measurements:
            for input_name in measurement.uses_inputs:
                if input_name not in input_names:
                    errors.append(
                        f"Measurement '{measurement.name}' references unknown input '{input_name}'"
                    )

        # Check input_ref in initial_conditions
        if self.calibration.state_variables:
            for sv in self.calibration.state_variables:
                if isinstance(sv.initial_condition, InputRefInitialCondition):
                    if sv.initial_condition.input_ref not in input_names:
                        errors.append(
                            f"State variable '{sv.name}' initial_condition references "
                            f"unknown input '{sv.initial_condition.input_ref}'"
                        )

        # Check InputRef in model parameter_roles
        model = self.calibration.model
        for field_name in ["rate_constant", "carrying_capacity", "vmax", "km", "forward_rate"]:
            if hasattr(model, field_name):
                value = getattr(model, field_name)
                if isinstance(value, InputRef) and value.input_ref not in input_names:
                    errors.append(
                        f"Model {field_name} references unknown input '{value.input_ref}'"
                    )

        if errors:
            raise ValueError(
                "Invalid input references:\n  - "
                + "\n  - ".join(errors)
                + f"\nAvailable inputs: {sorted(input_names)}"
            )

        return self

    @model_validator(mode="after")
    def validate_source_refs(self) -> "SimplifiedIsolatedTarget":
        """Validate that all source_refs point to existing source tags."""
        source_tags = {self.primary_data_source.source_tag}
        if self.secondary_data_sources:
            source_tags.update(s.source_tag for s in self.secondary_data_sources)

        errors = []
        for inp in self.inputs:
            if inp.source_ref not in source_tags:
                errors.append(f"Input '{inp.name}' has source_ref '{inp.source_ref}'")

        if errors:
            raise ValueError(
                "Invalid source references:\n  - "
                + "\n  - ".join(errors)
                + f"\nAvailable source tags: {sorted(source_tags)}"
            )

        return self

    @model_validator(mode="after")
    def validate_parameter_roles(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that parameter role strings reference existing parameters.

        When a parameter_role is a string (not InputRef), it should match
        a parameter name in calibration.parameters.
        """
        param_names = {p.name for p in self.calibration.parameters}
        errors = []

        model = self.calibration.model
        for field_name in ["rate_constant", "carrying_capacity", "vmax", "km", "forward_rate"]:
            if hasattr(model, field_name):
                value = getattr(model, field_name)
                if isinstance(value, str) and value not in param_names:
                    errors.append(f"Model {field_name}='{value}' is not in calibration.parameters")

        if errors:
            raise ValueError(
                "Invalid parameter references:\n  - "
                + "\n  - ".join(errors)
                + f"\nAvailable parameters: {sorted(param_names)}"
            )

        return self

    @model_validator(mode="after")
    def validate_observable_state_vars(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that observable state_variables reference existing state variables.
        """
        if not self.calibration.state_variables:
            return self  # No state variables to check against

        sv_names = {sv.name for sv in self.calibration.state_variables}
        errors = []

        for measurement in self.calibration.measurements:
            if measurement.observable:
                for sv_name in measurement.observable.state_variables:
                    if sv_name not in sv_names:
                        errors.append(
                            f"Measurement '{measurement.name}' observable references "
                            f"unknown state variable '{sv_name}'"
                        )

        if errors:
            raise ValueError(
                "Invalid state variable references:\n  - "
                + "\n  - ".join(errors)
                + f"\nAvailable state variables: {sorted(sv_names)}"
            )

        return self

    @model_validator(mode="after")
    def validate_ode_model_requirements(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that ODE-based models have required state_variables and span.
        """
        ode_model_types = {
            "first_order_decay",
            "exponential_growth",
            "logistic",
            "michaelis_menten",
            "two_state",
            "saturation",
            "custom",
        }

        model_type = self.calibration.model.type
        if model_type not in ode_model_types:
            return self  # Non-ODE models don't need state_variables

        errors = []

        if not self.calibration.state_variables:
            errors.append(f"Model type '{model_type}' requires state_variables")
        elif model_type == "two_state" and len(self.calibration.state_variables) != 2:
            errors.append(
                f"Model type 'two_state' requires exactly 2 state_variables, "
                f"got {len(self.calibration.state_variables)}"
            )

        if not self.calibration.independent_variable:
            errors.append(f"Model type '{model_type}' requires independent_variable")
        elif not self.calibration.independent_variable.span:
            errors.append(f"Model type '{model_type}' requires independent_variable.span")

        if errors:
            raise ValueError("Missing required fields for ODE model:\n  - " + "\n  - ".join(errors))

        return self

    @model_validator(mode="after")
    def validate_custom_code_syntax(self) -> "SimplifiedIsolatedTarget":
        """
        Validate syntax and function signature for custom code blocks.

        Checks:
        - CustomModel.code has 'def ode(t, y, params)'
        - Custom observable.code has 'def compute(t, y, y_start)'
        - distribution_code has 'def derive_distribution(inputs, ureg)'
        """
        import ast

        errors = []

        # Check CustomModel.code
        model = self.calibration.model
        if hasattr(model, "code") and model.code:
            try:
                tree = ast.parse(model.code)
                # Find function definition
                func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                if not func_defs:
                    errors.append("CustomModel.code must define a function 'ode'")
                elif func_defs[0].name != "ode":
                    errors.append(
                        f"CustomModel.code function must be named 'ode', "
                        f"got '{func_defs[0].name}'"
                    )
            except SyntaxError as e:
                errors.append(f"CustomModel.code syntax error: {e}")

        # Check custom observable code
        for measurement in self.calibration.measurements:
            if measurement.observable and measurement.observable.code:
                try:
                    tree = ast.parse(measurement.observable.code)
                    func_defs = [
                        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    ]
                    if not func_defs:
                        errors.append(
                            f"Measurement '{measurement.name}' observable.code "
                            f"must define a function 'compute'"
                        )
                    elif func_defs[0].name != "compute":
                        errors.append(
                            f"Measurement '{measurement.name}' observable.code function "
                            f"must be named 'compute', got '{func_defs[0].name}'"
                        )
                except SyntaxError as e:
                    errors.append(
                        f"Measurement '{measurement.name}' observable.code syntax error: {e}"
                    )

            # Check distribution_code
            if measurement.distribution_code:
                try:
                    tree = ast.parse(measurement.distribution_code)
                    func_defs = [
                        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    ]
                    if not func_defs:
                        errors.append(
                            f"Measurement '{measurement.name}' distribution_code "
                            f"must define a function 'derive_distribution'"
                        )
                    elif func_defs[0].name != "derive_distribution":
                        errors.append(
                            f"Measurement '{measurement.name}' distribution_code function "
                            f"must be named 'derive_distribution', got '{func_defs[0].name}'"
                        )
                except SyntaxError as e:
                    errors.append(
                        f"Measurement '{measurement.name}' distribution_code syntax error: {e}"
                    )

        if errors:
            raise ValueError("Code validation errors:\n  - " + "\n  - ".join(errors))

        return self

    @model_validator(mode="after")
    def validate_span_ordering(self) -> "SimplifiedIsolatedTarget":
        """Validate that span[0] < span[1] and both are non-negative."""
        iv = self.calibration.independent_variable
        if iv is None or iv.span is None:
            return self

        t_start, t_end = iv.span
        if t_start < 0:
            raise ValueError(f"independent_variable.span[0] must be non-negative, got {t_start}")
        if t_end <= t_start:
            raise ValueError(f"independent_variable.span[1] must be > span[0], got {iv.span}")

        return self

    @model_validator(mode="after")
    def validate_input_values_in_snippets(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that extracted values appear in their value_snippet.

        Catches hallucinations where the LLM extracts a value that doesn't
        appear in the cited text. Skips inputs without snippets or with
        input_type='experimental_condition' (protocol choices may not have
        explicit numeric values in text).
        """
        from qsp_llm_workflows.core.calibration.validators import check_value_in_text

        errors = []
        for inp in self.inputs:
            # Skip if no snippet provided
            if not inp.value_snippet:
                continue

            # Skip types that may not have explicit numeric values in text
            if inp.input_type in (
                InputType.EXPERIMENTAL_CONDITION,
                InputType.INFERRED_ESTIMATE,
                InputType.ASSUMED_VALUE,
            ):
                continue

            if not check_value_in_text(inp.value_snippet, inp.value):
                errors.append(
                    f"Input '{inp.name}': value {inp.value} not found in snippet "
                    f"'{inp.value_snippet[:80]}{'...' if len(inp.value_snippet) > 80 else ''}'"
                )

        if errors:
            raise ValueError(
                "Value-snippet mismatches (possible hallucination):\n  - "
                + "\n  - ".join(errors)
                + "\n\nCheck that extracted values match the source text."
            )

        return self

    @model_validator(mode="after")
    def validate_doi_resolution_and_metadata(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that DOIs resolve and metadata matches.

        Checks:
        1. DOI resolves via CrossRef
        2. Title matches (fuzzy match)
        3. Year matches (if provided)
        4. First author matches source_tag pattern (if provided)
        """
        from qsp_llm_workflows.core.calibration.validators import (
            resolve_doi,
            fuzzy_match,
        )

        errors = []

        def check_doi_source(
            doi: str, source_tag: str, title: Optional[str], year: Optional[int], prefix: str
        ):
            """Validate a DOI and check metadata matches."""
            metadata = resolve_doi(doi)
            if metadata is None:
                errors.append(
                    f"{prefix} DOI '{doi}' failed to resolve. " f"Verify at https://doi.org/{doi}"
                )
                return

            # Check title match (fuzzy)
            if title and metadata.get("title"):
                if not fuzzy_match(title, metadata["title"], threshold=0.7):
                    errors.append(
                        f"{prefix} title mismatch:\n"
                        f"    Recorded: '{title[:60]}...'\n"
                        f"    CrossRef: '{metadata['title'][:60]}...'"
                    )

            # Check year match
            if year and metadata.get("year"):
                if year != metadata["year"]:
                    errors.append(
                        f"{prefix} year mismatch: "
                        f"recorded {year}, CrossRef says {metadata['year']}"
                    )

            # Check first author appears in source_tag
            if metadata.get("first_author") and source_tag:
                first_author = metadata["first_author"].lower()
                tag_lower = source_tag.lower()
                if first_author not in tag_lower:
                    errors.append(
                        f"{prefix} source_tag '{source_tag}' doesn't contain "
                        f"first author '{metadata['first_author']}' from CrossRef"
                    )

        # Check primary source (DOI required)
        check_doi_source(
            self.primary_data_source.doi,
            self.primary_data_source.source_tag,
            self.primary_data_source.title,
            self.primary_data_source.year,
            "Primary source",
        )

        # Check secondary sources (only validate if they have DOI, URL-only sources skip validation)
        if self.secondary_data_sources:
            for i, source in enumerate(self.secondary_data_sources):
                if source.doi:
                    check_doi_source(
                        source.doi,
                        source.source_tag,
                        source.title,
                        source.year,
                        f"Secondary source [{i}]",
                    )
                # URL-only sources don't get DOI validation

        if errors:
            raise ValueError("DOI/metadata validation errors:\n  - " + "\n  - ".join(errors))

        return self

    @model_validator(mode="after")
    def validate_units_are_valid_pint(self) -> "SimplifiedIsolatedTarget":
        """
        Validate that all unit strings are valid Pint units.

        Checks units in:
        - inputs[].units
        - calibration.parameters[].units
        - calibration.state_variables[].units
        - calibration.independent_variable.units
        - calibration.measurements[].units
        """
        from qsp_llm_workflows.core.unit_registry import ureg

        errors = []

        def check_unit(unit_str: str, location: str):
            try:
                ureg(unit_str)
            except Exception as e:
                errors.append(f"{location}: '{unit_str}' is not a valid Pint unit ({e})")

        # Check input units
        for inp in self.inputs:
            check_unit(inp.units, f"Input '{inp.name}'")

        # Check parameter units
        for param in self.calibration.parameters:
            check_unit(param.units, f"Parameter '{param.name}'")

        # Check state variable units
        if self.calibration.state_variables:
            for sv in self.calibration.state_variables:
                check_unit(sv.units, f"State variable '{sv.name}'")

        # Check independent variable units
        if self.calibration.independent_variable:
            check_unit(
                self.calibration.independent_variable.units,
                "Independent variable",
            )

        # Check measurement units
        for m in self.calibration.measurements:
            check_unit(m.units, f"Measurement '{m.name}'")

        if errors:
            raise ValueError("Invalid Pint units:\n  - " + "\n  - ".join(errors))

        return self


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "InputType",
    "InputRole",
    "ExtractionMethod",
    "ObservableType",
    "CurveType",
    "PriorDistribution",
    # Input models
    "Uncertainty",
    "Input",
    # Prior models
    "Prior",
    # Calibration models
    "Parameter",
    "FixedInitialCondition",
    "InputRefInitialCondition",
    "StateVariable",
    "InputRef",
    "ParameterRole",
    # Model types
    "BaseModelSpec",
    "FirstOrderDecayModel",
    "ExponentialGrowthModel",
    "LogisticModel",
    "MichaelisMentenModel",
    "TwoStateModel",
    "SaturationModel",
    "DirectConversionModel",
    "DirectFitModel",
    "CustomModel",
    "Model",
    # Other calibration models
    "IndependentVariable",
    "Observable",
    "Likelihood",
    "Measurement",
    "Calibration",
    # Context models
    "PrimaryDataSource",
    "SecondaryDataSource",
    "CellLine",
    "CellType",
    "CultureConditions",
    "ExperimentalContext",
    # Top-level
    "SimplifiedIsolatedTarget",
]


# =============================================================================
# VALIDATOR IDEAS (TO IMPLEMENT LATER)
# =============================================================================
#
# 1. validate_input_refs
#    - Check that all uses_inputs reference existing input names
#    - Check that initial_condition.input_ref references existing input names
#    - Check that parameter_roles with InputRef reference existing input names
#
# 2. validate_source_refs
#    - Check that all input.source_ref match primary_data_source.source_tag
#      or one of secondary_data_sources[].source_tag
#
# 3. validate_evaluation_points_match_inputs
#    - For direct measurements (no distribution_code), len(evaluation_points)
#      should equal len(uses_inputs)
#
# 4. validate_state_variables_for_ode_models
#    - ODE model types (first_order_decay, exponential_growth, logistic,
#      michaelis_menten, custom) require state_variables
#    - ODE model types require independent_variable with span
#
# 5. validate_model_fields
#    - direct_conversion requires formula
#    - direct_fit requires curve
#    - custom requires code
#    - ODE types require parameter_roles
#
# 6. validate_observable_fields
#    - time_to_threshold requires threshold and direction
#    - custom requires code
#
# 7. validate_uncertainty
#    - Uncertainty should have at least one of ci95 or sd
#
# 8. validate_snippet_contains_value
#    - If value_snippet is provided, check that value appears in it
#      (with tolerance for formatting differences)
#
# 9. validate_units_consistency (advanced)
#    - Check that parameter units are consistent with model type
#    - Check that state variable units are consistent with observable
#
# 10. validate_custom_code_syntax
#     - Parse custom ODE/observable code to check for syntax errors
#     - Verify function signature matches expected pattern
