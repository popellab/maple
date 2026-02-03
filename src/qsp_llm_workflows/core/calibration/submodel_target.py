#!/usr/bin/env python3
"""
Submodel-based calibration target models for QSP parameter inference.

This module implements the SubmodelTarget schema that separates:
- `inputs`: What was extracted from papers (with full provenance)
- `calibration`: How to use those inputs for inference
"""

import warnings
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, model_validator

# Import relevance enums from central location
from qsp_llm_workflows.core.calibration.enums import (
    IndicationMatch,
    PerturbationType,
    SourceQuality,
    TMECompatibility,
)
from qsp_llm_workflows.core.calibration.code_validator import find_accessed_params


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


class SourceAccess(str, Enum):
    """Accessibility of the source for automated snippet validation."""

    OPEN_ACCESS = "open_access"  # Full text available, can auto-validate
    RESTRICTED = "restricted"  # Can't access full text, requires manual verification


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
    source_access: Optional[SourceAccess] = Field(
        default=None,
        description="Accessibility of source for auto-validation. "
        "Set to 'restricted' to skip automated snippet validation.",
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
    prior: Prior = Field(description="Prior distribution for Bayesian inference")


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
# CALIBRATION - FORWARD MODEL TYPES (each with specific required parameters)
# =============================================================================


class BaseForwardModelSpec(BaseModel):
    """Base class for all forward model specifications."""

    data_rationale: str = Field(
        description="Why this model type fits the experimental data (assay design, conditions, readout)"
    )
    submodel_rationale: str = Field(
        description="Why this is a valid submodel of the full QSP model (which reactions, what assumptions)"
    )
    # Fields moved from Calibration into forward_model
    independent_variable: Optional["IndependentVariable"] = Field(
        default=None,
        description="Independent variable (required for ODE models)",
    )
    state_variables: Optional[List["StateVariable"]] = Field(
        default=None,
        description="State variables for ODE-based models",
    )


class FirstOrderDecayModel(BaseForwardModelSpec):
    """First-order decay: dy/dt = -k * y"""

    type: Literal["first_order_decay"] = "first_order_decay"
    rate_constant: ParameterRole = Field(description="Rate constant parameter name or input_ref")


class ExponentialGrowthModel(BaseForwardModelSpec):
    """Exponential growth: dy/dt = k * y"""

    type: Literal["exponential_growth"] = "exponential_growth"
    rate_constant: ParameterRole = Field(description="Rate constant parameter name or input_ref")


class LogisticModel(BaseForwardModelSpec):
    """Logistic growth: dy/dt = k * y * (1 - y/K)"""

    type: Literal["logistic"] = "logistic"
    rate_constant: ParameterRole = Field(
        description="Growth rate constant parameter name or input_ref"
    )
    carrying_capacity: ParameterRole = Field(
        description="Carrying capacity parameter name or input_ref"
    )


class MichaelisMentenModel(BaseForwardModelSpec):
    """Michaelis-Menten kinetics: dy/dt = -Vmax * y / (Km + y)"""

    type: Literal["michaelis_menten"] = "michaelis_menten"
    vmax: ParameterRole = Field(description="Maximum rate parameter name or input_ref")
    km: ParameterRole = Field(description="Michaelis constant parameter name or input_ref")


class TwoStateModel(BaseForwardModelSpec):
    """Two-state transition: A → B with first-order kinetics.

    State variables: [A, B] where dA/dt = -k*A, dB/dt = +k*A
    Useful for activation, differentiation, or state transition dynamics.
    """

    type: Literal["two_state"] = "two_state"
    forward_rate: ParameterRole = Field(
        description="Forward transition rate constant (A → B) parameter name or input_ref"
    )


class SaturationModel(BaseForwardModelSpec):
    """First-order approach to saturation: dy/dt = k * (1 - y)

    State variable y approaches 1 asymptotically from below.
    Useful for recruitment, filling, or saturation dynamics where y is a
    dimensionless fraction (0 to 1) of some carrying capacity.
    """

    type: Literal["saturation"] = "saturation"
    rate_constant: ParameterRole = Field(
        description="Approach rate constant parameter name or input_ref"
    )


class AlgebraicModel(BaseForwardModelSpec):
    """Algebraic relationship between parameters and observable (no ODE).

    The `code` field implements the FORWARD model: given parameter values,
    predict the observable. This is consistent with ODE models where the
    model predicts trajectories from parameters.

    Example for k = ln(2) / t_half:
    - Parameter to infer: k (rate constant)
    - Observable measured: t_half (half-life)
    - Forward model: t_half = ln(2) / k

    The inference engine finds parameters where the predicted observable
    matches the measured data (within measurement error).

    Use for:
    - Single-parameter conversions: t_half = ln(2) / k
    - Multi-parameter relationships: steady_state = k_prod / k_deg
    - Any non-ODE formula connecting parameters to observables
    """

    type: Literal["algebraic"] = "algebraic"
    formula: str = Field(
        description="Descriptive formula showing the relationship "
        "(e.g., 't_half = ln(2) / k' or 'C_ss = k_prod / k_deg')"
    )
    code: str = Field(
        description="Python FORWARD model: given params, predict observable. "
        "Signature: def compute(params: dict, inputs: dict, ureg) -> Quantity. "
        "Example: return np.log(2) / params['k'] for predicting t_half from k."
    )
    code_julia: str = Field(
        description="Julia FORWARD model for inference. "
        "Signature: function compute(params::Dict, inputs::Dict) -> value"
    )


class DirectFitModel(BaseForwardModelSpec):
    """Direct curve fitting (no ODE): e.g., Hill equation for IC50"""

    type: Literal["direct_fit"] = "direct_fit"
    curve: CurveType = Field(description="Curve type to fit (hill, linear, exponential)")


class CustomODEModel(BaseForwardModelSpec):
    """Custom ODE with user-provided code."""

    type: Literal["custom_ode"] = "custom_ode"
    code: str = Field(
        description="Python ODE function. Signature: def ode(t, y, params, inputs) -> dict"
    )
    code_julia: str = Field(
        description="Julia ODE function for inference. "
        "Signature: function ode!(du, u, p, t) where du is modified in-place."
    )


# Discriminated union of all forward model types
ForwardModel = Annotated[
    Union[
        FirstOrderDecayModel,
        ExponentialGrowthModel,
        LogisticModel,
        MichaelisMentenModel,
        TwoStateModel,
        SaturationModel,
        AlgebraicModel,
        DirectFitModel,
        CustomODEModel,
    ],
    Field(discriminator="type"),
]

# Backwards compatibility alias
Model = ForwardModel


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
# CALIBRATION - ERROR MODEL
# =============================================================================


class ErrorModel(BaseModel):
    """
    An error model entry specifying how to compare model predictions to data.

    The error model describes:
    - Which inputs from the data are used
    - How to compute measurement error/uncertainty
    - What likelihood to use for inference

    For ODE models, evaluation_points specifies when to compare.
    For algebraic models, evaluation_points is not needed.
    """

    name: str = Field(description="Error model entry name")
    observable: Optional[Observable] = Field(
        default=None,
        description="How to compute the observable from state variables (for ODE models)",
    )
    units: str = Field(description="Units of the measurement")
    uses_inputs: List[str] = Field(description="Names of inputs that feed this measurement")
    evaluation_points: Optional[List[float]] = Field(
        default=None,
        description="Points at which to evaluate the model (only for ODE models; "
        "units from forward_model.independent_variable)",
    )
    sample_size: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="Sample size (single int or list matching evaluation_points)",
    )
    sample_size_rationale: Optional[str] = Field(
        default=None,
        description="Rationale for sample size, especially if assumed or uncertain",
    )
    observation_code: str = Field(
        description="Python code to derive the observation (point estimate + uncertainty) from inputs. "
        "Signature: def derive_observation(inputs, sample_size, ureg) -> dict. "
        "Required return keys: "
        "'value' (Pint Quantity with units matching error_model.units), "
        "'sd' (measurement uncertainty - dimensionless for lognormal, units for normal). "
        "Optional return keys: "
        "'sd_uncertain' (bool - if True, inference adds prior on SD), "
        "'n' (int - sample size for reference). "
        "Use to compile literature data (ranges, CIs, multiple values) into a point estimate "
        "and characterize how uncertain that observation is."
    )
    likelihood: Likelihood = Field(description="Likelihood specification")


# Backwards compatibility alias
Measurement = ErrorModel


# =============================================================================
# CALIBRATION (TOP-LEVEL)
# =============================================================================


class Calibration(BaseModel):
    """
    Everything needed for inference code generation.

    Contains:
    - parameters: What we're inferring
    - forward_model: Physics/math that maps parameters to predictions
      (includes state_variables and independent_variable for ODE models)
    - error_model: Statistics that maps predictions + data to likelihood
    - identifiability_notes: What can/can't be learned from this data
    """

    parameters: List[Parameter] = Field(description="Parameters to estimate during inference")
    forward_model: ForwardModel = Field(
        description="Forward model specification (physics/math: params → predictions). "
        "Includes state_variables and independent_variable for ODE models."
    )
    error_model: List[ErrorModel] = Field(
        description="Error model entries specifying how to compare predictions to data "
        "(statistics: predicted observable → likelihood)",
    )
    identifiability_notes: str = Field(
        description="Discussion of parameter identifiability: which parameters are constrained, "
        "which are correlated, what additional data would be needed"
    )

    # Backwards compatibility properties
    @property
    def model(self) -> ForwardModel:
        """Backwards compatibility: access forward_model as model."""
        return self.forward_model

    @property
    def measurements(self) -> List[ErrorModel]:
        """Backwards compatibility: access error_model as measurements."""
        return self.error_model

    @property
    def state_variables(self) -> Optional[List[StateVariable]]:
        """Backwards compatibility: access forward_model.state_variables."""
        return self.forward_model.state_variables

    @property
    def independent_variable(self) -> Optional[IndependentVariable]:
        """Backwards compatibility: access forward_model.independent_variable."""
        return self.forward_model.independent_variable


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
    source_quality: Optional[SourceQuality] = Field(
        default=None,
        description=(
            "Quality tier of this secondary source. Important when secondary sources "
            "provide quantitative values (e.g., half-lives, cell densities). "
            "Flag non_peer_reviewed sources like Wikipedia."
        ),
    )

    @model_validator(mode="after")
    def validate_doi_or_url(self) -> "SecondaryDataSource":
        """Ensure at least one of doi or url is provided."""
        if not self.doi and not self.url:
            raise ValueError(f"Secondary source '{self.source_tag}' must have either doi or url")
        return self

    @model_validator(mode="after")
    def warn_non_peer_reviewed_secondary(self) -> "SecondaryDataSource":
        """Warn if secondary source is non-peer-reviewed."""
        if self.source_quality == SourceQuality.NON_PEER_REVIEWED:
            warnings.warn(
                f"Secondary source '{self.source_tag}' is non-peer-reviewed.\n"
                f"If this source provides quantitative values used in calibration, "
                f"consider finding a peer-reviewed primary source instead.",
                UserWarning,
            )
        # Also warn if URL suggests non-peer-reviewed but source_quality not set
        if self.url and not self.source_quality:
            non_peer_domains = ["wikipedia.org", "reddit.com", "quora.com", "stackexchange.com"]
            if any(domain in self.url.lower() for domain in non_peer_domains):
                warnings.warn(
                    f"Secondary source '{self.source_tag}' URL suggests non-peer-reviewed source "
                    f"({self.url}), but source_quality is not set.\n"
                    f"Set source_quality to 'non_peer_reviewed' and document rationale.",
                    UserWarning,
                )
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
# SOURCE RELEVANCE ASSESSMENT
# =============================================================================


class SourceRelevanceAssessment(BaseModel):
    """
    Structured assessment of source-to-target relevance for calibration targets.

    This model captures how well the source data translates to the target model,
    including indication match, source quality, perturbation context, and TME
    compatibility. Validators use this information to flag potential issues.
    """

    # Indication relevance
    indication_match: IndicationMatch = Field(
        description=(
            "How well does the source indication match the target indication?\n"
            "- exact: Same disease (e.g., PDAC data for PDAC model)\n"
            "- related: Same organ or disease class (e.g., other pancreatic diseases)\n"
            "- proxy: Different tissue used as mechanistic proxy (e.g., melanoma for PDAC)\n"
            "- unrelated: No clear biological connection"
        )
    )
    indication_match_justification: str = Field(
        min_length=50,
        description=(
            "Justify the indication match rating. If PROXY or UNRELATED, explain "
            "why this source is acceptable and what translation uncertainty is expected."
        ),
    )

    # Species
    species_source: str = Field(description="Species in the source study (human, mouse, rat, etc.)")
    species_target: str = Field(
        default="human", description="Target species for the model (usually 'human')"
    )

    # Source quality
    source_quality: SourceQuality = Field(
        description=(
            "Quality tier of the primary data source.\n"
            "IMPORTANT: 'non_peer_reviewed' includes Wikipedia, preprints, and "
            "unreviewed sources. Avoid if possible; if used, document rationale."
        )
    )

    # Perturbation context
    perturbation_type: PerturbationType = Field(
        description=(
            "Type of experimental perturbation in the source study.\n"
            "If 'pharmacological' or 'genetic_perturbation', explain in "
            "perturbation_relevance how this relates to physiological parameter values."
        )
    )
    perturbation_relevance: Optional[str] = Field(
        default=None,
        description=(
            "For pharmacological/genetic perturbations: explain relevance to the "
            "physiological parameter being estimated. E.g., if using drug-induced "
            "death rates, explain whether this represents an upper bound, typical "
            "value, or requires scaling."
        ),
    )

    # TME compatibility (for immune/stromal parameters)
    tme_compatibility: Optional[TMECompatibility] = Field(
        default=None,
        description=(
            "For immune/stromal parameters: TME compatibility assessment.\n"
            "- high: Source TME similar to target (e.g., both desmoplastic)\n"
            "- moderate: Some TME differences that may affect values\n"
            "- low: Major differences (e.g., T cell-permissive model for T cell-excluded tumor)"
        ),
    )
    tme_compatibility_notes: Optional[str] = Field(
        default=None,
        description=(
            "Notes on TME differences and their expected impact on the parameter. "
            "E.g., 'EG7 thymoma is highly T cell-permissive; PDAC is T cell-excluded. "
            "Expect 10-100x overestimation of infiltration rates.'"
        ),
    )

    # Translation uncertainty
    estimated_translation_uncertainty_fold: float = Field(
        ge=1.0,
        le=1000.0,
        description=(
            "Estimated fold-uncertainty due to source-to-target translation.\n"
            "1.0 = no additional uncertainty (exact match)\n"
            "3.0 = typical for cross-species or related indication\n"
            "10.0 = typical for proxy indication or low TME compatibility\n"
            "30-100 = cross-indication with major biological differences"
        ),
    )

    # Computed confidence score (optional, can be set by validators)
    validation_warnings: Optional[List[str]] = Field(
        default=None,
        description="Validation warnings generated by automated checks (populated by validators)",
    )


# =============================================================================
# SUBMODEL TARGET (TOP-LEVEL)
# =============================================================================


class SubmodelTarget(BaseModel):
    """
    Submodel-based calibration target for QSP parameter inference.

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

    # Source relevance assessment (REQUIRED)
    source_relevance: SourceRelevanceAssessment = Field(
        description=(
            "Structured assessment of how well the source data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
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
    def validate_input_refs(self) -> "SubmodelTarget":
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
            from qsp_llm_workflows.core.calibration.exceptions import InputReferenceError

            errors.append(f"Available inputs: {sorted(input_names)}")
            raise InputReferenceError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_source_refs(self) -> "SubmodelTarget":
        """Validate that all source_refs point to existing source tags."""
        source_tags = {self.primary_data_source.source_tag}
        if self.secondary_data_sources:
            source_tags.update(s.source_tag for s in self.secondary_data_sources)

        errors = []
        for inp in self.inputs:
            if inp.source_ref not in source_tags:
                errors.append(f"Input '{inp.name}' has source_ref '{inp.source_ref}'")

        if errors:
            from qsp_llm_workflows.core.calibration.exceptions import SourceRefError

            errors.append(f"Available source tags: {sorted(source_tags)}")
            raise SourceRefError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_parameter_roles(self) -> "SubmodelTarget":
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
    def validate_observable_state_vars(self) -> "SubmodelTarget":
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
    def validate_observable_type_code_consistency(self) -> "SubmodelTarget":
        """
        Validate that observable type and code presence are consistent.

        - type="custom" REQUIRES code to be present
        - type="identity" FORBIDS code (uses first state variable directly)
        """
        errors = []

        for entry in self.calibration.error_model:
            if not entry.observable:
                continue

            obs = entry.observable

            if obs.type == ObservableType.CUSTOM:
                if not obs.code:
                    errors.append(
                        f"Error model '{entry.name}': observable type is 'custom' but "
                        f"no code is provided. Custom observables require a compute function."
                    )
            elif obs.type == ObservableType.IDENTITY:
                if obs.code:
                    errors.append(
                        f"Error model '{entry.name}': observable type is 'identity' but "
                        f"code is provided. Identity observables use the first state variable "
                        f"directly - remove the code or change type to 'custom'."
                    )

        if errors:
            raise ValueError(
                "Observable type/code consistency errors:\n  - " + "\n  - ".join(errors)
            )

        return self

    @model_validator(mode="after")
    def validate_algebraic_model_no_observable(self) -> "SubmodelTarget":
        """
        Warn if algebraic models have observable defined in error_model.

        For algebraic models, the forward_model.code directly computes the predicted
        observable, so having a separate observable.code is redundant and confusing.
        The observable field is designed for ODE models where state variables need
        to be transformed into measured quantities.
        """
        model = self.calibration.model
        if model.type != "algebraic":
            return self

        for entry in self.calibration.error_model:
            if entry.observable and entry.observable.code:
                warnings.warn(
                    f"Error model '{entry.name}' has observable.code but forward_model "
                    f"is 'algebraic'.\n\n"
                    f"For algebraic models, forward_model.code directly computes the "
                    f"predicted observable. The observable.code field is for ODE models "
                    f"where state variables need transformation.\n\n"
                    f"Consider:\n"
                    f"  1. Remove observable.code (use forward_model.code output directly)\n"
                    f"  2. If transformation is needed, incorporate it into forward_model.code\n"
                    f"  3. Set observable to None or use type='identity'",
                    UserWarning,
                )

        return self

    @model_validator(mode="after")
    def validate_observable_code_signature(self) -> "SubmodelTarget":
        """
        Validate that custom observable code has the correct function signature.

        For ODE models with custom observables:
        - Expected: def compute(t, y, y_start) -> float

        The signature allows computing derived quantities from:
        - t: time point (for time-dependent observables)
        - y: current state vector (state values at time t)
        - y_start: initial state vector (for fold-change calculations)
        """
        import ast

        errors = []

        for entry in self.calibration.error_model:
            if not entry.observable or not entry.observable.code:
                continue

            obs = entry.observable

            try:
                tree = ast.parse(obs.code)
                func_defs = [
                    node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                ]

                if not func_defs:
                    # No function found - handled by validate_custom_code_syntax
                    continue

                func = func_defs[0]
                if func.name != "compute":
                    # Wrong name - handled by validate_custom_code_syntax
                    continue

                # Check argument count
                n_args = len(func.args.args)

                # For ODE models, expect (t, y, y_start) = 3 args
                # For algebraic models, we've already warned, but if they have code,
                # it might use (inputs) = 1 arg pattern
                model_type = self.calibration.model.type

                if model_type in {
                    "first_order_decay",
                    "exponential_growth",
                    "logistic",
                    "michaelis_menten",
                    "two_state",
                    "saturation",
                    "custom_ode",
                }:
                    if n_args != 3:
                        errors.append(
                            f"Error model '{entry.name}': observable.code compute() has "
                            f"{n_args} argument(s), expected 3 (t, y, y_start) for ODE models.\n"
                            f"Signature should be: def compute(t, y, y_start) -> float"
                        )
                # For algebraic models, we've already issued a warning about having
                # observable.code at all, so don't add noise about signature

            except SyntaxError:
                # Syntax errors handled by validate_custom_code_syntax
                pass

        if errors:
            raise ValueError(
                "Observable code signature errors:\n  - " + "\n  - ".join(errors)
            )

        return self

    @model_validator(mode="after")
    def validate_observation_sd_units_for_likelihood(self) -> "SubmodelTarget":
        """
        Validate that observation_code SD units match the likelihood type.

        - For normal/truncated_normal likelihoods: SD must have same units as measurement
        - For lognormal/beta likelihoods: SD should be dimensionless (log-scale or [0,1])

        This prevents unit mismatches that cause inference failures.
        """
        from qsp_llm_workflows.core.unit_registry import ureg
        import numpy as np

        # Build inputs dict with units
        inputs_dict = {}
        for inp in self.inputs:
            try:
                inputs_dict[inp.name] = inp.value * ureg(inp.units)
            except Exception:
                inputs_dict[inp.name] = inp.value

        errors = []

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            # Determine what SD units should be based on likelihood
            likelihood_dist = entry.likelihood.distribution.lower()

            # Likelihoods that operate on log-scale or normalized scale
            dimensionless_likelihoods = {"lognormal", "beta", "dirichlet"}

            # Likelihoods that operate in measurement units
            measurement_scale_likelihoods = {
                "normal",
                "truncated_normal",
                "student_t",
                "half_normal",
                "exponential",
            }

            try:
                # Execute observation_code
                local_scope = {"np": np, "numpy": np, "ureg": ureg}
                exec(entry.observation_code, local_scope)
                derive_observation = local_scope.get("derive_observation")

                if derive_observation is None:
                    continue

                result = derive_observation(inputs_dict, entry.sample_size, ureg)
                if not isinstance(result, dict) or "sd" not in result:
                    continue

                sd = result["sd"]
                sd_has_units = hasattr(sd, "dimensionality")
                sd_is_dimensionless = (
                    not sd_has_units
                    or (sd_has_units and sd.dimensionless)
                )

                expected_units = ureg(entry.units)
                measurement_is_dimensionless = expected_units.dimensionless

                if likelihood_dist in dimensionless_likelihoods:
                    # SD should be dimensionless for log/normalized likelihoods
                    if sd_has_units and not sd.dimensionless:
                        errors.append(
                            f"Error model '{entry.name}': likelihood is '{likelihood_dist}' "
                            f"(operates on log/normalized scale) but derive_observation() returns "
                            f"SD with units '{sd.units}'.\n"
                            f"For {likelihood_dist} likelihoods, SD should be dimensionless.\n"
                            f"Fix: return SD without units or as ureg.dimensionless"
                        )

                elif likelihood_dist in measurement_scale_likelihoods:
                    # SD should match measurement units
                    if not measurement_is_dimensionless:
                        if sd_is_dimensionless:
                            errors.append(
                                f"Error model '{entry.name}': likelihood is '{likelihood_dist}' "
                                f"but derive_observation() returns dimensionless SD.\n"
                                f"Measurement has units '{entry.units}', so SD must also have "
                                f"units.\n"
                                f"Fix: return SD with units, e.g., "
                                f"`'sd': value * ureg('{entry.units}')`"
                            )
                        elif sd_has_units:
                            if sd.dimensionality != expected_units.dimensionality:
                                errors.append(
                                    f"Error model '{entry.name}': SD units ({sd.units}) don't "
                                    f"match measurement units ({entry.units}).\n"
                                    f"For {likelihood_dist} likelihood, SD must be in the same "
                                    f"units as the measurement."
                                )

            except Exception:
                # Execution errors handled by other validators
                pass

        if errors:
            raise ValueError(
                "Measurement error SD units / likelihood type mismatch:\n  - "
                + "\n  - ".join(errors)
            )

        return self

    @model_validator(mode="after")
    def validate_evaluation_points_within_span(self) -> "SubmodelTarget":
        """
        Validate that evaluation_points fall within independent_variable.span.

        For ODE models, evaluation_points specify when to compare model to data.
        These should be within the integration span, otherwise the ODE solver
        won't have values at those points.
        """
        iv = self.calibration.independent_variable
        if iv is None or iv.span is None:
            return self  # Non-ODE models don't have span

        t_start, t_end = iv.span
        errors = []

        for entry in self.calibration.error_model:
            if entry.evaluation_points is None:
                continue

            for i, t in enumerate(entry.evaluation_points):
                if t < t_start or t > t_end:
                    errors.append(
                        f"Error model '{entry.name}': evaluation_point[{i}] = {t} is outside "
                        f"independent_variable.span [{t_start}, {t_end}]"
                    )

        if errors:
            raise ValueError(
                "Evaluation points outside integration span:\n  - " + "\n  - ".join(errors)
                + f"\n\nAdjust evaluation_points to be within [{t_start}, {t_end}] "
                f"or extend independent_variable.span."
            )

        return self

    @model_validator(mode="after")
    def validate_ode_model_requirements(self) -> "SubmodelTarget":
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

        # Check that ODE models have evaluation_points in error_model
        for entry in self.calibration.error_model:
            if entry.evaluation_points is None:
                errors.append(
                    f"Error model '{entry.name}' requires evaluation_points for ODE model type '{model_type}'"
                )

        if errors:
            raise ValueError("Missing required fields for ODE model:\n  - " + "\n  - ".join(errors))

        return self

    @model_validator(mode="after")
    def validate_custom_code_syntax(self) -> "SubmodelTarget":
        """
        Validate syntax and function signature for custom code blocks.

        Checks:
        - CustomODEModel.code has 'def ode(t, y, params, inputs)'
        - AlgebraicModel.code has 'def compute(params, inputs, ureg)'
        - Custom observable.code has 'def compute(t, y, y_start)'
        - observation_code has 'def derive_observation(inputs, sample_size, ureg)'
        """
        import ast

        errors = []

        # Check model.code based on model type
        model = self.calibration.model
        if hasattr(model, "code") and model.code:
            try:
                tree = ast.parse(model.code)
                # Find function definition
                func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

                # Determine expected function name based on model type
                if model.type == "custom_ode":
                    expected_func = "ode"
                    model_name = "CustomODEModel"
                elif model.type == "algebraic":
                    expected_func = "compute"
                    model_name = "AlgebraicModel"
                else:
                    # Other model types with code - skip validation
                    expected_func = None
                    model_name = None

                if expected_func is not None:
                    if not func_defs:
                        errors.append(f"{model_name}.code must define a function '{expected_func}'")
                    elif func_defs[0].name != expected_func:
                        errors.append(
                            f"{model_name}.code function must be named '{expected_func}', "
                            f"got '{func_defs[0].name}'"
                        )
            except SyntaxError as e:
                errors.append(f"Model code syntax error: {e}")

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

            # Check observation_code
            if measurement.observation_code:
                try:
                    tree = ast.parse(measurement.observation_code)
                    func_defs = [
                        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    ]
                    if not func_defs:
                        errors.append(
                            f"Error model '{measurement.name}' observation_code "
                            f"must define a function 'derive_observation'"
                        )
                    elif func_defs[0].name != "derive_observation":
                        errors.append(
                            f"Error model '{measurement.name}' observation_code function "
                            f"must be named 'derive_observation', got '{func_defs[0].name}'"
                        )
                    else:
                        # Check argument count: should be (inputs, sample_size, ureg)
                        n_args = len(func_defs[0].args.args)
                        if n_args != 3:
                            errors.append(
                                f"Error model '{measurement.name}' observation_code function "
                                f"'derive_observation' must have 3 arguments (inputs, sample_size, ureg), "
                                f"got {n_args}"
                            )
                except SyntaxError as e:
                    errors.append(
                        f"Error model '{measurement.name}' observation_code syntax error: {e}"
                    )

        if errors:
            raise ValueError("Code validation errors:\n  - " + "\n  - ".join(errors))

        return self

    @model_validator(mode="after")
    def validate_observation_code_execution(self) -> "SubmodelTarget":
        """
        Execute observation_code and validate return value structure.

        The observation_code compiles literature data into:
        - A point estimate ('value') for comparison with forward model prediction
        - Measurement uncertainty ('sd') for the likelihood

        Expected return structure:
        - 'value': Pint Quantity (required) - the observed point estimate
        - 'sd': float or Quantity (required) - measurement uncertainty
        - 'sd_uncertain': bool (optional) - if True, inference adds prior on SD
        - 'n': int (optional) - sample size for reference

        See: https://mc-stan.org/docs/stan-users-guide/measurement-error.html
        """
        from qsp_llm_workflows.core.unit_registry import ureg
        import numpy as np

        errors = []

        # Build inputs dict from self.inputs
        inputs_dict = {}
        for inp in self.inputs:
            inputs_dict[inp.name] = inp.value * ureg(inp.units)

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            try:
                # Compile and execute
                local_scope = {"np": np, "numpy": np, "ureg": ureg}
                exec(entry.observation_code, local_scope)
                derive_observation = local_scope.get("derive_observation")

                if derive_observation is None:
                    errors.append(
                        f"Error model '{entry.name}': observation_code "
                        f"did not define 'derive_observation' function"
                    )
                    continue

                # Execute with inputs and sample_size
                result = derive_observation(inputs_dict, entry.sample_size, ureg)

                # Validate return structure is dict
                if not isinstance(result, dict):
                    errors.append(
                        f"Error model '{entry.name}': derive_observation must return dict, "
                        f"got {type(result).__name__}"
                    )
                    continue

                # Check required 'value' key
                if "value" not in result:
                    errors.append(
                        f"Error model '{entry.name}': derive_observation must return dict "
                        f"with 'value' key, got keys: {list(result.keys())}"
                    )
                else:
                    # Validate 'value' is a Pint Quantity
                    value = result["value"]
                    if not hasattr(value, "magnitude") or not hasattr(value, "units"):
                        errors.append(
                            f"Error model '{entry.name}': 'value' must be a Pint Quantity, "
                            f"got {type(value).__name__}. Example: return {{'value': x * ureg('pg/mL'), ...}}"
                        )
                    else:
                        # Validate units match error_model.units
                        expected_units = ureg(entry.units)
                        if value.dimensionality != expected_units.dimensionality:
                            errors.append(
                                f"Error model '{entry.name}': 'value' has units '{value.units}' "
                                f"but error_model.units is '{entry.units}'. Dimensionality mismatch."
                            )

                # Check required 'sd' key
                if "sd" not in result:
                    errors.append(
                        f"Error model '{entry.name}': derive_observation must return dict "
                        f"with 'sd' key, got keys: {list(result.keys())}"
                    )
                else:
                    # Validate sd is positive
                    sd = result["sd"]
                    sd_value = sd.magnitude if hasattr(sd, "magnitude") else sd
                    if not isinstance(sd_value, (int, float)):
                        errors.append(
                            f"Error model '{entry.name}': 'sd' must be numeric, "
                            f"got {type(sd_value).__name__}"
                        )
                    elif sd_value <= 0:
                        errors.append(
                            f"Error model '{entry.name}': 'sd' must be positive, got {sd_value}"
                        )

                # Validate optional 'sd_uncertain' is bool if present
                if "sd_uncertain" in result:
                    if not isinstance(result["sd_uncertain"], bool):
                        errors.append(
                            f"Error model '{entry.name}': 'sd_uncertain' must be bool, "
                            f"got {type(result['sd_uncertain']).__name__}"
                        )

                # Validate optional 'n' is positive int if present
                if "n" in result:
                    n = result["n"]
                    if not isinstance(n, int) or n <= 0:
                        errors.append(
                            f"Error model '{entry.name}': 'n' must be positive int, "
                            f"got {n}"
                        )

            except Exception as e:
                errors.append(
                    f"Error model '{entry.name}': observation_code execution error: {e}"
                )

        if errors:
            raise ValueError(
                "observation_code validation errors:\n  - " + "\n  - ".join(errors)
            )

        return self

    @model_validator(mode="after")
    def warn_multi_param_algebraic_identifiability(self) -> "SubmodelTarget":
        """
        Warn when AlgebraicModel has more parameters than measurement points.

        For algebraic models, N parameters generally require N independent
        constraints to be identifiable. This validator warns when the number
        of parameters exceeds the number of measurement points, suggesting
        potential identifiability issues.
        """
        model = self.calibration.model
        if model.type != "algebraic":
            return self

        n_params = len(self.calibration.parameters)
        # For algebraic models, count 1 per error_model entry (no evaluation_points needed)
        n_measurements = sum(
            len(m.evaluation_points) if m.evaluation_points else 1
            for m in self.calibration.error_model
        )

        if n_params > n_measurements:
            warnings.warn(
                f"AlgebraicModel has {n_params} parameters but only {n_measurements} "
                f"measurement point(s). This may indicate identifiability issues.\n"
                f"Consider:\n"
                f"  - Using an ODE model with time-course data to separate parameters\n"
                f"  - Adding additional measurement points/conditions\n"
                f"  - Documenting which parameter combinations are identifiable in "
                f"identifiability_notes",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def validate_all_parameters_used_in_forward_model(self) -> "SubmodelTarget":
        """
        Validate that all parameters in calibration.parameters are used in the forward model.

        If a parameter is defined in calibration.parameters but not referenced in the
        forward model code, it cannot be identified from the data. This is an error
        because it indicates either:
        1. A copy-paste error where parameters were added but not used
        2. A misunderstanding of the forward model structure
        3. Multi-parameter targets where some parameters are just "along for the ride"

        For multi-parameter calibration, ALL parameters must influence the model output
        to be identifiable. Parameters that don't appear in the forward model will have
        posteriors equal to their priors, which defeats the purpose of calibration.

        Applies to model types with custom code: algebraic, custom_ode.
        Uses AST parsing to find actual params['name'] or params.get('name') accesses.
        """
        model = self.calibration.model

        # Only check model types that have custom code
        if model.type not in ("algebraic", "custom_ode"):
            return self

        if not hasattr(model, "code") or not model.code:
            return self

        param_names = set(p.name for p in self.calibration.parameters)

        # Use AST parsing to find actual parameter accesses in the code
        # This catches params['k_test'], params["k_test"], params.get('k_test')
        accessed_params = find_accessed_params(model.code, dict_name="params")

        unused_params = param_names - accessed_params

        if unused_params:
            raise ValueError(
                f"Parameter(s) not accessed in forward model code: {sorted(unused_params)}\n"
                f"Defined parameters: {sorted(param_names)}\n"
                f"Accessed parameters: {sorted(accessed_params)}\n\n"
                f"All parameters in calibration.parameters MUST be accessed via "
                f"params['name'] or params.get('name') in the forward model code "
                f"to be identifiable from data. Parameters that don't influence "
                f"the model output cannot be calibrated.\n\n"
                f"Fix by either:\n"
                f"1. Updating the forward model to use these parameters\n"
                f"2. Removing unused parameters from calibration.parameters"
            )

        return self

    @model_validator(mode="after")
    def warn_observation_sd_unreasonable(self) -> "SubmodelTarget":
        """
        Warn when observation SD is unreasonably large or small.

        Compares the SD from observation_code to the magnitude of the
        observed value. Warns if SD is >100x or <0.001x the observed value,
        which often indicates a units mismatch or typo.
        """
        from qsp_llm_workflows.core.unit_registry import ureg
        import numpy as np

        # Build inputs dict
        inputs_dict = {}
        for inp in self.inputs:
            try:
                inputs_dict[inp.name] = inp.value * ureg(inp.units)
            except Exception:
                inputs_dict[inp.name] = inp.value

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            # Skip this check for lognormal/beta likelihoods where SD is dimensionless
            # and comparing to dimensional observed values doesn't make sense
            likelihood_dist = entry.likelihood.distribution.lower()
            if likelihood_dist in {"lognormal", "beta", "dirichlet"}:
                continue

            try:
                # Execute observation_code
                local_scope = {"np": np, "numpy": np, "ureg": ureg}
                exec(entry.observation_code, local_scope)
                derive_observation = local_scope.get("derive_observation")
                if derive_observation is None:
                    continue

                result = derive_observation(inputs_dict, entry.sample_size, ureg)
                if not isinstance(result, dict) or "sd" not in result or "value" not in result:
                    continue

                sd = result["sd"]
                sd_value = sd.magnitude if hasattr(sd, "magnitude") else sd

                observed = result["value"]
                observed_value = observed.magnitude if hasattr(observed, "magnitude") else observed

                if observed_value == 0:
                    continue

                ratio = sd_value / abs(observed_value)

                if ratio > 100:
                    warnings.warn(
                        f"Error model '{entry.name}': SD ({sd_value:.2e}) is {ratio:.0f}x "
                        f"larger than observed value ({observed_value:.2e}).\n"
                        f"This may indicate a units mismatch. Check observation_code.",
                        UserWarning,
                    )
                elif ratio < 0.001:
                    warnings.warn(
                        f"Error model '{entry.name}': SD ({sd_value:.2e}) is {1/ratio:.0f}x "
                        f"smaller than observed value ({observed_value:.2e}).\n"
                        f"This may indicate a units mismatch or overly confident error estimate.",
                        UserWarning,
                    )

            except Exception:
                # Don't warn on execution errors - other validators handle that
                pass

        return self

    @model_validator(mode="after")
    def validate_span_ordering(self) -> "SubmodelTarget":
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
    def validate_input_values_in_snippets(self) -> "SubmodelTarget":
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
            from qsp_llm_workflows.core.calibration.exceptions import SnippetValueMismatchError

            raise SnippetValueMismatchError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_doi_resolution_and_metadata(self) -> "SubmodelTarget":
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
                from qsp_llm_workflows.core.calibration.exceptions import DOIResolutionError

                raise DOIResolutionError(doi, source_type=prefix)
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
            from qsp_llm_workflows.core.calibration.exceptions import DOIMetadataMismatchError

            raise DOIMetadataMismatchError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_units_are_valid_pint(self) -> "SubmodelTarget":
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

    @model_validator(mode="after")
    def validate_parameter_units_match_model(self, info: ValidationInfo) -> "SubmodelTarget":
        """
        Validate calibration parameter units match expected model parameter units.

        Checks that the dimensionality of parameter units matches what the full QSP
        model expects. This catches errors like using concentration units (nanomolar)
        instead of amount units (nanomole) for secretion rates.

        Requires context:
            model_structure: ModelStructure instance with parameter definitions

        Note: When context is not provided, a warning is issued and validation is skipped.
        The production pipeline (JuliaTranslator, JointInferenceBuilder, immediate_processor)
        always provides model_structure context, ensuring validation occurs in production.
        """
        if not info.context or "model_structure" not in info.context:
            warnings.warn(
                "model_structure not provided in validation context. "
                "Skipping parameter unit validation against QSP model. "
                "Use SubmodelTarget.model_validate(data, context={'model_structure': ...}) "
                "to enable unit validation.",
                UserWarning,
            )
            return self

        from qsp_llm_workflows.core.unit_registry import ureg

        model_structure = info.context["model_structure"]
        model_params = {p.name: p for p in model_structure.parameters}

        for param in self.calibration.parameters:
            model_param = model_params.get(param.name)
            if not model_param:
                # Unknown parameter - separate validator can handle this if needed
                continue

            try:
                expected = ureg(model_param.units)
                actual = ureg(param.units)
                if actual.dimensionality != expected.dimensionality:
                    raise ValueError(
                        f"Parameter '{param.name}' unit dimensionality mismatch:\n"
                        f"  SubmodelTarget units: '{param.units}' "
                        f"(dimensionality: {actual.dimensionality})\n"
                        f"  Model expected units: '{model_param.units}' "
                        f"(dimensionality: {expected.dimensionality})\n"
                        f"Common issue: using concentration (e.g., nanomolar) instead of "
                        f"amount (e.g., nanomole) for rate parameters."
                    )
            except ValueError:
                raise
            except Exception:
                # Unit parsing issue - handled by validate_units_are_valid_pint
                pass

        return self

    @model_validator(mode="after")
    def validate_algebraic_model_output_units(self) -> "SubmodelTarget":
        """
        Validate that AlgebraicModel.code output has correct units.

        Executes the forward model with sample parameter values and checks
        that the returned Quantity has units compatible with measurement.units.
        This catches bugs where the forward model returns wrong dimensionality.
        """
        model = self.calibration.model
        if model.type != "algebraic":
            return self

        if not hasattr(model, "code") or not model.code:
            return self

        from qsp_llm_workflows.core.unit_registry import ureg
        from qsp_llm_workflows.core.calibration.submodel_utils import get_prior_median
        import numpy as np

        # Build params dict with prior medians
        params = {}
        for param in self.calibration.parameters:
            median = get_prior_median(param.prior)
            if median is not None:
                params[param.name] = median

        if not params:
            return self  # Can't test without parameter values

        # Build inputs dict with units
        inputs_dict = {}
        for inp in self.inputs:
            try:
                inputs_dict[inp.name] = inp.value * ureg(inp.units)
            except Exception:
                inputs_dict[inp.name] = inp.value

        # Execute forward model
        try:
            local_scope = {"np": np, "numpy": np, "ureg": ureg}
            exec(model.code, local_scope)
            compute_fn = local_scope.get("compute")

            if compute_fn is None:
                return self  # Syntax validator handles this

            result = compute_fn(params, inputs_dict, ureg)

            # Check if result has units
            if not hasattr(result, "dimensionality"):
                # Result is dimensionless or raw number - check against measurement
                for measurement in self.calibration.measurements:
                    expected_units = ureg(measurement.units)
                    if expected_units.dimensionless:
                        continue  # OK - both dimensionless
                    warnings.warn(
                        f"AlgebraicModel.code returns dimensionless value, but "
                        f"measurement '{measurement.name}' expects units '{measurement.units}'.\n"
                        f"The forward model should return a Pint Quantity with correct units.",
                        UserWarning,
                    )
                return self

            # Check dimensionality matches measurement
            for measurement in self.calibration.measurements:
                try:
                    expected_units = ureg(measurement.units)
                    if result.dimensionality != expected_units.dimensionality:
                        raise ValueError(
                            f"AlgebraicModel.code output units mismatch:\n"
                            f"  Forward model returns: {result.units} "
                            f"(dimensionality: {result.dimensionality})\n"
                            f"  Measurement '{measurement.name}' expects: {measurement.units} "
                            f"(dimensionality: {expected_units.dimensionality})\n"
                            f"Check that the forward model formula produces the correct units."
                        )
                except ValueError:
                    raise
                except Exception:
                    pass  # Unit parsing issues handled elsewhere

        except ValueError:
            raise
        except Exception as e:
            # Execution errors - don't fail validation, other validators handle this
            warnings.warn(
                f"Could not validate AlgebraicModel.code output units: {e}",
                UserWarning,
            )

        return self

    # NOTE: validate_measurement_error_sd_units was removed and replaced by
    # validate_measurement_error_sd_units_for_likelihood which properly considers
    # the likelihood type (lognormal SD should be dimensionless, normal SD should
    # have measurement units).

    @model_validator(mode="after")
    def warn_input_measurement_unit_mismatch(self) -> "SubmodelTarget":
        """
        Warn when input units don't match measurement units.

        For simple cases (no conversion in code), the inputs used by a measurement
        should have the same units as the measurement. Warns if there's a mismatch,
        though some mismatches are intentional (code does the conversion).
        """
        from qsp_llm_workflows.core.unit_registry import ureg

        # Build input lookup
        input_map = {inp.name: inp for inp in self.inputs}

        for measurement in self.calibration.measurements:
            try:
                expected_units = ureg(measurement.units)
            except Exception:
                continue  # Invalid units handled elsewhere

            for input_name in measurement.uses_inputs:
                inp = input_map.get(input_name)
                if not inp:
                    continue  # Missing input handled elsewhere

                try:
                    input_units = ureg(inp.units)

                    if input_units.dimensionality != expected_units.dimensionality:
                        # Check if this is an AlgebraicModel (conversion expected)
                        model = self.calibration.model
                        if model.type == "algebraic":
                            # For algebraic models, mismatch might be intentional
                            # (the code does the conversion)
                            warnings.warn(
                                f"Input '{inp.name}' has units '{inp.units}' but "
                                f"measurement '{measurement.name}' expects '{measurement.units}'.\n"
                                f"If AlgebraicModel.code converts between these units, this is OK.\n"
                                f"Otherwise, check for unit errors.",
                                UserWarning,
                            )
                        else:
                            # For ODE/other models, this is more likely an error
                            warnings.warn(
                                f"Input '{inp.name}' has units '{inp.units}' "
                                f"(dimensionality: {input_units.dimensionality}) but "
                                f"measurement '{measurement.name}' expects '{measurement.units}' "
                                f"(dimensionality: {expected_units.dimensionality}).\n"
                                f"Check for unit conversion errors.",
                                UserWarning,
                            )

                except Exception:
                    pass  # Unit parsing issues handled elsewhere

        return self

    @model_validator(mode="after")
    def validate_prior_predictive_scale(self) -> "SubmodelTarget":
        """
        Validate that prior predictive is on same scale as observation.

        For all model types:
        1. Sample parameter values from the prior (using median)
        2. Run the model forward (ODE integration or direct conversion)
        3. Compute the observable
        4. Compare to the observation

        Raises error if they differ by more than 3 orders of magnitude.

        This catches unit conversion errors like k_CCL2_sec where:
        - Prior: LogNormal(-20.8, 1.0) -> median ~ 9e-10
        - Observation: 8.5 (raw input, not converted)
        - Mismatch: ~10 orders of magnitude
        """
        import math
        import numpy as np
        from qsp_llm_workflows.core.unit_registry import ureg
        from qsp_llm_workflows.core.calibration.submodel_utils import (
            get_prior_median,
            run_prior_predictive,
            PriorPredictiveError,
        )

        # Get all parameters and their prior medians
        if not self.calibration.parameters:
            return self

        # Build dict of all parameter medians (for multi-param models)
        all_param_medians = {}
        for param in self.calibration.parameters:
            median = get_prior_median(param.prior)
            if median is not None:
                all_param_medians[param.name] = median

        if not all_param_medians:
            return self  # No valid priors to check

        # Use first parameter for error messages (backwards compatible)
        param = self.calibration.parameters[0]
        prior_median = all_param_medians.get(param.name)

        # Build inputs dict with units
        inputs_dict = {}
        for inp in self.inputs:
            inputs_dict[inp.name] = inp.value * ureg(inp.units)
        input_values = {inp.name: inp.value for inp in self.inputs}

        # Validate each error model entry (check at least the first one)
        for entry in self.calibration.error_model[:1]:  # Just first entry for now
            obs_median = None

            if entry and entry.observation_code:
                try:
                    local_scope = {"np": np, "numpy": np, "ureg": ureg}
                    exec(entry.observation_code, local_scope)
                    derive_observation = local_scope.get("derive_observation")
                    if derive_observation:
                        result = derive_observation(inputs_dict, entry.sample_size, ureg)
                        if isinstance(result, dict) and "value" in result:
                            value = result["value"]
                            obs_median = value.magnitude if hasattr(value, "magnitude") else value
                except Exception:
                    pass  # Fall through to error below

            if obs_median is None:
                raise ValueError(
                    f"Prior predictive check failed: could not extract observation value.\n"
                    f"  Error model: {entry.name if entry else 'None'}\n\n"
                    f"Check that observation_code returns {{'value': <Quantity>, 'sd': ...}}"
                )

            if obs_median == 0:
                continue  # Zero observation is valid, just can't do log comparison

            # Run prior predictive to get model prediction
            try:
                predicted = run_prior_predictive(
                    model=self.calibration.model,
                    prior=param.prior,
                    param_name=param.name,
                    state_variables=self.calibration.state_variables,
                    independent_variable=self.calibration.independent_variable,
                    measurement=entry,
                    input_values=input_values,
                    all_param_medians=all_param_medians,
                )
            except PriorPredictiveError as e:
                raise ValueError(
                    f"Prior predictive check failed:\n  {e}\n\n"
                    f"  Model type: {self.calibration.model.type}\n"
                    f"  Parameters: {list(all_param_medians.keys())}"
                ) from e

            if predicted == 0:
                continue  # Zero prediction is valid, just can't do log comparison

            # Compare prediction to observation
            try:
                log_diff = abs(math.log10(abs(predicted)) - math.log10(abs(obs_median)))
            except (ValueError, ZeroDivisionError):
                continue

            if log_diff > 3:
                raise ValueError(
                    f"Prior predictive check failed for error model '{entry.name}':\n"
                    f"  Parameters: {all_param_medians}\n"
                    f"  Model prediction: {predicted:.2e}\n"
                    f"  Observation: {obs_median:.2e}\n"
                    f"  Difference: ~{10**log_diff:.0e}x ({log_diff:.1f} orders of magnitude)\n\n"
                    f"This indicates a unit conversion error. Check:\n"
                    f"  1. Prior parameters (mu, sigma) match the parameter units\n"
                    f"  2. observation_code correctly converts input units to parameter units\n"
                    f"  3. Input values and units are correct\n"
                    f"  4. observable.state_variables or observable.code correctly extracts the prediction"
                )

        return self

    @model_validator(mode="after")
    def validate_clipping_suggests_lognormal(self) -> "SubmodelTarget":
        """
        Warn if observation_code uses clipping to avoid negative values.

        Clipping (np.clip, np.maximum, max(0, ...)) suggests the data is
        positive-only, which is better modeled with a lognormal distribution
        than a normal distribution with clipping.

        Normal distributions for positive-only data introduce bias when clipped.
        """
        for entry in self.calibration.error_model:
            if entry.observation_code is None:
                continue

            code = entry.observation_code
            clipping_patterns = ["np.clip", "np.maximum", "np.minimum", "max(0", "min("]

            if any(pattern in code for pattern in clipping_patterns):
                warnings.warn(
                    f"Error model '{entry.name}' observation_code uses clipping "
                    f"(np.clip/np.maximum/etc) to avoid negative values.\n"
                    f"This suggests size/volume/mass data that may be better modeled with "
                    f"a lognormal distribution.\n"
                    f"Normal distributions for positive-only data introduce bias when clipped.\n"
                    f"Consider converting mean +/- SD to lognormal parameters:\n"
                    f"  mu_log = ln(mean^2 / sqrt(mean^2 + sd^2))\n"
                    f"  sigma_log = sqrt(ln(1 + sd^2/mean^2))",
                    UserWarning,
                )

        return self

    @model_validator(mode="after")
    def validate_no_hardcoded_values_in_observation_code(self) -> "SubmodelTarget":
        """
        Error if observation_code contains hardcoded numeric values that should be inputs.

        All numerical values (except minimal constants) must enter through the inputs
        dict to ensure traceability and facilitate sensitivity analysis.

        Allowed constants:
        - 0, 1, 2 (indexing, basic arithmetic)
        - 1.96 (95% CI convention)

        NOT allowed (should be inputs):
        - Time conversions like 24.0, 60.0 (use ureg instead)
        - Percentage conversions like 100.0 (use ureg instead)
        - Assumed fractions like 0.5, 0.25 (add as assumed_value input)
        - Fold uncertainties like 3.0, 5.0, 10.0 (add as assumed_value input)
        """
        import ast

        # Minimal set of truly necessary constants
        ALLOWED_CONSTANTS = {0, 0.0, 1, 1.0, 2, 2.0, 1.96}

        errors = []

        for entry in self.calibration.error_model:
            if entry.observation_code is None:
                continue

            try:
                tree = ast.parse(entry.observation_code)
            except SyntaxError:
                # Syntax errors caught by other validators
                continue

            # Find all numeric literals in the code
            suspicious_values = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    value = node.value
                    # Skip allowed constants
                    if value in ALLOWED_CONSTANTS:
                        continue
                    # Skip values close to allowed (floating point tolerance)
                    if any(abs(value - allowed) < 1e-10 for allowed in ALLOWED_CONSTANTS):
                        continue
                    suspicious_values.append(value)

            if suspicious_values:
                unique_values = sorted(set(suspicious_values))
                errors.append(
                    f"Error model '{entry.name}' observation_code contains hardcoded "
                    f"numeric values: {unique_values}\n"
                    f"  All numerical parameters must be defined as inputs, not hardcoded.\n"
                    f"  Add these as inputs with input_type='assumed_value' and document in value_snippet.\n"
                    f"  For unit conversions, use ureg (e.g., inp.to('day').magnitude) instead of magic numbers."
                )

        if errors:
            raise ValueError(
                "Hardcoded values in observation_code:\n\n" + "\n\n".join(errors)
            )

        return self

    @model_validator(mode="after")
    def validate_large_variance_documented(self) -> "SubmodelTarget":
        """
        Warn if large variance (CV > 50%) is not documented in identifiability_notes.

        High coefficient of variation suggests substantial variability that should
        be acknowledged and explained in the identifiability discussion.
        """
        # Find mean and SD/SE inputs
        mean_input = None
        std_input = None

        for inp in self.inputs:
            name_lower = inp.name.lower()
            # Skip vector-valued inputs
            if isinstance(inp.value, list):
                continue

            if "mean" in name_lower and mean_input is None:
                mean_input = inp
            if any(x in name_lower for x in ["sd", "std", "se", "stderr", "stdev"]):
                std_input = inp

        if mean_input and std_input:
            mean_val = mean_input.value
            std_val = std_input.value

            if mean_val != 0:
                cv = abs(std_val / mean_val)

                if cv > 0.5:  # CV > 50%
                    # Check if identifiability_notes mentions variance
                    notes = self.calibration.identifiability_notes.lower()
                    variance_keywords = [
                        "variance",
                        "variability",
                        "uncertain",
                        "cv",
                        "wide",
                        "heterogen",
                        "variation",
                        "spread",
                        "dispersion",
                    ]

                    if not any(keyword in notes for keyword in variance_keywords):
                        warnings.warn(
                            f"Large coefficient of variation (CV = {cv:.1%}) detected "
                            f"but not discussed in identifiability_notes.\n"
                            f"Consider adding discussion of whether this reflects:\n"
                            f"  - Biological variability\n"
                            f"  - Measurement error\n"
                            f"  - Heterogeneous population",
                            UserWarning,
                        )

        return self

    @model_validator(mode="after")
    def validate_observation_code_return_signature(self) -> "SubmodelTarget":
        """
        Validate that observation_code returns the required 'value' and 'sd' keys.

        The derive_observation function MUST return a dict with:
        - 'value': the observed point estimate (Pint Quantity)
        - 'sd': standard deviation of the measurement error (float or Quantity)

        This catches malformed observation_code that would fail at runtime.
        """
        import ast

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            code = entry.observation_code

            # Parse the code
            try:
                tree = ast.parse(code)
            except SyntaxError:
                # Syntax errors are caught by validate_custom_code_syntax
                continue

            # Find all return statements in derive_observation function
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "derive_observation":
                    # Find return statements within this function
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value:
                            # Check if return value is a dict
                            if isinstance(child.value, ast.Dict):
                                # Extract key names from the dict literal
                                keys = set()
                                for key in child.value.keys:
                                    if isinstance(key, ast.Constant):
                                        keys.add(key.value)
                                    elif isinstance(key, ast.Str):  # Python 3.7 compat
                                        keys.add(key.s)

                                # Check for required keys
                                missing_keys = {"value", "sd"} - keys
                                if missing_keys:
                                    raise ValueError(
                                        f"Error model '{entry.name}' observation_code "
                                        f"return dict is missing required key(s): {missing_keys}\n\n"
                                        f"Required return signature:\n"
                                        f"  return {{'value': <observed_quantity>, 'sd': <uncertainty>}}"
                                    )

            # Also check via regex for return statements that build dict inline
            # This catches cases where AST parsing might miss dynamic dict construction
            if "return" in code and "derive_observation" in code:
                import re

                has_value = bool(re.search(r"['\"]value['\"]", code))
                has_sd = bool(re.search(r"['\"]sd['\"]", code))

                if not has_value or not has_sd:
                    missing = []
                    if not has_value:
                        missing.append("'value'")
                    if not has_sd:
                        missing.append("'sd'")
                    raise ValueError(
                        f"Error model '{entry.name}' observation_code "
                        f"appears to be missing required return key(s): {', '.join(missing)}\n\n"
                        f"Required return signature:\n"
                        f"  return {{'value': <observed_quantity>, 'sd': <uncertainty>}}"
                    )

        return self

    @model_validator(mode="after")
    def validate_no_invisible_characters(self) -> "SubmodelTarget":
        """
        Validate that string fields don't contain invisible/control characters.

        These characters can be accidentally copy-pasted from PDFs and cause
        subtle parsing or display issues. Visible Unicode (Greek letters,
        accented characters, math symbols) is allowed.

        Blocked characters:
        - Control characters (U+0000-U+001F, U+007F-U+009F)
        - Zero-width characters (U+200B-U+200D, U+FEFF)
        - Soft hyphen (U+00AD)
        - Other invisible formatting characters
        """
        import unicodedata

        errors = []

        # Characters that are invisible or cause issues
        INVISIBLE_CHARS = {
            # Zero-width characters
            "\u200B",  # Zero-width space
            "\u200C",  # Zero-width non-joiner
            "\u200D",  # Zero-width joiner
            "\uFEFF",  # Byte order mark / zero-width no-break space
            # Soft hyphen
            "\u00AD",  # Soft hyphen (invisible in most contexts)
            # Other problematic invisibles
            "\u2060",  # Word joiner
            "\u2061",  # Function application
            "\u2062",  # Invisible times
            "\u2063",  # Invisible separator
            "\u2064",  # Invisible plus
            "\u180E",  # Mongolian vowel separator
        }

        def check_invisible(value: str, location: str) -> None:
            """Check if string contains invisible/control characters."""
            if not value:
                return
            for i, char in enumerate(value):
                cp = ord(char)
                # Control characters (C0 and C1 control codes, except common whitespace)
                if cp < 0x20 and char not in "\t\n\r":
                    errors.append(
                        f"{location}: contains control character " f"U+{cp:04X} at position {i}"
                    )
                elif 0x7F <= cp <= 0x9F:
                    errors.append(
                        f"{location}: contains control character " f"U+{cp:04X} at position {i}"
                    )
                elif char in INVISIBLE_CHARS:
                    name = unicodedata.name(char, f"U+{cp:04X}")
                    errors.append(
                        f"{location}: contains invisible character '{name}' "
                        f"(U+{cp:04X}) at position {i}"
                    )

        def check_all_strings(obj, prefix: str) -> None:
            """Recursively check all string fields."""
            if obj is None:
                return
            if isinstance(obj, str):
                check_invisible(obj, prefix)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_all_strings(item, f"{prefix}[{i}]")
            elif isinstance(obj, dict):
                for key, val in obj.items():
                    check_all_strings(val, f"{prefix}.{key}")

        # Check all string fields via model dump
        data = self.model_dump()
        check_all_strings(data, "")

        if errors:
            raise ValueError(
                "Invisible/control characters detected (likely from PDF copy-paste):\n  - "
                + "\n  - ".join(errors[:10])  # Limit to first 10 errors
                + ("\n  ... and more" if len(errors) > 10 else "")
                + "\n\nThese characters are invisible but can cause parsing issues. "
                + "Re-type the affected text manually."
            )

        return self

    # -------------------------------------------------------------------------
    # SOURCE RELEVANCE VALIDATORS
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_source_quality_peer_reviewed(self) -> "SubmodelTarget":
        """Warn about non-peer-reviewed primary sources (Wikipedia, preprints)."""
        if self.source_relevance.source_quality == SourceQuality.NON_PEER_REVIEWED:
            warnings.warn(
                f"Primary source quality is 'non_peer_reviewed' for target '{self.target_id}'.\n"
                "This includes Wikipedia, preprints, and unreviewed sources.\n\n"
                "Recommended actions:\n"
                "  1. Replace with peer-reviewed primary literature if possible\n"
                "  2. Document rationale in key_study_limitations\n"
                "  3. Increase estimated_translation_uncertainty_fold (recommend 3x minimum)\n"
                "  4. Ensure prior σ reflects this additional uncertainty",
                UserWarning,
            )
        return self

    @model_validator(mode="after")
    def validate_secondary_sources_quality(self) -> "SubmodelTarget":
        """Warn about non-peer-reviewed secondary sources providing quantitative values."""
        if not self.secondary_data_sources:
            return self

        for source in self.secondary_data_sources:
            if source.source_quality == SourceQuality.NON_PEER_REVIEWED:
                # Check if this source is referenced in any input
                source_used_for_values = False
                for inp in self.inputs:
                    if inp.source_ref == source.source_tag:
                        source_used_for_values = True
                        break

                if source_used_for_values:
                    warnings.warn(
                        f"Secondary source '{source.source_tag}' is non-peer-reviewed "
                        f"AND provides values used in calibration.\n"
                        f"This is a significant reliability concern. Consider:\n"
                        f"  1. Finding a peer-reviewed source for these values\n"
                        f"  2. Documenting in key_study_limitations\n"
                        f"  3. Widening the prior to reflect additional uncertainty",
                        UserWarning,
                    )

        return self

    @model_validator(mode="after")
    def validate_cross_indication_uncertainty(self) -> "SubmodelTarget":
        """Flag cross-indication extrapolation with insufficient uncertainty."""
        sr = self.source_relevance
        if sr.indication_match in (IndicationMatch.PROXY, IndicationMatch.UNRELATED):
            if sr.estimated_translation_uncertainty_fold < 3.0:
                raise ValueError(
                    f"Cross-indication extrapolation ({sr.indication_match.value}) "
                    f"with translation uncertainty of only {sr.estimated_translation_uncertainty_fold}x.\n\n"
                    f"Recommended minimum uncertainty:\n"
                    f"  - proxy: 3-10x\n"
                    f"  - unrelated: 10-100x\n\n"
                    f"Increase estimated_translation_uncertainty_fold to reflect "
                    f"biological differences between source and target indications."
                )
        return self

    @model_validator(mode="after")
    def validate_pharmacological_perturbation_justification(self) -> "SubmodelTarget":
        """Flag pharmacological perturbation without justification."""
        sr = self.source_relevance
        if sr.perturbation_type == PerturbationType.PHARMACOLOGICAL:
            if not sr.perturbation_relevance:
                raise ValueError(
                    "Perturbation type is 'pharmacological' but perturbation_relevance "
                    "is not provided.\n\n"
                    "When using drug-induced measurements to estimate physiological parameters, "
                    "you must explain:\n"
                    "  - Whether the value represents an upper/lower bound or typical value\n"
                    "  - Whether scaling or adjustment is needed\n"
                    "  - How supraphysiological drug concentrations affect interpretation"
                )
        return self

    @model_validator(mode="after")
    def validate_genetic_perturbation_justification(self) -> "SubmodelTarget":
        """Flag genetic perturbation without justification."""
        sr = self.source_relevance
        if sr.perturbation_type == PerturbationType.GENETIC:
            if not sr.perturbation_relevance:
                raise ValueError(
                    "Perturbation type is 'genetic_perturbation' but perturbation_relevance "
                    "is not provided.\n\n"
                    "When using KO/knockdown/overexpression data to estimate physiological "
                    "parameters, you must explain:\n"
                    "  - How the genetic perturbation relates to the wild-type parameter\n"
                    "  - Whether the measurement provides bounds or direct estimates\n"
                    "  - Compensatory mechanisms that may affect interpretation"
                )
        return self

    @model_validator(mode="after")
    def validate_low_tme_compatibility_uncertainty(self) -> "SubmodelTarget":
        """Flag low TME compatibility with insufficient uncertainty."""
        sr = self.source_relevance
        if sr.tme_compatibility == TMECompatibility.LOW:
            if sr.estimated_translation_uncertainty_fold < 10.0:
                raise ValueError(
                    f"TME compatibility is 'low' but translation uncertainty is only "
                    f"{sr.estimated_translation_uncertainty_fold}x.\n\n"
                    f"Low TME compatibility (e.g., T cell-permissive model for T cell-excluded "
                    f"tumor like PDAC) typically requires 10-100x uncertainty.\n\n"
                    f"Increase estimated_translation_uncertainty_fold to reflect major "
                    f"microenvironment differences."
                )
            if not sr.tme_compatibility_notes:
                raise ValueError(
                    "TME compatibility is 'low' but tme_compatibility_notes is not provided.\n\n"
                    "Document the specific TME differences and their expected impact:\n"
                    "  - Stromal density differences\n"
                    "  - Immune infiltration patterns\n"
                    "  - Chemokine/cytokine milieu\n"
                    "  - Expected direction and magnitude of bias"
                )
        return self

    @model_validator(mode="after")
    def validate_cross_species_uncertainty(self) -> "SubmodelTarget":
        """Flag cross-species extrapolation with insufficient uncertainty."""
        sr = self.source_relevance
        if sr.species_source != sr.species_target:
            if sr.estimated_translation_uncertainty_fold < 2.0:
                raise ValueError(
                    f"Cross-species extrapolation ({sr.species_source} -> {sr.species_target}) "
                    f"with translation uncertainty of only {sr.estimated_translation_uncertainty_fold}x.\n\n"
                    f"Cross-species extrapolation typically requires at least 2-3x uncertainty "
                    f"to account for species-specific biology.\n\n"
                    f"Increase estimated_translation_uncertainty_fold."
                )
        return self

    @model_validator(mode="after")
    def validate_prior_reflects_translation_uncertainty(self) -> "SubmodelTarget":
        """
        Warn if prior σ doesn't reflect the estimated translation uncertainty.

        For lognormal priors, σ on the log scale corresponds to multiplicative uncertainty:
        - σ = 0.69 → ~2x uncertainty (e^0.69 ≈ 2)
        - σ = 1.10 → ~3x uncertainty (e^1.10 ≈ 3)
        - σ = 2.30 → ~10x uncertainty (e^2.30 ≈ 10)
        - σ = 3.00 → ~20x uncertainty (e^3.00 ≈ 20)

        The prior σ should be at least ln(estimated_translation_uncertainty_fold) to
        properly capture source-to-target translation uncertainty.
        """
        import math

        sr = self.source_relevance
        translation_fold = sr.estimated_translation_uncertainty_fold

        # Only check if translation uncertainty is significant (>1.5x)
        if translation_fold <= 1.5:
            return self

        # Compute minimum recommended σ for lognormal prior
        min_recommended_sigma = math.log(translation_fold)

        for param in self.calibration.parameters:
            if param.prior is None:
                continue

            # Only applies to lognormal priors (most common for rate constants)
            if param.prior.distribution != "lognormal":
                continue

            actual_sigma = param.prior.sigma
            if actual_sigma is None:
                continue

            # Check if prior σ is at least 70% of recommended (allow some flexibility)
            if actual_sigma < 0.7 * min_recommended_sigma:
                warnings.warn(
                    f"Parameter '{param.name}' has lognormal prior σ={actual_sigma:.2f}, "
                    f"but estimated_translation_uncertainty_fold={translation_fold}x "
                    f"suggests σ should be at least {min_recommended_sigma:.2f}.\n\n"
                    f"The prior may be too narrow to capture source-to-target translation "
                    f"uncertainty. Consider:\n"
                    f"  - Increasing prior σ to ~{min_recommended_sigma:.1f}\n"
                    f"  - Or reducing estimated_translation_uncertainty_fold if justified\n\n"
                    f"Reference: σ=ln(fold) for lognormal → "
                    f"σ={math.log(3):.2f} for 3x, σ={math.log(10):.2f} for 10x",
                    UserWarning,
                )

        return self

    @model_validator(mode="after")
    def validate_algebraic_prior_predictive(self) -> "SubmodelTarget":
        """
        For algebraic models, validate that the forward model prediction
        is consistent with measured data.

        Uses the prior median for parameters, runs the forward model,
        and compares predicted observable to measured data. If they differ
        by more than 10x, there's likely a unit error in the model code.

        This is a prior predictive check: does the model (with reasonable
        parameter values) predict observables on the same scale as the data?
        """
        import math

        # Only applies to algebraic models
        if self.calibration.model.type != "algebraic":
            return self

        model = self.calibration.model
        if not hasattr(model, 'code') or not model.code:
            return self

        # Build inputs dict with pint quantities
        from qsp_llm_workflows.core.unit_registry import create_unit_registry

        ureg = create_unit_registry()
        inputs = {}
        for inp in self.inputs:
            if inp.units and inp.units != "dimensionless":
                try:
                    inputs[inp.name] = inp.value * ureg(inp.units)
                except Exception:
                    inputs[inp.name] = inp.value * ureg.dimensionless
            else:
                inputs[inp.name] = inp.value * ureg.dimensionless

        # Build params dict from prior medians
        params = {}
        for param in self.calibration.parameters:
            if param.prior is None:
                continue
            if param.prior.distribution == "lognormal":
                params[param.name] = math.exp(param.prior.mu)
            elif param.prior.distribution == "normal":
                params[param.name] = param.prior.mu
            elif param.prior.distribution == "uniform":
                params[param.name] = (param.prior.lower + param.prior.upper) / 2

        if not params:
            return self

        # Execute the forward model
        local_ns: dict = {}
        try:
            exec(model.code, {"__builtins__": __builtins__}, local_ns)
        except Exception:
            # Syntax errors caught by other validators
            return self

        if "compute" not in local_ns:
            return self

        import numpy as np
        try:
            predicted = local_ns["compute"](params, inputs, ureg)
        except Exception:
            # Execution errors caught by other validators
            return self

        # Handle multi-output forward models (dict return type)
        if isinstance(predicted, dict):
            # For multi-output models, we skip this simple check and rely on
            # validate_prior_predictive_scale which handles dict outputs properly
            # via the error_model's observable specification
            return self

        # Get magnitude for comparison
        if hasattr(predicted, 'magnitude'):
            predicted_value = float(predicted.magnitude)
        else:
            predicted_value = float(predicted)

        if predicted_value <= 0:
            return self

        # Compare to measured data (inputs with role=target)
        threshold_fold = 10.0
        for inp in self.inputs:
            if inp.role != InputRole.TARGET:
                continue

            measured_value = inp.value
            if measured_value <= 0:
                continue

            ratio = predicted_value / measured_value

            if ratio > threshold_fold or ratio < 1.0 / threshold_fold:
                raise ValueError(
                    f"Prior predictive check failed for algebraic model:\n"
                    f"  Predicted observable (from model.code): {predicted_value:.2e}\n"
                    f"  Measured data ('{inp.name}'): {measured_value:.2e}\n"
                    f"  Ratio: {ratio:.1f}x\n\n"
                    f"The forward model prediction differs from measured data by {abs(ratio):.0f}x.\n"
                    f"This likely indicates a unit error in AlgebraicModel.code.\n\n"
                    f"Check that model.code correctly computes the observable from parameters."
                )

        return self

    # -------------------------------------------------------------------------
    # ADDITIONAL VALIDATORS (from validator_ideas.md)
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_sample_size_list_length(self) -> "SubmodelTarget":
        """
        Validate sample_size list length matches evaluation_points.

        If sample_size is a list, it must have the same length as evaluation_points.
        This catches common mistakes like providing sample sizes for only some time points.

        Example of bug this catches:
            evaluation_points: [7, 14, 21]  # 3 time points
            sample_size: [10, 12]           # Only 2 sample sizes - mismatch!
        """
        errors = []

        for entry in self.calibration.error_model:
            if entry.evaluation_points and isinstance(entry.sample_size, list):
                if len(entry.sample_size) != len(entry.evaluation_points):
                    errors.append(
                        f"Error model '{entry.name}': sample_size has {len(entry.sample_size)} "
                        f"elements but evaluation_points has {len(entry.evaluation_points)}. "
                        f"They must match when sample_size is a list."
                    )

        if errors:
            raise ValueError(
                "Sample size / evaluation points length mismatch:\n  - " + "\n  - ".join(errors)
            )

        return self

    @model_validator(mode="after")
    def validate_ode_requires_observable(self) -> "SubmodelTarget":
        """
        Validate that ODE models have observable defined in error_model.

        For ODE models, the observable transforms state variables into the measured
        quantity. Without it, inference doesn't know what to compare model output to.

        ODE model types: first_order_decay, exponential_growth, logistic,
        michaelis_menten, two_state, saturation, custom_ode
        """
        ode_model_types = {
            "first_order_decay",
            "exponential_growth",
            "logistic",
            "michaelis_menten",
            "two_state",
            "saturation",
            "custom_ode",
        }

        model_type = self.calibration.model.type
        if model_type not in ode_model_types:
            return self  # Not an ODE model

        errors = []
        for entry in self.calibration.error_model:
            if entry.observable is None:
                errors.append(
                    f"Error model '{entry.name}': ODE model type '{model_type}' "
                    f"requires observable to be defined. The observable specifies how to "
                    f"transform state variables into the measured quantity."
                )

        if errors:
            raise ValueError(
                "Missing observable for ODE model:\n  - " + "\n  - ".join(errors)
            )

        return self

    @model_validator(mode="after")
    def warn_likelihood_distribution_unknown(self) -> "SubmodelTarget":
        """
        Warn if likelihood distribution is not a known type.

        Catches typos like 'lognomal' instead of 'lognormal'.

        Known distributions: normal, lognormal, truncated_normal, half_normal,
        beta, dirichlet, student_t, exponential, gamma, poisson
        """
        KNOWN_LIKELIHOODS = {
            "normal",
            "lognormal",
            "truncated_normal",
            "half_normal",
            "beta",
            "dirichlet",
            "student_t",
            "exponential",
            "gamma",
            "poisson",
        }

        for entry in self.calibration.error_model:
            dist = entry.likelihood.distribution.lower()
            if dist not in KNOWN_LIKELIHOODS:
                warnings.warn(
                    f"Error model '{entry.name}': likelihood distribution '{dist}' "
                    f"is not a recognized type.\n"
                    f"Known types: {sorted(KNOWN_LIKELIHOODS)}.\n"
                    f"If this is intentional (custom distribution), ignore this warning.",
                    UserWarning,
                )

        return self

    def _compute_confidence_score(self) -> float:
        """
        Compute overall confidence score based on source relevance factors.

        Returns a score from 0.0 to 1.0 where:
        - 1.0 = Perfect match (exact indication, human clinical, no perturbation)
        - 0.5 = Moderate confidence (related indication, animal data)
        - <0.3 = Low confidence (proxy/unrelated, non-peer-reviewed)
        """
        sr = self.source_relevance
        score = 1.0

        # Indication match penalty
        match_penalties = {
            IndicationMatch.EXACT: 1.0,
            IndicationMatch.RELATED: 0.8,
            IndicationMatch.PROXY: 0.5,
            IndicationMatch.UNRELATED: 0.2,
        }
        score *= match_penalties.get(sr.indication_match, 0.5)

        # Source quality penalty
        quality_penalties = {
            SourceQuality.PRIMARY_HUMAN_CLINICAL: 1.0,
            SourceQuality.PRIMARY_HUMAN_IN_VITRO: 0.9,
            SourceQuality.PRIMARY_ANIMAL_IN_VIVO: 0.8,
            SourceQuality.PRIMARY_ANIMAL_IN_VITRO: 0.7,
            SourceQuality.REVIEW: 0.6,
            SourceQuality.TEXTBOOK: 0.5,
            SourceQuality.NON_PEER_REVIEWED: 0.3,
        }
        score *= quality_penalties.get(sr.source_quality, 0.5)

        # Species penalty
        if sr.species_source != sr.species_target:
            score *= 0.8

        # TME penalty for low compatibility
        if sr.tme_compatibility == TMECompatibility.LOW:
            score *= 0.5
        elif sr.tme_compatibility == TMECompatibility.MODERATE:
            score *= 0.8

        # Perturbation penalty
        if sr.perturbation_type == PerturbationType.PHARMACOLOGICAL:
            score *= 0.8
        elif sr.perturbation_type == PerturbationType.GENETIC:
            score *= 0.7

        return round(score, 2)

    @property
    def confidence_score(self) -> float:
        """Computed confidence score based on source relevance factors."""
        return self._compute_confidence_score()


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
    # Source relevance enums
    "IndicationMatch",
    "SourceQuality",
    "PerturbationType",
    "TMECompatibility",
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
    "AlgebraicModel",
    "DirectFitModel",
    "CustomODEModel",
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
    # Source relevance
    "SourceRelevanceAssessment",
    # Top-level
    "SubmodelTarget",
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
#    - For direct measurements, len(evaluation_points)
#      should equal len(uses_inputs)
#
# 4. validate_state_variables_for_ode_models
#    - ODE model types (first_order_decay, exponential_growth, logistic,
#      michaelis_menten, custom) require state_variables
#    - ODE model types require independent_variable with span
#
# 5. validate_model_fields
#    - algebraic requires formula
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
