#!/usr/bin/env python3
"""
Submodel-based calibration target models for QSP parameter inference.

This module implements the SubmodelTarget schema that separates:
- `inputs`: What was extracted from papers (with full provenance)
- `calibration`: How to use those inputs for inference
"""

import warnings
from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

# Import relevance enums from central location
from maple.core.calibration.enums import (
    IndicationMatch,
    PerturbationType,
    SourceQuality,
    TMECompatibility,
)
from maple.core.calibration.code_validator import find_accessed_params
from maple.core.calibration.exceptions import DimensionalityMismatchError
from maple.core.calibration.shared_models import (
    FigureExcerpt,
    SourceRelevanceAssessment,
    TableExcerpt,
)


# =============================================================================
# ENUMS
# =============================================================================


class InputType(str, Enum):
    """Type of input extracted from literature."""

    DIRECT_MEASUREMENT = (
        "direct_measurement"  # Value traceable to paper text (requires snippet/table_excerpt)
    )
    UNIT_CONVERSION = (
        "unit_conversion"  # Dimensionless conversion factor (e.g., IQR-to-SD, pM-per-nM)
    )
    REFERENCE_VALUE = (
        "reference_value"  # Normalization/reference constant (e.g., V_T_ref, tumor_cell_density)
    )
    DERIVED_ARITHMETIC = "derived_arithmetic"
    """Deterministic arithmetic derivation from other inputs.

    Use when a value is calculated from extracted inputs via an explicit
    formula (e.g., E = 3*G' for incompressible materials, slope * time,
    2 ROIs * 2 gels = 4 observations). The formula is evaluated and
    checked against the declared value. Snippet validation is skipped
    since the derived value won't appear in the source text.

    Requires ``formula`` and ``source_inputs`` fields on the Input.
    """


class CurveType(str, Enum):
    """Curve type for direct_fit models."""

    HILL = "hill"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class ObservableType(str, Enum):
    """Type of observable transformation."""

    IDENTITY = "identity"  # Return first state variable directly
    CUSTOM = "custom"  # User-provided code for any other transformation


# =============================================================================
# INPUTS
# =============================================================================


class Input(BaseModel):
    """
    A value extracted from literature with full provenance.

    Inputs are referenced by name from calibration.measurements and
    calibration.state_variables.initial_condition.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique identifier for this input (used in references)")
    value: float = Field(description="Extracted numeric value")
    units: str = Field(description="Units of the value")
    input_type: InputType = Field(
        description="Type of input: direct_measurement, unit_conversion, or reference_value"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Why this value was chosen. Required for unit_conversion and reference_value inputs.",
    )
    source_ref: str = Field(
        description="Reference to source_tag in primary_data_source or secondary_data_sources"
    )
    source_location: str = Field(
        description="Location within the source (e.g., 'Figure 3B, day 28')"
    )
    value_snippet: Optional[str] = Field(
        default=None,
        description="Exact text from paper containing the value (for validation). "
        "Use table_excerpt instead when the value comes from a table.",
    )
    table_excerpt: Optional[TableExcerpt] = Field(
        default=None,
        description="Structured table excerpt when value comes from a table. "
        "Preferred over value_snippet for table-sourced data.",
    )
    figure_excerpt: Optional[FigureExcerpt] = Field(
        default=None,
        description="Structured figure excerpt when value is read from a figure. "
        "Figure-derived values are flagged for manual review instead of "
        "failing snippet validation.",
    )
    source_inputs: Optional[List[str]] = Field(
        default=None,
        description="Names of other inputs used in the formula. "
        "Required for derived_arithmetic inputs. "
        "All referenced names must exist as inputs in the same target.",
    )
    formula: Optional[str] = Field(
        default=None,
        description="Arithmetic formula deriving this value from source_inputs. "
        "Required for derived_arithmetic inputs. Use input names directly "
        "in the expression (e.g., '3 * Gprime_stiff_kPa'). "
        "The validator evaluates this and checks it matches the declared value.",
    )


# =============================================================================
# CALIBRATION - PARAMETERS
# =============================================================================


class InlinePrior(BaseModel):
    """Inline prior specification for nuisance parameters."""

    model_config = ConfigDict(extra="forbid")

    distribution: str = Field(description="Distribution: lognormal, normal, or uniform")
    mu: Optional[float] = Field(default=None, description="Location parameter (lognormal, normal)")
    sigma: Optional[float] = Field(default=None, description="Scale parameter (lognormal, normal)")
    lower: Optional[float] = Field(default=None, description="Lower bound (uniform)")
    upper: Optional[float] = Field(default=None, description="Upper bound (uniform)")


class Parameter(BaseModel):
    """A parameter to be estimated during inference.

    Set ``nuisance=True`` for parameters that are needed by the forward model
    but are not part of the full QSP model (e.g., a proliferation rate that
    helps constrain the activation rate of interest). Nuisance parameters:
      - carry their own inline ``prior`` (since they are not in the priors CSV)
      - are sampled during MCMC alongside QSP parameters
      - are excluded from the output marginals and copula
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Parameter name from the full QSP model")
    units: str = Field(description="Parameter units")
    nuisance: bool = Field(
        default=False,
        description="If true, parameter is estimated during MCMC but excluded from output priors",
    )
    prior: Optional[InlinePrior] = Field(
        default=None,
        description="Inline prior specification. Required for nuisance parameters, "
        "forbidden for QSP parameters (which get priors from the CSV).",
    )

    @model_validator(mode="after")
    def validate_nuisance_prior(self):
        """Nuisance parameters must have an inline prior; non-nuisance parameters must not."""
        if self.nuisance and self.prior is None:
            raise ValueError(
                f"Nuisance parameter '{self.name}' must have an inline prior specification."
            )
        if not self.nuisance and self.prior is not None:
            raise ValueError(
                f"Non-nuisance parameter '{self.name}' must not have an inline prior. "
                f"QSP parameters get priors from the CSV."
            )
        return self


# =============================================================================
# CALIBRATION - STATE VARIABLES
# =============================================================================


class FixedInitialCondition(BaseModel):
    """Initial condition with a fixed/normalized value."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="Fixed initial value")
    rationale: str = Field(description="Why this value was chosen")


class InputRefInitialCondition(BaseModel):
    """Initial condition referencing a measured input."""

    model_config = ConfigDict(extra="forbid")

    input_ref: str = Field(description="Name of input to use as initial condition")
    rationale: str = Field(description="Why this input is used as IC")


class StateVariable(BaseModel):
    """A state variable in the ODE system."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    input_ref: str = Field(description="Name of input to use as fixed value")


class ReferenceRef(BaseModel):
    """Reference to a curated value in the reference database (reference_values.yaml)."""

    model_config = ConfigDict(extra="forbid")

    reference_ref: str = Field(description="Name of value in reference_values.yaml")


# ParameterRole can be a string (parameter name to estimate), InputRef (fixed from extraction data),
# or ReferenceRef (fixed from the curated reference database)
ParameterRole = Union[str, InputRef, ReferenceRef]


# =============================================================================
# CALIBRATION - FORWARD MODEL TYPES (each with specific required parameters)
# =============================================================================


class BaseForwardModelSpec(BaseModel):
    """Base class for all forward model specifications."""

    model_config = ConfigDict(extra="forbid")

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


# =============================================================================
# STEADY-STATE ALGEBRAIC MODEL TYPES (structured, auto-generated code)
# =============================================================================


class BaseSteadyStateModel(BaseForwardModelSpec):
    """Base class for all steady-state algebraic model types.

    Provides the unit_conversion_factor field shared by all steady-state models.
    """

    unit_conversion_factor: ParameterRole = Field(
        default="1.0",
        description="Scalar factor to correct for mismatched time units between rate "
        "parameters. E.g., if target_rate is per-minute and loss_rate is per-day, "
        "set to 1440.0 (minutes/day). Default 1.0 (rates share the same time unit).",
    )


class SteadyStateDensityModel(BaseSteadyStateModel):
    """Steady-state tissue cell density model.

    Maps a trafficking/recruitment rate to observed cell density (cells/mm^2)
    via decomposed model quantities. Use when paper reports IHC or mIF data.

    Formula: density = target_rate * source_pool * recruitment_efficiency
                       * (1 - exclusion_fraction) / loss_rate * section_volume_factor

    For absolute counts (cells/mL), set section_volume_factor = 1.0.
    """

    type: Literal["steady_state_density"] = "steady_state_density"
    target_rate: ParameterRole = Field(
        description="Trafficking/recruitment rate parameter (e.g., q_CD8_T_in, k_Mac_rec)"
    )
    source_pool: ParameterRole = Field(
        description="Circulating/source cell count (e.g., V_C.CD8 ~ 2e9 cells)"
    )
    loss_rate: ParameterRole = Field(description="Intratumoral cell loss/death rate (1/day)")
    section_volume_factor: ParameterRole = Field(
        description="Section volume per mm^2 (cm^3/mm^2). "
        "Standard 4-um section: 4e-6. Set to 1.0 for volumetric counts."
    )
    recruitment_efficiency: ParameterRole = Field(
        default="1.0",
        description="Chemokine-dependent recruitment efficiency, ~1 for large tumors. "
        "Default 1.0 (saturated).",
    )
    exclusion_fraction: ParameterRole = Field(
        default="0.0",
        description="Immune exclusion fraction (e.g., H_CXCL12 for T cells). "
        "0 for myeloid cells. Default 0.0.",
    )


class SteadyStateFractionModel(BaseSteadyStateModel):
    """Steady-state cell fraction model.

    Maps a recruitment/trafficking rate to observed cell fraction (% of parent
    population) from flow cytometry or quantitative IHC.

    Formula: fraction = target_rate * drive_factor / (loss_rate * parent_density)

    CRITICAL: parent_density provides the dimensional anchor. Without it,
    the model constrains only the ratio target_rate/loss_rate, which is
    incommensurable with absolute-rate targets.
    """

    type: Literal["steady_state_fraction"] = "steady_state_fraction"
    target_rate: ParameterRole = Field(
        description="Recruitment/trafficking rate parameter (e.g., k_MDSC_rec)"
    )
    loss_rate: ParameterRole = Field(description="Cell death/loss rate in tumor (1/day)")
    parent_density: ParameterRole = Field(
        description="Total parent population density (e.g., CD45+ cells/mL). "
        "Provides dimensional anchor for absolute rate inference."
    )
    drive_factor: ParameterRole = Field(
        default="1.0",
        description="Effective chemokine/cytokine drive factor. Default 1.0 (saturated).",
    )


class SteadyStateConcentrationModel(BaseSteadyStateModel):
    """Steady-state soluble factor concentration model.

    Maps a per-cell secretion rate to observed concentration (pg/mL, nM)
    from serum ELISA, tissue lysate, or similar assays.

    Formula: concentration = secretion_rate * source_count / (clearance_rate * distribution_volume)
    """

    type: Literal["steady_state_concentration"] = "steady_state_concentration"
    secretion_rate: ParameterRole = Field(
        description="Per-cell secretion rate parameter (e.g., k_CCL2_sec)"
    )
    source_count: ParameterRole = Field(
        description="Number of source cells (e.g., tumor cell count)"
    )
    clearance_rate: ParameterRole = Field(description="Degradation/clearance rate (1/day)")
    distribution_volume: ParameterRole = Field(
        description="Volume of distribution (e.g., V_blood ~ 5L, V_tumor)"
    )


class SteadyStateRatioModel(BaseSteadyStateModel):
    """Steady-state population ratio model.

    Maps relative rates to an observed ratio of two cell populations
    (e.g., M2:M1, CD4:CD8, Treg:Teff) from IHC or flow cytometry.

    Formula: ratio = rate_numerator * drive_numerator / (rate_denominator * drive_denominator)

    Confounding factors (total density, section geometry, staining efficiency)
    cancel in the ratio, making this the most robust observable type.
    """

    type: Literal["steady_state_ratio"] = "steady_state_ratio"
    rate_numerator: ParameterRole = Field(
        description="Rate governing the numerator population (e.g., k_M2_pol)"
    )
    rate_denominator: ParameterRole = Field(
        description="Rate governing the denominator population (e.g., k_M1_pol)"
    )
    drive_numerator: ParameterRole = Field(
        default="1.0",
        description="Drive factor for numerator population. Default 1.0.",
    )
    drive_denominator: ParameterRole = Field(
        default="1.0",
        description="Drive factor for denominator population. Default 1.0.",
    )


class SteadyStateProliferationIndexModel(BaseSteadyStateModel):
    """Steady-state proliferation index model.

    Maps a proliferation/growth rate to observed fraction of marker-positive
    cells (Ki-67+, BrdU+, EdU+) from IHC or flow cytometry.

    Formula: f_marker = prolif_rate * visible_duration / (prolif_rate * visible_duration + loss_rate)

    This is nonlinear (saturating) and cannot be expressed as a simple
    product/quotient, which is why it needs its own type.
    """

    type: Literal["steady_state_proliferation_index"] = "steady_state_proliferation_index"
    proliferation_rate: ParameterRole = Field(
        description="Growth/proliferation rate parameter (e.g., k_C1_growth)"
    )
    visible_duration: ParameterRole = Field(
        description="Duration of marker-visible cell cycle phase (days). "
        "Ki-67: ~1-2 days. BrdU S-phase: ~8 hours."
    )
    loss_rate: ParameterRole = Field(
        default="0.0",
        description="Effective cell loss rate (1/day). "
        "0 for net-growing populations, nonzero for steady-state tumors.",
    )


# =============================================================================
# IN VITRO ACCUMULATION MODEL
# =============================================================================


class BatchAccumulationModel(BaseForwardModelSpec):
    """Batch (in vitro) accumulation model for secretion assays.

    Maps a per-cell secretion rate to measured mass or concentration of
    analyte accumulated in culture medium over a fixed incubation period.

    Formula: predicted = secretion_rate * cell_count * incubation_time
                         * molecular_weight * unit_conversion_factor / medium_volume

    Use this for ELISA-based secretion assays where cells secrete a protein
    (e.g., CCL2, IL-6) into a known volume of medium over a known time, and
    the readout is mass (ng) or concentration (pg/mL).

    Unit conversion examples:
    - k in nmol/cell/day, output ng:
        ucf = 1.0 (since nmol * g/mol * 1e-9 * 1e9 = 1.0)
    - k in nmol/cell/day, output pg/mL:
        ucf = 1000.0 (since 1e-9 * 1e12 = 1e3), divide by V_mL
    - If incubation_time is in hours but secretion_rate is per-day:
        fold time conversion into ucf (e.g., 1000.0/24.0 = 41.6667)
    """

    type: Literal["batch_accumulation"] = "batch_accumulation"
    secretion_rate: ParameterRole = Field(
        description="Per-cell secretion rate parameter (e.g., k_CCL2_sec in nmol/cell/day)"
    )
    cell_count: ParameterRole = Field(description="Number of producing cells in the assay")
    incubation_time: ParameterRole = Field(
        description="Accumulation duration (should match time units of secretion_rate, "
        "or use unit_conversion_factor to bridge)"
    )
    molecular_weight: ParameterRole = Field(
        description="Molecular weight for mass-mole conversion (g/mol). "
        "Can be a reference_ref to the reference database."
    )
    medium_volume: ParameterRole = Field(
        default="1.0",
        description="Volume of medium (mL) for concentration outputs. "
        "Set to 1.0 for mass outputs (ng, pg).",
    )
    unit_conversion_factor: ParameterRole = Field(
        default="1.0",
        description="Mass/time unit conversion factor. "
        "E.g., 1.0 for ng output, 1000.0 for pg output, "
        "41.6667 for pg output with hours-to-days time conversion.",
    )


# =============================================================================
# GENERIC ALGEBRAIC MODEL (fallback for novel forward models)
# =============================================================================


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

    Prefer using a typed model (steady_state_density, steady_state_fraction,
    steady_state_concentration, steady_state_ratio,
    steady_state_proliferation_index, batch_accumulation) when applicable.
    Use AlgebraicModel only for relationships that don't fit any typed model.
    """

    type: Literal["algebraic"] = "algebraic"
    formula: str = Field(
        description="Descriptive formula showing the relationship "
        "(e.g., 't_half = ln(2) / k' or 'C_ss = k_prod / k_deg')"
    )
    code: str = Field(
        description="Python FORWARD model: given params, predict observable. "
        "Signature: def compute(params: dict, inputs: dict) -> float. "
        "Must use only np.* functions (mapped to jax.numpy at inference time). "
        "Example: return np.log(2) / params['k'] for predicting t_half from k."
    )

    @model_validator(mode="after")
    def lint_jax_traceability(self) -> "AlgebraicModel":
        """Warn if forward model code contains Python if/while statements.

        These cause JAX TracerBoolConversionError during numpyro MCMC because
        JAX traces through the function and cannot convert traced arrays to
        Python bools.  Use ``jnp.where()`` instead.
        """
        import ast
        import textwrap

        try:
            tree = ast.parse(textwrap.dedent(self.code))
        except SyntaxError:
            return self

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                kind = "if" if isinstance(node, ast.If) else "while"
                cond_src = ast.get_source_segment(self.code, node.test) or "<condition>"
                warnings.warn(
                    f"forward_model.code contains Python `{kind}` "
                    f"(line {node.lineno}: `{cond_src}`) which is not JAX-traceable "
                    f"and will cause TracerBoolConversionError during MCMC. "
                    f"Use jnp.where() or remove the branch.",
                    UserWarning,
                    stacklevel=2,
                )
            elif isinstance(node, ast.IfExp):
                cond_src = ast.get_source_segment(self.code, node.test) or "<condition>"
                warnings.warn(
                    f"forward_model.code contains inline ternary `x if {cond_src} else y` "
                    f"(line {node.lineno}) which is not JAX-traceable. "
                    f"Use jnp.where(condition, if_true, if_false) instead.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


class DirectFitModel(BaseForwardModelSpec):
    """Direct curve fitting (no ODE): dose-response curves, titrations.

    Auto-generates forward model code for common curve types.
    The x-value is provided per error model entry via the `x_input` field,
    enabling multi-point dose-response evaluation.

    Curve types:
    - hill: y = baseline + (maximum - baseline) / (1 + (x / ec50)^n_hill)
    - linear: y = slope * x + intercept
    - exponential: y = amplitude * exp(rate * x)
    """

    type: Literal["direct_fit"] = "direct_fit"
    curve: CurveType = Field(description="Curve type to fit (hill, linear, exponential)")

    # Hill curve fields (curve == "hill")
    ec50: Optional[ParameterRole] = Field(
        default=None,
        description="Half-maximal effective concentration (parameter to estimate or input_ref)",
    )
    n_hill: Optional[ParameterRole] = Field(
        default="1.0",
        description="Hill coefficient. Default 1.0 (standard Michaelis-Menten shape).",
    )
    baseline: Optional[ParameterRole] = Field(
        default="0.0",
        description="Response at zero dose. Default 0.0.",
    )
    maximum: Optional[ParameterRole] = Field(
        default="1.0",
        description="Maximum response at saturating dose. Default 1.0.",
    )

    # Linear fields (curve == "linear")
    slope: Optional[ParameterRole] = Field(
        default=None, description="Slope parameter (parameter to estimate or input_ref)"
    )
    intercept: Optional[ParameterRole] = Field(
        default="0.0", description="Y-intercept. Default 0.0."
    )

    # Exponential fields (curve == "exponential")
    amplitude: Optional[ParameterRole] = Field(
        default=None, description="Pre-exponential factor (parameter to estimate or input_ref)"
    )
    rate: Optional[ParameterRole] = Field(
        default=None, description="Exponential rate constant (parameter to estimate or input_ref)"
    )

    @model_validator(mode="after")
    def validate_curve_fields(self) -> "DirectFitModel":
        """Validate that required fields are populated for the chosen curve type."""
        from maple.core.calibration.exceptions import MissingFieldError

        if self.curve == CurveType.HILL:
            if self.ec50 is None:
                raise MissingFieldError("direct_fit with curve=hill requires 'ec50' field")
        elif self.curve == CurveType.LINEAR:
            if self.slope is None:
                raise MissingFieldError("direct_fit with curve=linear requires 'slope' field")
        elif self.curve == CurveType.EXPONENTIAL:
            if self.amplitude is None:
                raise MissingFieldError(
                    "direct_fit with curve=exponential requires 'amplitude' field"
                )
            if self.rate is None:
                raise MissingFieldError("direct_fit with curve=exponential requires 'rate' field")
        return self


class PowerLawModel(BaseForwardModelSpec):
    """Power-law scaling relationship: y = coefficient * (x / reference_x) ^ exponent.

    Common for biophysical scaling (stiffness vs collagen density, pore size vs concentration,
    diffusion coefficient vs gel density).
    """

    type: Literal["power_law"] = "power_law"
    coefficient: ParameterRole = Field(
        description="Reference value / coefficient (e.g., E_ref, d_pore_ref)"
    )
    reference_x: ParameterRole = Field(
        description="Reference x value for normalization (e.g., phi_col_ref, c_ref)"
    )
    exponent: ParameterRole = Field(
        description="Power-law exponent (parameter to estimate or fixed literal, e.g., '0.5')"
    )


class CustomODEModel(BaseForwardModelSpec):
    """Custom ODE with user-provided code."""

    type: Literal["custom_ode"] = "custom_ode"
    code: str = Field(
        description="Python ODE function. Signature: def ode(t, y, params, inputs) -> dict"
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
        SteadyStateDensityModel,
        SteadyStateFractionModel,
        SteadyStateConcentrationModel,
        SteadyStateRatioModel,
        SteadyStateProliferationIndexModel,
        BatchAccumulationModel,
        AlgebraicModel,
        DirectFitModel,
        PowerLawModel,
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

    model_config = ConfigDict(extra="forbid")

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
# =============================================================================
# CALIBRATION - ERROR MODEL
# =============================================================================


class Observable(BaseModel):
    """
    Transform model output into the measured quantity.

    For ODE models: transforms state variable vector into the observable.
    For algebraic models: selects/transforms from the dict returned by compute().
    """

    model_config = ConfigDict(extra="forbid")

    type: ObservableType = Field(description="Observable type")
    state_variables: Optional[List[str]] = Field(
        default=None,
        description="State variable(s) used in the observable. "
        "Required for ODE models (defines y vector ordering). "
        "Optional for algebraic models where y is the compute() return value.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Why this observable type was chosen",
    )
    code: Optional[str] = Field(
        default=None,
        description="Python code for custom observable transformation",
    )


class ErrorModel(BaseModel):
    """
    An error model entry specifying how to compare model predictions to data.

    The error model describes:
    - Which inputs from the data are used
    - How to generate bootstrap samples of the observation
    - The likelihood family is inferred from bootstrap samples via fit_distributions()

    For ODE models, evaluation_points specifies when to compare.
    For algebraic models, evaluation_points is not needed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Error model entry name")
    observable: Optional[Observable] = Field(
        default=None,
        description="How to compute the observable from state variables (for ODE models)",
    )
    units: str = Field(description="Units of the measurement")
    uses_inputs: Optional[List[str]] = Field(
        default=None,
        description="Names of inputs that feed this measurement",
    )
    x_input: Optional[str] = Field(
        default=None,
        description="Name of input providing the independent variable value (x) for this "
        "error model entry. Required for direct_fit and power_law forward models. "
        "Each entry can reference a different input, enabling multi-point evaluation "
        "(e.g., different doses in a dose-response curve).",
    )
    evaluation_points: Optional[List[float]] = Field(
        default=None,
        description="Points at which to evaluate the model (only for ODE models; "
        "units from forward_model.independent_variable)",
    )
    sample_size_input: str = Field(
        description="Name of input providing sample size",
    )
    observation_code: str = Field(
        description="Python code to generate bootstrap samples of the observation from inputs. "
        "Signature: def derive_observation(inputs, sample_size, rng, n_bootstrap) -> np.ndarray. "
        "The rng argument is a numpy Generator (np.random.default_rng) provided by the framework. "
        "The n_bootstrap argument is the number of samples to generate (from the n_bootstrap field). "
        "Must return a 1D numpy array of parametric bootstrap samples. "
        "The framework derives value (median), SD, and CI95 from the samples. "
        "Choose the bootstrap distribution to match the data-generating process: "
        "normal for mean±SD data, lognormal for positive quantities with log-space spread, "
        "poisson for count data, etc. "
        "For multi-timepoint models, this is called once per evaluation_point.",
    )
    n_bootstrap: int = Field(
        default=10000,
        description="Number of parametric bootstrap samples to generate. "
        "Default 10000 provides stable median/SD/CI95 estimates.",
    )


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

    model_config = ConfigDict(extra="forbid")

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
    """Primary literature data source. Requires DOI or PMID (for pre-DOI papers)."""

    model_config = ConfigDict(extra="forbid")

    doi: Optional[str] = Field(default=None, min_length=1, description="DOI of the source")
    pmid: Optional[str] = Field(
        default=None,
        min_length=1,
        description="PubMed ID (fallback for pre-DOI-era papers)",
    )
    title: Optional[str] = Field(default=None, description="Title of the source")
    authors: Optional[List[str]] = Field(default=None, description="Author list")
    year: Optional[int] = Field(default=None, description="Publication year")
    source_tag: str = Field(description="Short identifier for referencing (e.g., 'Smith2023')")
    source_relevance: SourceRelevanceAssessment = Field(
        description=(
            "Structured assessment of how well this source's data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
    )

    @model_validator(mode="after")
    def validate_doi_or_pmid(self) -> "PrimaryDataSource":
        """Ensure at least one of doi or pmid is provided."""
        if not self.doi and not self.pmid:
            raise ValueError(
                "Primary data source must have either 'doi' or 'pmid'. "
                "Use 'pmid' for pre-DOI-era papers."
            )
        return self


class SecondaryDataSource(BaseModel):
    """Secondary literature data source. Requires either DOI or URL."""

    model_config = ConfigDict(extra="forbid")

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
    source_relevance: SourceRelevanceAssessment = Field(
        description=(
            "Structured assessment of how well this source's data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
    )

    @model_validator(mode="after")
    def validate_doi_or_url(self) -> "SecondaryDataSource":
        """Ensure at least one of doi or url is provided."""
        if not self.doi and not self.url:
            from maple.core.calibration.exceptions import MissingFieldError

            raise MissingFieldError(
                f"Secondary source '{self.source_tag}' must have either doi or url"
            )
        return self

    @model_validator(mode="after")
    def warn_non_peer_reviewed_secondary(self) -> "SecondaryDataSource":
        """Warn if secondary source is non-peer-reviewed."""
        if self.source_relevance.source_quality == SourceQuality.NON_PEER_REVIEWED:
            warnings.warn(
                f"Secondary source '{self.source_tag}' is non-peer-reviewed.\n"
                f"If this source provides quantitative values used in calibration, "
                f"consider finding a peer-reviewed primary source instead.",
                UserWarning,
            )
        # Also warn if URL suggests non-peer-reviewed but source_quality not set
        if self.url and self.source_relevance.source_quality is None:
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


class CellType(BaseModel):
    """Cell type information (for primary cells)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Cell type name")
    phenotype: Optional[str] = Field(default=None, description="Cell phenotype")
    isolation_method: Optional[str] = Field(default=None, description="How cells were isolated")
    source: Optional[str] = Field(
        default=None,
        description="Where cells came from (e.g., 'peripheral blood from healthy donors')",
    )


class SubmodelCellLine(BaseModel):
    """Cell line information for submodel experimental context."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Cell line name")
    species: Optional[str] = Field(default=None, description="Species of origin")
    tissue_origin: Optional[str] = Field(default=None, description="Tissue of origin")
    cell_type: Optional[str] = Field(default=None, description="Cell type")
    source: Optional[str] = Field(default=None, description="Supplier or origin")
    passage_range: Optional[str] = Field(default=None, description="Passage range (e.g., 'P5-P15')")


class SubmodelCultureConditions(BaseModel):
    """Cell culture conditions for submodel experimental context."""

    model_config = ConfigDict(extra="forbid")

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
    medium: Optional[str] = Field(
        default=None, description="Culture medium (e.g., 'RPMI-1640 + 10% FCS')"
    )
    supplements: Optional[str] = Field(
        default=None, description="Media supplements (e.g., 'GM-CSF (50 ng/mL) + IL-4 (10 ng/mL)')"
    )
    duration: Optional[str] = Field(
        default=None,
        description="Culture duration (e.g., '7 days differentiation + 12h maturation')",
    )
    notes: Optional[str] = Field(default=None, description="Additional culture condition notes")


class ExperimentalContext(BaseModel):
    """Experimental context for the submodel target.

    NOTE: Only contains species, system, cell_lines, cell_types, culture_conditions,
    and indication. Fields like study_interpretation, key_assumptions, and
    key_study_limitations are TOP-LEVEL fields on SubmodelTarget, NOT nested here.
    """

    model_config = ConfigDict(extra="forbid")

    species: str = Field(description="Species (human, mouse, rat, etc.)")
    system: str = Field(
        description="Experimental system (in_vitro_primary_cells, in_vitro_immortalized, etc.)"
    )
    cell_lines: Optional[List[SubmodelCellLine]] = Field(
        default=None, description="Cell lines used"
    )
    cell_types: Optional[List[CellType]] = Field(
        default=None, description="Cell types used (for primary cells)"
    )
    culture_conditions: Optional[SubmodelCultureConditions] = Field(
        default=None, description="Culture conditions"
    )
    indication: Optional[str] = Field(default=None, description="Disease indication (PDAC, etc.)")

    @model_validator(mode="before")
    @classmethod
    def reject_misplaced_fields(cls, data: Any) -> Any:
        """Give actionable error when fields are nested here instead of top-level."""
        if isinstance(data, dict):
            misplaced = {
                "study_interpretation",
                "key_assumptions",
                "key_study_limitations",
            }
            found = misplaced & set(data.keys())
            if found:
                raise ValueError(
                    f"Fields {found} are top-level SubmodelTarget fields, "
                    f"NOT nested inside experimental_context. "
                    f"Move them out of experimental_context to the root level."
                )
        return data


# =============================================================================
# PARAMETER ROLE INTROSPECTION HELPER
# =============================================================================

# Fields on forward models that are NOT ParameterRole values
_NON_ROLE_FIELDS = frozenset(
    {
        "type",
        "code",
        "formula",
        "data_rationale",
        "submodel_rationale",
        "independent_variable",
        "state_variables",
        "curve",
    }
)


def _get_parameter_role_fields(model) -> dict:
    """Get all ParameterRole-typed fields and their values from a forward model.

    Iterates model_fields and returns {field_name: value} for fields whose
    values are str, InputRef, or ReferenceRef. Skips non-ParameterRole fields
    like 'type', 'code', 'data_rationale', etc.
    """
    result = {}
    for field_name in model.model_fields:
        if field_name in _NON_ROLE_FIELDS:
            continue
        value = getattr(model, field_name)
        if isinstance(value, (str, InputRef, ReferenceRef)):
            result[field_name] = value
    return result


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

    model_config = ConfigDict(extra="forbid")

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

    # Extraction metadata
    tags: Optional[List[str]] = Field(
        default=None,
        description="Metadata tags for categorization and filtering",
    )
    extraction_model: Optional[str] = Field(
        default=None,
        description="LLM model used for extraction (e.g., 'claude-sonnet-4-20250514')",
    )
    extraction_reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level used during extraction (e.g., 'high')",
    )

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _all_source_relevances(self) -> list[tuple[str, "SourceRelevanceAssessment"]]:
        """Return (source_tag, source_relevance) for primary + all secondary sources."""
        result = [(self.primary_data_source.source_tag, self.primary_data_source.source_relevance)]
        if self.secondary_data_sources:
            for s in self.secondary_data_sources:
                result.append((s.source_tag, s.source_relevance))
        return result

    @property
    def source_relevance_map(self) -> dict[str, "SourceRelevanceAssessment"]:
        """Map source_tag -> SourceRelevanceAssessment for all sources."""
        return dict(self._all_source_relevances())

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

        # Check uses_inputs and sample_size_input in measurements
        for measurement in self.calibration.measurements:
            if measurement.uses_inputs:
                for input_name in measurement.uses_inputs:
                    if input_name not in input_names:
                        errors.append(
                            f"Measurement '{measurement.name}' references unknown input '{input_name}'"
                        )
            if measurement.sample_size_input and measurement.sample_size_input not in input_names:
                errors.append(
                    f"Measurement '{measurement.name}' sample_size_input references "
                    f"unknown input '{measurement.sample_size_input}'"
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

        # Check InputRef in model parameter_roles (all model types)
        model = self.calibration.model
        for field_name, value in _get_parameter_role_fields(model).items():
            if isinstance(value, InputRef) and value.input_ref not in input_names:
                errors.append(f"Model {field_name} references unknown input '{value.input_ref}'")

        if errors:
            from maple.core.calibration.exceptions import InputReferenceError

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
            from maple.core.calibration.exceptions import SourceRefError

            errors.append(f"Available source tags: {sorted(source_tags)}")
            raise SourceRefError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_parameter_roles(self) -> "SubmodelTarget":
        """
        Validate that parameter role strings reference existing parameters.

        When a parameter_role is a string (not InputRef/ReferenceRef) and not a
        numeric literal, it should match a parameter name in calibration.parameters.
        """
        param_names = {p.name for p in self.calibration.parameters}
        errors = []

        model = self.calibration.model
        for field_name, value in _get_parameter_role_fields(model).items():
            if isinstance(value, str):
                # Skip numeric literal strings (e.g., "1.0", "1440.0", "2.5e+8")
                try:
                    float(value)
                    continue
                except ValueError:
                    pass
                if value not in param_names:
                    errors.append(f"Model {field_name}='{value}' is not in calibration.parameters")

        if errors:
            from maple.core.calibration.exceptions import ParameterReferenceError

            errors.append(f"Available parameters: {sorted(param_names)}")
            raise ParameterReferenceError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_reference_refs(self, info: ValidationInfo) -> "SubmodelTarget":
        """
        Validate that ReferenceRef values exist in the reference database.

        The reference database is passed via Pydantic validation context.
        If no context is provided (e.g., bare instantiation), skip silently.
        """
        reference_db = None
        if info.context and "reference_db" in info.context:
            reference_db = info.context["reference_db"]

        if reference_db is None:
            return self  # No reference database available, skip check

        model = self.calibration.model
        errors = []
        for field_name, value in _get_parameter_role_fields(model).items():
            if isinstance(value, ReferenceRef):
                if value.reference_ref not in reference_db:
                    errors.append(
                        f"Model {field_name} references unknown reference_ref "
                        f"'{value.reference_ref}'"
                    )

        if errors:
            from maple.core.calibration.exceptions import ReferenceRefError

            available = sorted(reference_db.keys()) if reference_db else []
            errors.append(f"Available reference values: {available}")
            raise ReferenceRefError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_structured_model_roles_resolvable(self, info: ValidationInfo) -> "SubmodelTarget":
        """
        Validate that all ParameterRole fields in structured algebraic models
        resolve to something concrete: a defined parameter, a valid InputRef,
        a ReferenceRef, or a numeric literal.

        Catches typos like source_pool: "circulating_cd8" when the input is
        named "circulating_cd8_count".
        """
        _STRUCTURED_ALGEBRAIC_TYPES = {
            "steady_state_density",
            "steady_state_fraction",
            "steady_state_concentration",
            "steady_state_ratio",
            "steady_state_proliferation_index",
            "batch_accumulation",
            "direct_fit",
            "power_law",
        }

        model = self.calibration.model
        if model.type not in _STRUCTURED_ALGEBRAIC_TYPES:
            return self

        param_names = {p.name for p in self.calibration.parameters}
        input_names = {inp.name for inp in self.inputs}

        errors = []
        for field_name, value in _get_parameter_role_fields(model).items():
            if isinstance(value, InputRef):
                # Already validated by validate_input_refs
                continue
            elif isinstance(value, ReferenceRef):
                # Already validated by validate_reference_refs
                continue
            elif isinstance(value, str):
                # Check if it's a numeric literal
                try:
                    float(value)
                    continue
                except ValueError:
                    pass
                # Must be a parameter name
                if value not in param_names:
                    errors.append(
                        f"Model {field_name}='{value}' is not a parameter, "
                        f"numeric literal, InputRef, or ReferenceRef"
                    )

        if errors:
            from maple.core.calibration.exceptions import ParameterReferenceError

            errors.append(f"Available parameters: {sorted(param_names)}")
            errors.append(f"Available inputs: {sorted(input_names)}")
            raise ParameterReferenceError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def warn_unit_conversion_factor_sanity(self) -> "SubmodelTarget":
        """
        Warn if unit_conversion_factor has a suspicious value.

        Only checks numeric literal UCFs — skips parameter names, InputRef,
        and ReferenceRef since those are resolved at runtime.
        """
        model = self.calibration.model
        if not hasattr(model, "unit_conversion_factor"):
            return self

        ucf_value = model.unit_conversion_factor

        # Skip non-literal UCFs (parameter references, InputRef, ReferenceRef)
        if isinstance(ucf_value, (InputRef, ReferenceRef)):
            return self
        if not isinstance(ucf_value, str):
            return self

        try:
            ucf_float = float(ucf_value)
        except ValueError:
            return self  # Non-numeric string — handled by validate_parameter_roles

        if ucf_float == 1.0:
            return self  # Default, no warning needed

        # Check for extreme values
        if ucf_float > 1e6 or (ucf_float > 0 and ucf_float < 1e-6):
            warnings.warn(
                f"unit_conversion_factor={ucf_value} is extreme (>1e6 or <1e-6). "
                f"Verify this is the correct time/unit conversion.",
                UserWarning,
                stacklevel=2,
            )

        # Check for "round" conversion factors — common time conversions
        # and simple powers of 10
        _COMMON_UCF = {
            24.0,
            60.0,
            1440.0,
            3600.0,
            86400.0,
            365.0,
            365.25,
            8760.0,
            525600.0,  # hours/year, minutes/year
            1000.0,
            1e6,
            1e9,
            1e-3,
            1e-6,
            1e-9,
            # Common products
            24.0 * 365.25,  # hours/year
            1440.0 / 24.0,  # = 60 (minutes/hour, already included)
        }

        is_common = ucf_float in _COMMON_UCF
        # Also accept powers of 10
        if not is_common and ucf_float > 0:
            import math

            log10 = math.log10(ucf_float)
            is_common = abs(log10 - round(log10)) < 1e-9

        if not is_common:
            warnings.warn(
                f"unit_conversion_factor={ucf_value} is not a common time/unit "
                f"conversion factor. Common values: 60 (min/hr), 1440 (min/day), "
                f"3600 (sec/hr), 86400 (sec/day), or powers of 10.",
                UserWarning,
                stacklevel=2,
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
            from maple.core.calibration.exceptions import StateVariableReferenceError

            errors.append(f"Available state variables: {sorted(sv_names)}")
            raise StateVariableReferenceError.from_errors(errors)

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
            from maple.core.calibration.exceptions import ObservableConfigError

            raise ObservableConfigError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_algebraic_model_observable_usage(self) -> "SubmodelTarget":
        """
        Check observable usage in algebraic models.

        Algebraic models support two patterns:
        1. Single-output: compute() returns a scalar, no observable needed.
        2. Multi-output: compute() returns a vector/dict, each error model entry
           uses observable.code to select/transform the relevant output. This is
           the same pattern as ODE models and is useful for multi-condition
           experiments (e.g., control vs treatment sharing the same forward model).

        Warns if SOME entries have observables and others don't (inconsistent).
        """
        model = self.calibration.model
        if model.type != "algebraic":
            return self

        has_obs = [
            e.observable is not None and e.observable.code is not None
            for e in self.calibration.error_model
        ]
        if any(has_obs) and not all(has_obs):
            warnings.warn(
                "Algebraic model has mixed observable usage: some error model entries "
                "have observable.code and others don't. For multi-output algebraic "
                "models, all entries should have observables to select their output.",
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
                func_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

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
            from maple.core.calibration.exceptions import CodeSignatureError

            raise CodeSignatureError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_observation_bootstrap_samples(self) -> "SubmodelTarget":
        """
        Validate that observation_code bootstrap samples have positive spread.

        Checks that derive_observation() returns samples with std > 0,
        indicating the bootstrap is producing meaningful variability.
        """
        import numpy as np

        # Build inputs dict (plain floats)
        inputs_dict = {}
        for inp in self.inputs:
            inputs_dict[inp.name] = inp.value

        errors = []

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            try:
                # Execute observation_code
                local_scope = {"np": np, "numpy": np}
                exec(entry.observation_code, local_scope)
                derive_observation = local_scope.get("derive_observation")

                if derive_observation is None:
                    continue

                sample_size = int(inputs_dict.get(entry.sample_size_input, 1))
                rng = np.random.default_rng(42)
                result = derive_observation(inputs_dict, sample_size, rng, entry.n_bootstrap)

                if not isinstance(result, np.ndarray):
                    continue  # Type errors handled by validate_observation_code_execution

                sd = np.std(result)
                if sd <= 0:
                    errors.append(
                        f"Error model '{entry.name}': bootstrap samples have zero spread "
                        f"(std=0). The parametric bootstrap must produce variability."
                    )

            except Exception:
                # Execution errors handled by other validators
                pass

        if errors:
            from maple.core.calibration.exceptions import UnitValidationError

            raise UnitValidationError.from_errors(errors, prefix="Bootstrap sample validation")

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
            from maple.core.calibration.exceptions import EvaluationPointsError

            errors.append(
                f"Adjust evaluation_points to be within [{t_start}, {t_end}] "
                f"or extend independent_variable.span."
            )
            raise EvaluationPointsError.from_errors(
                errors, prefix="Evaluation points outside integration span"
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
            from maple.core.calibration.exceptions import MissingFieldError

            raise MissingFieldError.from_errors(
                errors, prefix="Missing required fields for ODE model"
            )

        return self

    @model_validator(mode="after")
    def validate_custom_code_syntax(self) -> "SubmodelTarget":
        """
        Validate syntax and function signature for custom code blocks.

        Checks:
        - CustomODEModel.code has 'def ode(t, y, params, inputs)'
        - AlgebraicModel.code has 'def compute(params, inputs)'
        - Custom observable.code has 'def compute(t, y, y_start)'
        - observation_code has 'def derive_observation(inputs, sample_size)'
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
                        # Check argument count: should be (inputs, sample_size, rng, n_bootstrap)
                        n_args = len(func_defs[0].args.args)
                        if n_args != 4:
                            errors.append(
                                f"Error model '{measurement.name}' observation_code function "
                                f"'derive_observation' must have 4 arguments "
                                f"(inputs, sample_size, rng, n_bootstrap), got {n_args}"
                            )
                except SyntaxError as e:
                    errors.append(
                        f"Error model '{measurement.name}' observation_code syntax error: {e}"
                    )

        if errors:
            from maple.core.calibration.exceptions import CodeSyntaxError

            raise CodeSyntaxError("code validation", "\n  - ".join(errors))

        return self

    @model_validator(mode="after")
    def validate_observation_code_execution(self) -> "SubmodelTarget":
        """
        Execute observation_code and validate return value structure.

        The observation_code generates parametric bootstrap samples from the
        literature data. The framework derives point estimates (median, SD, CI95)
        from the returned sample array.

        Expected signature: def derive_observation(inputs, sample_size, rng, n_bootstrap) -> np.ndarray
        Expected return: 1D numpy array of bootstrap samples (length >= 100)
        """
        import numpy as np

        errors = []

        # Build inputs dict (plain floats)
        inputs_dict = {}
        for inp in self.inputs:
            inputs_dict[inp.name] = inp.value

        # Framework-controlled RNG for reproducibility
        rng = np.random.default_rng(42)

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            try:
                # Compile and execute
                local_scope = {"np": np, "numpy": np}
                exec(entry.observation_code, local_scope)
                derive_observation = local_scope.get("derive_observation")

                if derive_observation is None:
                    errors.append(
                        f"Error model '{entry.name}': observation_code "
                        f"did not define 'derive_observation' function"
                    )
                    continue

                # Execute with inputs, sample_size, and rng
                sample_size = int(inputs_dict.get(entry.sample_size_input, 1))
                result = derive_observation(inputs_dict, sample_size, rng, entry.n_bootstrap)

                # Validate return type is ndarray
                if not isinstance(result, np.ndarray):
                    errors.append(
                        f"Error model '{entry.name}': derive_observation must return "
                        f"np.ndarray, got {type(result).__name__}"
                    )
                    continue

                # Validate 1D
                if result.ndim != 1:
                    errors.append(
                        f"Error model '{entry.name}': derive_observation must return "
                        f"1D array, got {result.ndim}D with shape {result.shape}"
                    )
                    continue

                # Validate non-empty
                if len(result) < 100:
                    errors.append(
                        f"Error model '{entry.name}': bootstrap samples too few "
                        f"({len(result)}), need at least 100 for stable estimates"
                    )

                # Validate no NaN/Inf
                if np.any(~np.isfinite(result)):
                    n_bad = np.sum(~np.isfinite(result))
                    errors.append(
                        f"Error model '{entry.name}': bootstrap samples contain "
                        f"{n_bad} non-finite values (NaN or Inf)"
                    )

                # Validate positive SD (non-degenerate)
                if np.std(result) == 0:
                    errors.append(
                        f"Error model '{entry.name}': bootstrap samples have zero "
                        f"variance (all identical values)"
                    )

            except Exception as e:
                errors.append(f"Error model '{entry.name}': observation_code execution error: {e}")

        if errors:
            from maple.core.calibration.exceptions import CodeExecutionError

            raise CodeExecutionError("observation_code", "\n  - ".join(errors))

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
            from maple.core.calibration.exceptions import ParameterReferenceError

            raise ParameterReferenceError(
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
    def warn_observation_cv_unreasonable(self) -> "SubmodelTarget":
        """
        Warn when bootstrap sample CV is unreasonably large or small.

        Computes coefficient of variation (SD/mean) from bootstrap samples.
        Warns if CV > 100 or CV < 0.001, which often indicates a units
        mismatch, typo, or degenerate bootstrap.
        """
        import numpy as np

        # Build inputs dict (plain floats)
        inputs_dict = {}
        for inp in self.inputs:
            inputs_dict[inp.name] = inp.value

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            try:
                # Execute observation_code
                local_scope = {"np": np, "numpy": np}
                exec(entry.observation_code, local_scope)
                derive_observation = local_scope.get("derive_observation")
                if derive_observation is None:
                    continue

                sample_size = int(inputs_dict.get(entry.sample_size_input, 1))
                rng = np.random.default_rng(42)
                result = derive_observation(inputs_dict, sample_size, rng, entry.n_bootstrap)

                if not isinstance(result, np.ndarray):
                    continue

                mean_val = np.mean(result)
                sd_val = np.std(result)

                if mean_val == 0 or sd_val == 0:
                    continue

                cv = sd_val / abs(mean_val)

                if cv > 100:
                    warnings.warn(
                        f"Error model '{entry.name}': bootstrap CV ({cv:.1f}) is very large. "
                        f"Mean={mean_val:.2e}, SD={sd_val:.2e}.\n"
                        f"This may indicate a units mismatch. Check observation_code.",
                        UserWarning,
                    )
                elif cv < 0.001:
                    warnings.warn(
                        f"Error model '{entry.name}': bootstrap CV ({cv:.1e}) is very small. "
                        f"Mean={mean_val:.2e}, SD={sd_val:.2e}.\n"
                        f"This may indicate overly confident bootstrap parameters.",
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
        if t_start < 0 or t_end <= t_start:
            from maple.core.calibration.exceptions import SpanOrderingError

            raise SpanOrderingError(iv.span)

        return self

    @model_validator(mode="after")
    def validate_input_values_in_snippets(self) -> "SubmodelTarget":
        """
        Validate that extracted values appear in their value_snippet or table_excerpt.

        Catches hallucinations where the LLM extracts a value that doesn't
        appear in the cited text. Skips unit_conversion and reference_value
        inputs (these don't come from paper text).

        Inputs with figure_excerpt are accepted as valid provenance but
        emit a warning for manual review, since figure-derived values
        cannot be validated by text matching.
        """
        import warnings

        from maple.core.calibration.validators import check_value_in_text

        errors = []
        for inp in self.inputs:
            # Skip types that don't come from paper text
            if inp.input_type in (
                InputType.UNIT_CONVERSION,
                InputType.REFERENCE_VALUE,
                InputType.DERIVED_ARITHMETIC,
            ):
                continue

            has_snippet = bool(inp.value_snippet)
            has_table = bool(inp.table_excerpt)
            has_figure = bool(inp.figure_excerpt)

            # Require at least one provenance source for measurable inputs
            if not has_snippet and not has_table and not has_figure:
                errors.append(
                    f"Input '{inp.name}': no provenance provided (value_snippet, "
                    f"table_excerpt, or figure_excerpt). "
                    f"At least one is required for {inp.input_type.value} inputs."
                )
                continue

            # Figure-derived values: accept as provenance but warn for manual review
            if has_figure and not has_snippet and not has_table:
                warnings.warn(
                    f"Input '{inp.name}': value {inp.value} is read from "
                    f"{inp.figure_excerpt.figure_id} — requires manual review "
                    f"(figure-derived values cannot be text-validated).",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Check table_excerpt if present
            if has_table:
                if not check_value_in_text(inp.table_excerpt.value, inp.value):
                    errors.append(
                        f"Input '{inp.name}': value {inp.value} not found in "
                        f"table_excerpt.value '{inp.table_excerpt.value}'"
                    )

            # Check value_snippet if present
            if has_snippet:
                if not check_value_in_text(inp.value_snippet, inp.value):
                    errors.append(
                        f"Input '{inp.name}': value {inp.value} not found in snippet "
                        f"'{inp.value_snippet[:80]}{'...' if len(inp.value_snippet) > 80 else ''}'"
                    )

        if errors:
            from maple.core.calibration.exceptions import SnippetValueMismatchError

            raise SnippetValueMismatchError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_non_measurement_inputs_have_rationale(self) -> "SubmodelTarget":
        """
        Require rationale for unit_conversion, reference_value, and derived_arithmetic inputs.

        These input types don't come from paper text, so they need an explicit
        rationale explaining why this value was chosen or how it was derived.
        """
        errors = []
        for inp in self.inputs:
            if inp.input_type in (
                InputType.UNIT_CONVERSION,
                InputType.REFERENCE_VALUE,
                InputType.DERIVED_ARITHMETIC,
            ):
                if not inp.rationale:
                    errors.append(
                        f"Input '{inp.name}' has input_type='{inp.input_type.value}' "
                        f"but no rationale provided. Non-measurement inputs require "
                        f"a rationale explaining why this value was chosen."
                    )

        if errors:
            raise ValueError("\n".join(errors))

        return self

    @model_validator(mode="after")
    def validate_derived_arithmetic_inputs(self) -> "SubmodelTarget":
        """
        Validate derived_arithmetic inputs: require formula + source_inputs,
        check that source_inputs reference existing inputs, and verify that
        evaluating the formula produces the declared value.

        Also rejects formula/source_inputs on non-derived_arithmetic inputs.
        """
        import math
        import re

        input_map = {inp.name: inp.value for inp in self.inputs}
        errors = []

        for inp in self.inputs:
            if inp.input_type == InputType.DERIVED_ARITHMETIC:
                if not inp.formula:
                    errors.append(
                        f"Input '{inp.name}' has input_type='derived_arithmetic' "
                        f"but no formula provided."
                    )
                if not inp.source_inputs:
                    errors.append(
                        f"Input '{inp.name}' has input_type='derived_arithmetic' "
                        f"but no source_inputs provided."
                    )
                if not inp.formula or not inp.source_inputs:
                    continue

                # Check all source_inputs exist
                missing = [s for s in inp.source_inputs if s not in input_map]
                if missing:
                    errors.append(
                        f"Input '{inp.name}': source_inputs {missing} " f"not found in inputs."
                    )
                    continue

                # Evaluate formula and check against declared value
                # Build a safe namespace with source input values
                namespace = {s: input_map[s] for s in inp.source_inputs}
                # Allow basic math functions
                safe_builtins = {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "log": math.log,
                    "log2": math.log2,
                    "log10": math.log10,
                    "sqrt": math.sqrt,
                    "exp": math.exp,
                    "pi": math.pi,
                }
                namespace.update(safe_builtins)

                try:
                    # Validate the formula only uses known names
                    formula_names = set(re.findall(r"\b([a-zA-Z_]\w*)\b", inp.formula))
                    allowed_names = set(inp.source_inputs) | set(safe_builtins.keys())
                    unknown = formula_names - allowed_names
                    if unknown:
                        errors.append(
                            f"Input '{inp.name}': formula references unknown "
                            f"names {unknown}. Only source_inputs and math "
                            f"functions (abs, min, max, log, sqrt, exp) are allowed."
                        )
                        continue

                    computed = eval(inp.formula, {"__builtins__": {}}, namespace)  # noqa: S307
                    # Check with relative tolerance (1% for floating point)
                    if inp.value == 0:
                        if abs(computed) > 1e-10:
                            errors.append(
                                f"Input '{inp.name}': formula '{inp.formula}' "
                                f"evaluates to {computed}, but declared value is 0."
                            )
                    elif abs(computed - inp.value) / abs(inp.value) > 0.01:
                        errors.append(
                            f"Input '{inp.name}': formula '{inp.formula}' "
                            f"evaluates to {computed}, but declared value "
                            f"is {inp.value} (>{1}% mismatch)."
                        )
                except Exception as e:
                    errors.append(
                        f"Input '{inp.name}': formula '{inp.formula}' " f"failed to evaluate: {e}"
                    )
            else:
                # Non-derived_arithmetic inputs must not have formula/source_inputs
                if inp.formula is not None:
                    errors.append(
                        f"Input '{inp.name}' has input_type='{inp.input_type.value}' "
                        f"but has a formula field. formula is only valid for "
                        f"derived_arithmetic inputs."
                    )
                if inp.source_inputs is not None:
                    errors.append(
                        f"Input '{inp.name}' has input_type='{inp.input_type.value}' "
                        f"but has a source_inputs field. source_inputs is only valid "
                        f"for derived_arithmetic inputs."
                    )

        if errors:
            raise ValueError("\n".join(errors))

        return self

    @model_validator(mode="after")
    def validate_no_assumed_or_uncertainty_inputs(self) -> "SubmodelTarget":
        """
        Reject inputs that look like modeling assumptions rather than data.

        Two checks:
        1. Any input with 'assumed' in the name is rejected — naming something
           'assumed_*' signals it's a modeling choice, not paper data.
        2. Uncertainty-smelling names (cv, sigma, etc.) on unit_conversion or
           reference_value inputs are rejected — these don't belong in the
           data model at all.
        """
        # Patterns that suggest uncertainty factors (only checked on non-measurement types)
        UNCERTAINTY_PATTERNS = [
            "cv",
            "sigma",
            "uncertainty",
            "fold_uncertainty",
            "translation_sd",
            "translation_uncertainty",
        ]

        errors = []
        for inp in self.inputs:
            name_lower = inp.name.lower()

            # "assumed" in any input name is a red flag regardless of type
            if "assumed" in name_lower:
                errors.append(
                    f"Input '{inp.name}' contains 'assumed' in its name, "
                    f"indicating a modeling choice rather than extracted data. "
                    f"Modeling assumptions (uncertainty factors, CVs, fractions) "
                    f"should not be stored as inputs. Derive them in "
                    f"observation_code or handle downstream in inference."
                )
                continue

            # Uncertainty patterns on non-measurement types
            if inp.input_type in (
                InputType.UNIT_CONVERSION,
                InputType.REFERENCE_VALUE,
                InputType.DERIVED_ARITHMETIC,
            ):
                for pattern in UNCERTAINTY_PATTERNS:
                    if pattern in name_lower:
                        errors.append(
                            f"Input '{inp.name}' looks like an uncertainty factor "
                            f"(matches pattern '{pattern}') but has "
                            f"input_type='{inp.input_type.value}'. "
                            f"Uncertainty factors should not be stored as inputs."
                        )
                        break

        if errors:
            raise ValueError("\n".join(errors))

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
        from maple.core.calibration.validators import (
            resolve_doi,
            fuzzy_match,
        )

        errors = []

        def _extract_family_name(full_name: str) -> str:
            """Extract family name for CrossRef comparison.

            Expects family name only (e.g., 'Zhang', 'Wilson', 'den Braber').
            Also handles legacy formats: 'Family, Given' or 'Given Family'.
            """
            full_name = full_name.strip()
            if "," in full_name:
                return full_name.split(",")[0].strip()
            # If it's a single word, return as-is (expected case)
            parts = full_name.split()
            if len(parts) == 1:
                return parts[0]
            # Multi-word: if last token is a short initial, it's "Family I" format
            last = parts[-1].rstrip(".")
            if len(last) <= 2 and last.isupper() and len(parts) > 1:
                return " ".join(parts[:-1])
            # Otherwise assume "Given Family" format
            return parts[-1]

        def check_doi_source(
            doi: str,
            source_tag: str,
            title: Optional[str],
            year: Optional[int],
            authors: Optional[List[str]],
            prefix: str,
        ):
            """Validate a DOI and check metadata matches."""
            metadata = resolve_doi(doi)
            if metadata is None:
                from maple.core.calibration.exceptions import DOIResolutionError

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

            # Check first author match
            crossref_author = metadata.get("first_author")
            if crossref_author and authors and len(authors) > 0:
                recorded_family = _extract_family_name(authors[0])
                if not fuzzy_match(recorded_family, crossref_author, threshold=0.8):
                    errors.append(
                        f"{prefix} first author mismatch: "
                        f"recorded '{authors[0]}', CrossRef says '{crossref_author}'"
                    )

            # Check year match
            if year and metadata.get("year"):
                if year != metadata["year"]:
                    errors.append(
                        f"{prefix} year mismatch: "
                        f"recorded {year}, CrossRef says {metadata['year']}"
                    )

        # Check primary source (DOI validated if present; PMID-only sources skip DOI check)
        if self.primary_data_source.doi:
            check_doi_source(
                self.primary_data_source.doi,
                self.primary_data_source.source_tag,
                self.primary_data_source.title,
                self.primary_data_source.year,
                self.primary_data_source.authors,
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
                        source.authors,
                        f"Secondary source [{i}]",
                    )
                # URL-only sources don't get DOI validation

        if errors:
            from maple.core.calibration.exceptions import DOIMetadataMismatchError

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
        from maple.core.unit_registry import ureg

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
            from maple.core.calibration.exceptions import UnitParsingError

            raise UnitParsingError.from_errors(errors)

        return self

    @model_validator(mode="after")
    def validate_non_nuisance_params_in_model(self, info: ValidationInfo) -> "SubmodelTarget":
        """Validate that non-nuisance calibration parameters exist in the QSP model.

        Parameters marked nuisance=False (or nuisance not set) are QSP model
        parameters that must exist in model_structure. Parameters not found are
        likely invented by the extraction agent and should be marked nuisance=True
        with an inline prior, or removed.

        Requires context:
            model_structure: ModelStructure instance
        """
        if not info.context or "model_structure" not in info.context:
            return self

        model_structure = info.context["model_structure"]
        model_params = {p.name for p in model_structure.parameters}

        missing = []
        for param in self.calibration.parameters:
            if not param.nuisance and param.name not in model_params:
                missing.append(param.name)

        if missing:
            raise ValueError(
                f"Non-nuisance parameter(s) not found in QSP model: {missing}. "
                f"These must either exist in model_structure.json as real QSP parameters, "
                f"or be marked nuisance=True with an inline prior."
            )

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

        from maple.core.unit_registry import ureg

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
                    raise DimensionalityMismatchError(
                        f"Parameter '{param.name}' unit dimensionality mismatch:\n"
                        f"  SubmodelTarget units: '{param.units}' "
                        f"(dimensionality: {actual.dimensionality})\n"
                        f"  Model expected units: '{model_param.units}' "
                        f"(dimensionality: {expected.dimensionality})\n"
                        f"Common issue: using concentration (e.g., nanomolar) instead of "
                        f"amount (e.g., nanomole) for rate parameters."
                    )
            except DimensionalityMismatchError:
                raise
            except Exception:
                # Unit parsing issue - handled by validate_units_are_valid_pint
                pass

        return self

    # NOTE: validate_algebraic_model_output_units was removed — its coverage is
    # redundant with validate_observation_code_execution, and it depended on
    # param.prior (removed from schema).

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
        from maple.core.unit_registry import ureg

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
        - Time conversions like 24.0, 60.0 (add as unit_conversion input)
        - Percentage conversions like 100.0 (add as unit_conversion input)
        - Reference values like tissue density (add as reference_value input)
        - Any other numeric literal (add as direct_measurement with snippet)
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
                    f"  Add these as inputs with input_type='unit_conversion' (for conversion factors)\n"
                    f"  or 'reference_value' (for normalization constants), with a rationale.\n"
                    f"  For measured values, use input_type='direct_measurement' with value_snippet."
                )

        if errors:
            from maple.core.calibration.exceptions import HardcodedConstantError

            raise HardcodedConstantError.from_errors(errors)

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
    def validate_observation_code_returns_array_not_dict(self) -> "SubmodelTarget":
        """
        Warn if observation_code appears to return a dict (old format).

        The new format requires returning a 1D numpy array of bootstrap samples.
        Returning a dict with 'value'/'sd' keys is the old format and will fail
        at runtime. This static check catches it early.
        """
        import ast

        for entry in self.calibration.error_model:
            if not entry.observation_code:
                continue

            code = entry.observation_code

            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "derive_observation":
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and child.value:
                            if isinstance(child.value, ast.Dict):
                                from maple.core.calibration.exceptions import (
                                    ReturnStructureError,
                                )

                                raise ReturnStructureError(
                                    f"Error model '{entry.name}' observation_code "
                                    f"returns a dict (old format).\n\n"
                                    f"observation_code must return a 1D numpy array of "
                                    f"parametric bootstrap samples.\n"
                                    f"Example:\n"
                                    f"  return rng.normal(loc=mean, scale=sd, size=n_bootstrap)"
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
            from maple.core.calibration.exceptions import ControlCharacterError

            raise ControlCharacterError.from_errors(errors[:10])

        return self

    # -------------------------------------------------------------------------
    # SOURCE RELEVANCE VALIDATORS
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_source_quality_peer_reviewed(self) -> "SubmodelTarget":
        """Warn about non-peer-reviewed sources (Wikipedia, preprints)."""
        for tag, sr in self._all_source_relevances():
            if sr.source_quality == SourceQuality.NON_PEER_REVIEWED:
                # Check if this source is referenced in any input
                used_for_values = any(inp.source_ref == tag for inp in self.inputs)
                severity = "AND provides values used in calibration" if used_for_values else ""
                warnings.warn(
                    f"Source '{tag}' quality is 'non_peer_reviewed' {severity} "
                    f"for target '{self.target_id}'.\n"
                    "This includes Wikipedia, preprints, and unreviewed sources.\n\n"
                    "Recommended actions:\n"
                    "  1. Replace with peer-reviewed source if possible\n"
                    "  2. Document rationale in key_study_limitations\n"
                    "  3. Translation sigma inflation will be applied automatically",
                    UserWarning,
                )
        return self

    @model_validator(mode="after")
    def warn_cross_indication_extrapolation(self) -> "SubmodelTarget":
        """Warn about cross-indication extrapolation per source.

        Translation uncertainty is computed deterministically downstream
        from source_relevance fields, not specified manually.
        """
        for tag, sr in self._all_source_relevances():
            if sr.indication_match in (IndicationMatch.PROXY, IndicationMatch.UNRELATED):
                warnings.warn(
                    f"Cross-indication extrapolation in source '{tag}': "
                    f"indication_match='{sr.indication_match.value}'. "
                    f"Translation sigma inflation will be applied automatically.",
                    UserWarning,
                )
        return self

    @model_validator(mode="after")
    def validate_pharmacological_perturbation_justification(self) -> "SubmodelTarget":
        """Flag pharmacological perturbation without justification."""
        for tag, sr in self._all_source_relevances():
            if sr.perturbation_type == PerturbationType.PHARMACOLOGICAL:
                if not sr.perturbation_relevance:
                    from maple.core.calibration.exceptions import MissingFieldError

                    raise MissingFieldError(
                        f"Source '{tag}': perturbation type is 'pharmacological' but "
                        "perturbation_relevance is not provided.\n\n"
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
        for tag, sr in self._all_source_relevances():
            if sr.perturbation_type == PerturbationType.GENETIC:
                if not sr.perturbation_relevance:
                    from maple.core.calibration.exceptions import MissingFieldError

                    raise MissingFieldError(
                        f"Source '{tag}': perturbation type is 'genetic_perturbation' but "
                        "perturbation_relevance is not provided.\n\n"
                        "When using KO/knockdown/overexpression data to estimate physiological "
                        "parameters, you must explain:\n"
                        "  - How the genetic perturbation relates to the wild-type parameter\n"
                        "  - Whether the measurement provides bounds or direct estimates\n"
                        "  - Compensatory mechanisms that may affect interpretation"
                    )
        return self

    @model_validator(mode="after")
    def validate_low_tme_compatibility_notes(self) -> "SubmodelTarget":
        """Require documentation for low TME compatibility."""
        for tag, sr in self._all_source_relevances():
            if sr.tme_compatibility == TMECompatibility.LOW:
                if not sr.tme_compatibility_notes:
                    from maple.core.calibration.exceptions import MissingFieldError

                    raise MissingFieldError(
                        f"Source '{tag}': TME compatibility is 'low' but "
                        "tme_compatibility_notes is not provided.\n\n"
                        "Document the specific TME differences and their expected impact:\n"
                        "  - Stromal density differences\n"
                        "  - Immune infiltration patterns\n"
                        "  - Chemokine/cytokine milieu\n"
                        "  - Expected direction and magnitude of bias"
                    )
        return self

    @model_validator(mode="after")
    def warn_cross_species_extrapolation(self) -> "SubmodelTarget":
        """Warn about cross-species extrapolation per source.

        Translation uncertainty is computed deterministically downstream
        from source_relevance fields, not specified manually.
        """
        for tag, sr in self._all_source_relevances():
            if sr.species_source != sr.species_target:
                warnings.warn(
                    f"Cross-species extrapolation in source '{tag}': "
                    f"{sr.species_source} → {sr.species_target}. "
                    f"Translation sigma inflation will be applied automatically.",
                    UserWarning,
                )
        return self

    # NOTE: validate_prior_reflects_translation_uncertainty was removed —
    # priors are no longer part of the SubmodelTarget schema.

    # NOTE: validate_algebraic_prior_predictive was removed —
    # priors are no longer part of the SubmodelTarget schema.
    # Coverage is provided by validate_observation_code_execution.

    # NOTE: validate_sample_size_list_length was removed —
    # sample_size is now a single input reference (sample_size_input),
    # not an inline int/list field.

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
            from maple.core.calibration.exceptions import ObservableConfigError

            raise ObservableConfigError.from_errors(errors)

        # ODE observables require state_variables (defines y vector ordering)
        sv_errors = []
        for entry in self.calibration.error_model:
            if entry.observable is not None and not entry.observable.state_variables:
                sv_errors.append(
                    f"Error model '{entry.name}': ODE model type '{model_type}' "
                    f"requires observable.state_variables to define y vector ordering."
                )
        if sv_errors:
            from maple.core.calibration.exceptions import ObservableConfigError

            raise ObservableConfigError.from_errors(sv_errors)

        return self

    @model_validator(mode="after")
    def validate_snippets_against_pdfs(self, info: ValidationInfo) -> "SubmodelTarget":
        """Validate value_snippets and table_excerpts against source PDFs.

        Requires context:
            papers_dir: Path to directory containing source PDFs

        When papers_dir is provided, each direct_measurement input's snippet
        is fuzzy-matched against the extracted PDF text.  Failures raise
        ValueError so the extraction retry loop can re-attempt.

        Skipped silently when papers_dir is not in context (e.g., when loading
        existing YAMLs outside the extraction pipeline).
        """
        if not info.context or "papers_dir" not in info.context:
            return self

        papers_dir = info.context["papers_dir"]
        # validate_snippets_in_file expects a YAML path, but we already have
        # the parsed object.  Use the in-memory validation path instead.
        from maple.core.calibration.snippet_validator import (
            load_paper_texts,
        )
        from maple.core.calibration.validators import fuzzy_find_snippet_in_text

        # Collect source tags and metadata
        source_tags = {self.primary_data_source.source_tag}
        source_metadata = {
            self.primary_data_source.source_tag: {
                "doi": self.primary_data_source.doi,
                "url": None,
            }
        }
        if self.secondary_data_sources:
            for src in self.secondary_data_sources:
                source_tags.add(src.source_tag)
                source_metadata[src.source_tag] = {
                    "doi": src.doi,
                    "url": getattr(src, "url", None),
                }

        paper_data = load_paper_texts(source_tags, source_metadata, papers_dir)

        errors = []
        skip_types = {"reference_value", "derived_arithmetic", "unit_conversion"}

        for inp in self.inputs:
            if inp.input_type in skip_types:
                continue
            if not inp.value_snippet and not inp.table_excerpt:
                continue

            tag = inp.source_ref
            if tag not in paper_data:
                continue

            text, source_type = paper_data[tag]

            # Check value_snippet
            if inp.value_snippet:
                found, score, _ = fuzzy_find_snippet_in_text(inp.value_snippet, text, threshold=0.7)
                if not found:
                    errors.append(
                        f"Input '{inp.name}': value_snippet not found in "
                        f"{tag} [{source_type}] (best score: {score:.2f}). "
                        f"Snippet may be hallucinated or paraphrased — "
                        f"use verbatim text from the paper."
                    )

            # Check table_excerpt fields
            if inp.table_excerpt:
                te = inp.table_excerpt
                for field_name, threshold in [
                    ("table_id", 0.7),
                    ("column", 0.7),
                    ("value", 0.7),
                    ("row", 0.6),
                ]:
                    field_val = getattr(te, field_name, None)
                    if not field_val:
                        continue
                    found, score, _ = fuzzy_find_snippet_in_text(
                        str(field_val), text, threshold=threshold
                    )
                    if not found:
                        errors.append(
                            f"Input '{inp.name}': table_excerpt.{field_name}="
                            f"'{field_val}' not found in {tag} [{source_type}] "
                            f"(best score: {score:.2f})."
                        )

        if errors:
            from maple.core.calibration.exceptions import SnippetNotInSourceError

            raise SnippetNotInSourceError.from_errors(errors)

        return self


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "InputType",
    "ObservableType",
    "CurveType",
    # Source relevance enums
    "IndicationMatch",
    "SourceQuality",
    "PerturbationType",
    "TMECompatibility",
    # Input models
    "Input",
    "TableExcerpt",
    "FigureExcerpt",
    # Calibration models
    "Parameter",
    "FixedInitialCondition",
    "InputRefInitialCondition",
    "StateVariable",
    "InputRef",
    "ReferenceRef",
    "ParameterRole",
    # Model types
    "BaseForwardModelSpec",
    "FirstOrderDecayModel",
    "ExponentialGrowthModel",
    "LogisticModel",
    "MichaelisMentenModel",
    "TwoStateModel",
    "SaturationModel",
    "AlgebraicModel",
    "DirectFitModel",
    "PowerLawModel",
    "CustomODEModel",
    "Model",
    # Other calibration models
    "IndependentVariable",
    "Observable",
    "Measurement",
    "Calibration",
    # Context models
    "PrimaryDataSource",
    "SecondaryDataSource",
    "SubmodelCellLine",
    "CellType",
    "SubmodelCultureConditions",
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
