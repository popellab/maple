#!/usr/bin/env python3
"""
Shared Pydantic models used across different workflows.

These models are defined here to avoid circular imports between
calibration_target_models.py and pydantic_models.py.
"""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    """Classification of literature input data type for calibration targets."""

    DIRECT_PARAMETER = "direct_parameter"
    """Literature reports the model parameter value directly (e.g., 'k = 3/day')."""

    PROXY_MEASUREMENT = "proxy_measurement"
    """Literature reports a proxy that requires conversion (e.g., 'doubling time = 8h' → k = ln(2)/t)."""

    EXPERIMENTAL_CONDITION = "experimental_condition"
    """Protocol/experimental choice from paper (e.g., seeding density, E:T ratio)."""


class LiteratureInput(BaseModel):
    """
    A literature-extracted input value for calibration target derivation.

    Used in CalibrationTargetEstimates.inputs. All provenance fields are REQUIRED
    to ensure full traceability. source_ref must point to an actual literature source.

    Supports both scalar and vector-valued inputs:
    - Scalar: value=42.0 (single measurement, or constant applied to all index points)
    - Vector: value=[10.0, 20.0, 30.0] (measurements at each index point)

    For modeling assumptions (e.g., n_mc_samples), use ModelingAssumption instead.
    """

    name: str = Field(description="Input name (used as key in inputs dict)")
    value: Union[float, List[float]] = Field(
        description=(
            "Input value(s). Use a list for vector-valued data (e.g., measurements at "
            "multiple time points or doses). Scalar values are broadcast across all index points."
        )
    )
    units: str = Field(
        description="Input units (must be Pint-parseable, e.g., 'pg/mL', 'cell/mm^2', 'dimensionless')"
    )
    description: str = Field(description="What this input represents and how it was extracted")
    source_ref: str = Field(
        description=(
            "Source reference tag. MUST match a source_tag in primary_data_source or "
            "secondary_data_sources."
        )
    )
    value_location: str = Field(
        description="Where the value appears in the source (e.g., 'Table 2', 'Figure 3A', 'Results p.5')"
    )
    value_snippet: str = Field(
        description="Exact text snippet from the source containing or supporting the value(s)"
    )
    initializes_state: Optional[str] = Field(
        None,
        description=(
            "DEPRECATED: Use SubmodelStateVariable.initial_value_input instead.\n"
            "For IsolatedSystemTarget: the state variable this input provides "
            "the initial condition for (e.g., 'T_cells'). Must match a name in state_variables."
        ),
    )

    input_type: InputType = Field(
        default=InputType.DIRECT_PARAMETER,
        description=(
            "Classification of this input:\n"
            "- direct_parameter: Literature reports the model parameter directly (e.g., 'k = 3/day')\n"
            "- proxy_measurement: Requires conversion (e.g., 'doubling time = 8h' → rate constant)\n"
            "- experimental_condition: Protocol choice from paper (e.g., seeding density, E:T ratio)"
        ),
    )

    conversion_formula: Optional[str] = Field(
        None,
        description=(
            "For proxy_measurement type: formula showing how to convert to model parameter.\n"
            "Example: 'k_pro = ln(2) / doubling_time'\n"
            "Example: 'k_death = ln(2) / half_life'\n"
            "Not required for direct_parameter or experimental_condition types."
        ),
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued inputs are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued input cannot be an empty list")
        return v

    @field_validator("conversion_formula")
    @classmethod
    def require_conversion_for_proxy(cls, v: Optional[str], info) -> Optional[str]:
        """Warn if proxy_measurement type lacks conversion_formula."""
        import warnings

        # Access input_type from the data being validated
        input_type = info.data.get("input_type", InputType.DIRECT_PARAMETER)

        if input_type == InputType.PROXY_MEASUREMENT and v is None:
            warnings.warn(
                "LiteratureInput with input_type='proxy_measurement' should have conversion_formula "
                "documenting how to convert to model parameter (e.g., 'k = ln(2) / t_half').",
                UserWarning,
            )
        return v


class ModelingAssumption(BaseModel):
    """
    An assumed value for computation that is not from literature.

    Used in CalibrationTargetEstimates.assumptions for values like:
    - n_mc_samples: Number of Monte Carlo samples for bootstrap
    - assumed_cv: Assumed coefficient of variation when not reported
    - scaling_factor: Scaling factor for unit conversion

    Unlike LiteratureInput, this requires a rationale field instead of
    value_location and value_snippet.
    """

    name: str = Field(description="Assumption name (used as key in inputs dict)")
    value: Union[float, List[float]] = Field(
        description="Assumed value(s). Scalar for single values, list for per-index-point values."
    )
    units: str = Field(description="Units (must be Pint-parseable, e.g., 'dimensionless', '1/day')")
    description: str = Field(description="What this assumption represents")
    rationale: str = Field(
        description=(
            "Why this value was chosen. Required for all assumptions.\n"
            "Example: 'Standard sample size for stable percentile estimates'\n"
            "Example: 'Typical CV for biological measurements when not reported'"
        )
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued assumptions are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued assumption cannot be an empty list")
        return v


class KeyAssumption(BaseModel):
    """
    A single key assumption with its number and text.

    Note: CalibrationTarget now uses a simpler `caveats: List[str]` field.
    This class is kept for backward compatibility with ParameterMetadata and TestStatistic.
    """

    number: int = Field(description="Assumption number (1, 2, 3, ...)")
    text: str = Field(description="Assumption text")


class WeightScore(BaseModel):
    """A rubric-based weight score with justification."""

    value: float = Field(description="Rubric value (0-1)")
    justification: str = Field(description="Justification for this value")


class Source(BaseModel):
    """A bibliographic source (primary data)."""

    source_tag: str = Field(description="Unique tag for referencing")
    title: str = Field(description="Full title")
    first_author: str = Field(description="First author last name")
    year: int = Field(description="Publication year")
    doi: Optional[str] = Field(None, description="DOI (or null)")


class SecondarySource(BaseModel):
    """A secondary data source (reference values, textbooks)."""

    source_tag: str = Field(description="Unique tag for referencing")
    title: str = Field(description="Full title")
    first_author: str = Field(description="First author last name")
    year: int = Field(description="Publication year")
    doi_or_url: Optional[str] = Field(None, description="DOI or URL (or null)")


# ============================================================================
# Provenance Models
# ============================================================================


class Snippet(BaseModel):
    """A text snippet from a source paper."""

    text: str = Field(description="Exact text from the paper")
    source_tag: str = Field(
        description="Reference to source (must match a source_tag in primary_data_source or secondary_data_sources)"
    )
    figure_or_table: Optional[str] = Field(
        None, description="Figure/table reference (e.g., 'Figure 3A', 'Table 2')"
    )


class Validation(BaseModel):
    """Validation metadata (auto-populated by validation suite)."""

    tags: List[str] = Field(default_factory=list, description="Validation tags")
    validated_at: Optional[str] = Field(None, description="ISO timestamp of validation")


class MismatchDimension(str, Enum):
    """Dimensions along which experimental context can differ from model context."""

    SPECIES = "species"
    """Species mismatch (e.g., mouse vs human)."""

    SYSTEM = "system"
    """Experimental system mismatch (e.g., in vitro vs in vivo, cell line vs primary)."""

    INDICATION = "indication"
    """Disease/indication mismatch (e.g., melanoma data for PDAC model)."""

    COMPARTMENT = "compartment"
    """Anatomical compartment mismatch (e.g., blood vs tumor)."""

    ACTIVATION_STATE = "activation_state"
    """Cell activation state mismatch (e.g., activated vs exhausted T cells)."""

    TREATMENT = "treatment"
    """Treatment context mismatch (e.g., treatment-naive vs pre-treated)."""

    PROTEIN_SOURCE = "protein_source"
    """Protein source mismatch (e.g., recombinant vs endogenous)."""

    OTHER = "other"
    """Other context mismatch not covered by above categories."""


class ContextMismatch(BaseModel):
    """
    Structured documentation of context mismatch between experimental data and model.

    Used to explicitly document when experimental data context differs from the
    model context being calibrated, along with expected bias direction and any
    adjustments applied.
    """

    dimension: MismatchDimension = Field(
        description="Which dimension of context differs (species, system, etc.)"
    )

    source_context: str = Field(
        description=(
            "Context of the experimental data source.\n"
            "Example: 'Mouse splenocytes from LCMV infection model'"
        )
    )

    model_context: str = Field(
        description=(
            "Context that the model represents.\n"
            "Example: 'Human tumor-infiltrating lymphocytes in PDAC'"
        )
    )

    expected_bias: Optional[str] = Field(
        default=None,
        description=(
            "Expected direction and magnitude of bias due to this mismatch.\n"
            "Example: 'Proliferation rates likely 5-10× higher in acute infection vs chronic tumor'\n"
            "Example: 'Mouse clearance typically faster than human (allometric scaling)'"
        ),
    )

    adjustment_applied: Optional[str] = Field(
        default=None,
        description=(
            "Any scaling or adjustment applied to account for this mismatch.\n"
            "Example: 'Allometric scaling: CL_human = CL_mouse × (70/0.025)^0.75'\n"
            "Example: 'No adjustment applied; noted as caveat'"
        ),
    )


# ============================================================================
# In Vitro / Experimental System Models
# ============================================================================


class CellSpecies(str, Enum):
    """Species origin of cell line."""

    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    OTHER = "other"


class CellLine(BaseModel):
    """Specification of a cell line used in an experiment."""

    name: str = Field(
        description="Cell line name (e.g., 'Jurkat', 'HeLa', 'MCF-7', 'primary CD8 T cells')"
    )
    species: CellSpecies = Field(description="Species origin of cell line")
    cell_type: str = Field(description="Cell type (e.g., 'T cell', 'epithelial', 'fibroblast')")
    additional_info: Optional[str] = Field(
        None, description="Additional info (e.g., 'immortalized', 'primary', 'GFP-expressing')"
    )


class CultureConditions(BaseModel):
    """Culture conditions for in vitro experimental systems."""

    medium: Optional[str] = Field(None, description="Culture medium (e.g., 'RPMI-1640', 'DMEM')")
    duration_hours: Optional[float] = Field(None, description="Culture duration in hours")
    additional: Optional[dict] = Field(
        None,
        description="Additional conditions (serum, supplements, temperature, CO2, etc.)",
    )


# ============================================================================
# Multi-Point Data Models (for trajectories and dose-response)
# ============================================================================


class UncertaintyType(str, Enum):
    """Type of uncertainty measure reported in literature."""

    SD = "sd"  # Standard deviation
    SE = "se"  # Standard error
    CI95 = "ci95"  # 95% confidence interval
    RANGE = "range"  # Min-max range
    IQR = "iqr"  # Interquartile range


class TrajectoryData(BaseModel):
    """
    Time-course data with multiple measurements over time.

    Use this for kinetic experiments where the same observable is measured
    at multiple time points (e.g., proliferation curves, cytokine kinetics).
    """

    time_points: List[float] = Field(description="Time points at which measurements were taken")
    time_unit: str = Field(description="Pint-parseable unit for time points (e.g., 'hour', 'day')")
    values: List[float] = Field(
        description="Measured values at each time point (same length as time_points)"
    )
    value_unit: str = Field(
        description="Pint-parseable unit for values (e.g., 'cell', 'nanomolar')"
    )
    uncertainty: Optional[List[float]] = Field(
        None,
        description="Uncertainty at each time point (same length as time_points, or null)",
    )
    uncertainty_type: Optional[UncertaintyType] = Field(
        None, description="Type of uncertainty measure"
    )
    n_replicates: Optional[int] = Field(
        None, description="Number of replicates (if same for all time points)"
    )
    source_ref: Optional[str] = Field(None, description="Source reference tag for this data")
    figure_or_table: Optional[str] = Field(
        None, description="Figure/table reference (e.g., 'Figure 2A')"
    )


class DoseResponseData(BaseModel):
    """
    Dose-response data with measurements at multiple concentrations/doses.

    Use this for experiments varying a single parameter (concentration, E:T ratio, etc.)
    and measuring the response (e.g., killing curves, EC50 determination).
    """

    doses: List[float] = Field(description="Dose/concentration values tested")
    dose_unit: str = Field(description="Pint-parseable unit for doses (e.g., 'nanomolar', 'ng/mL')")
    dose_parameter: str = Field(
        description=(
            "What is being varied (e.g., 'IL2_concentration', 'drug_concentration', "
            "'ET_ratio', 'cell_density')"
        )
    )
    responses: List[float] = Field(
        description="Response values at each dose (same length as doses)"
    )
    response_unit: str = Field(
        description="Pint-parseable unit for responses (e.g., 'dimensionless', 'percent')"
    )
    uncertainty: Optional[List[float]] = Field(
        None,
        description="Uncertainty at each dose (same length as doses, or null)",
    )
    uncertainty_type: Optional[UncertaintyType] = Field(
        None, description="Type of uncertainty measure"
    )
    n_replicates: Optional[int] = Field(
        None, description="Number of replicates (if same for all doses)"
    )
    time_point: Optional[float] = Field(
        None, description="Time point at which dose-response was measured"
    )
    time_point_unit: Optional[str] = Field(None, description="Unit for time_point (e.g., 'hour')")
    source_ref: Optional[str] = Field(None, description="Source reference tag for this data")
    figure_or_table: Optional[str] = Field(
        None, description="Figure/table reference (e.g., 'Figure 3B')"
    )
