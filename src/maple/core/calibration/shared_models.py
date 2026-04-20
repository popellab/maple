#!/usr/bin/env python3
"""
Shared Pydantic models used across different workflows.

These models are defined here to avoid circular imports between
calibration_target_models.py and pydantic_models.py.
"""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from maple.core.calibration.enums import (
    ExperimentalSystem,
    ExtractionMethod,
    IndicationMatch,
    MeasurementDirectness,
    PerturbationType,
    SourceQuality,
    SourceType,
    TemporalResolution,
    TMECompatibility,
)


class InputType(str, Enum):
    """Classification of literature input data type for calibration targets."""

    DIRECT_PARAMETER = "direct_parameter"
    """Literature reports the model parameter value directly (e.g., 'k = 3/day')."""

    PROXY_MEASUREMENT = "proxy_measurement"
    """Literature reports a proxy that requires conversion (e.g., 'doubling time = 8h' → k = ln(2)/t)."""

    EXPERIMENTAL_CONDITION = "experimental_condition"
    """Protocol/experimental choice from paper (e.g., seeding density, E:T ratio)."""

    INFERRED_ESTIMATE = "inferred_estimate"
    """Value interpreted from qualitative text (e.g., 'maintained viability' → 0.95).

    Use when the numeric value does not appear literally in the paper but is
    a reasonable interpretation of qualitative statements. The value_snippet
    should contain the qualitative text that supports the interpretation,
    and the description should explain how the value was derived.

    Snippet validation is skipped for this input type since the value is
    not expected to appear literally in the text.
    """


class UncertaintyType(str, Enum):
    """Type of uncertainty measure reported in literature."""

    SD = "sd"  # Standard deviation
    SE = "se"  # Standard error
    CI95 = "ci95"  # 95% confidence interval
    RANGE = "range"  # Min-max range
    IQR = "iqr"  # Interquartile range


class TableExcerpt(BaseModel):
    """
    Structured excerpt from a table in a paper.

    Use this instead of value_snippet when the value comes from a table,
    where PDF text extraction produces unreadable concatenated rows.
    External validators check that table_id, column, row, and value all
    appear somewhere in the extracted paper text.
    """

    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(
        description="Table identifier (e.g., 'Table 2', 'Supplementary Table S1')"
    )
    column: str = Field(description="Column header the value falls under")
    row: str = Field(description="Row label/identifier")
    value: str = Field(
        description="Value as it appears in the table cell (e.g., '29 ± 10'). "
        "Used for validation against extracted paper text."
    )
    context: str = Field(
        description="Additional context (e.g., units in column header, table caption, "
        "or surrounding text that clarifies the value)",
    )


class FigureExcerpt(BaseModel):
    """
    Structured excerpt from a figure in a paper.

    Use this instead of value_snippet when the value is read from a figure
    (e.g., scatter plots, bar charts, dose-response curves). Figure-derived
    values cannot be validated by text matching, so inputs with figure_excerpt
    are flagged for manual review instead of failing snippet validation.
    """

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(
        description="Figure identifier (e.g., 'Figure 1C', 'Supplementary Figure S2A')"
    )
    value: str = Field(
        description="Value as read from the figure (e.g., '~5', '2-5 range'). "
        "Used for documentation; not validated by text matching."
    )
    description: str = Field(
        description="What was read from the figure (e.g., 'highest data point in scatter plot at 16h')"
    )
    context: str = Field(
        description="Additional context (e.g., figure caption text, axis labels, "
        "or experimental conditions shown in the panel)",
    )


class EstimateInput(BaseModel):
    """
    A literature-extracted input value for distribution derivation.

    Used in CalibrationTargetEstimates.inputs for values that feed into
    distribution_code to derive calibration target estimates. All provenance
    fields are REQUIRED to ensure full traceability.

    Supports both scalar and vector-valued inputs:
    - Scalar: value=42.0 (single measurement, or constant applied to all index points)
    - Vector: value=[10.0, 20.0, 30.0] (measurements at each index point)

    For experimental conditions used in submodel/observable code, use SubmodelInput.
    For modeling assumptions (e.g., n_mc_samples), use ModelingAssumption.
    """

    model_config = ConfigDict(extra="forbid")

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
    value_snippet: Optional[str] = Field(
        default=None,
        description="Exact text snippet from the source containing or supporting the value(s). "
        "Use table_excerpt when the value comes from a table, or figure_excerpt when the value "
        "is digitized from a figure. At least one of value_snippet, table_excerpt, or "
        "figure_excerpt must be provided.",
    )
    table_excerpt: Optional[TableExcerpt] = Field(
        default=None,
        description="Structured table excerpt when value comes from a table. "
        "Preferred over value_snippet for table-sourced data because PDF text extraction "
        "often produces unreadable concatenated rows.",
    )
    figure_excerpt: Optional[FigureExcerpt] = Field(
        default=None,
        description="Structured figure excerpt when value is digitized from a figure. "
        "Figure-derived values are flagged for manual review instead of failing snippet "
        "validation (since figure pixels are not in the PDF text layer).",
    )

    input_type: InputType = Field(
        default=InputType.DIRECT_PARAMETER,
        description=(
            "Classification of this input:\n"
            "- direct_parameter: Literature reports the value directly (e.g., 'mean = 42.0')\n"
            "- proxy_measurement: Requires conversion (e.g., 'doubling time = 8h' → rate constant)\n"
            "- experimental_condition: Protocol choice from paper (e.g., seeding density, E:T ratio)\n"
            "- inferred_estimate: Value interpreted from qualitative text (e.g., 'maintained viability' → 0.95).\n"
            "  Use when numeric value doesn't appear literally but is a reasonable interpretation.\n"
            "  Snippet validation is skipped for this type."
        ),
    )

    conversion_formula: Optional[str] = Field(
        None,
        description=(
            "For proxy_measurement type: formula showing how to convert to model parameter.\n\n"
            "Common conversions by parameter type:\n"
            "Rate constants (1/time):\n"
            "- Doubling time → k = ln(2) / t_double\n"
            "- Half-life → k = ln(2) / t_half\n"
            "- Fold change → k = ln(fold) / time\n"
            "- Mean residence time → k = 1 / MRT\n\n"
            "Binding parameters:\n"
            "- Ka (association) → Kd = 1 / Ka\n"
            "- koff from Kd, kon → koff = Kd * kon\n\n"
            "PK parameters:\n"
            "- CL + Vd → kel = CL / Vd\n"
            "- AUC → CL = Dose * F / AUC\n\n"
            "Production rates:\n"
            "- Steady-state → k_prod = k_decay * C_ss\n\n"
            "Not required for direct_parameter or experimental_condition types."
        ),
    )

    # Figure extraction fields
    source_type: SourceType = Field(
        default=SourceType.TEXT,
        description=(
            "Type of source from which the value was extracted:\n"
            "- text: Body text, results section, or abstract (default)\n"
            "- table: Table\n"
            "- figure: Figure (requires figure_id and extraction_method)"
        ),
    )

    figure_id: Optional[str] = Field(
        None,
        description=(
            "Figure identifier (e.g., 'Figure 2A', 'Fig. 3B'). "
            "Required when source_type='figure'."
        ),
    )

    extraction_method: Optional[ExtractionMethod] = Field(
        None,
        description=(
            "Method used to extract value from figure. Required when source_type='figure'.\n"
            "- manual: Manual reading from figure axes\n"
            "- digitizer: Generic digitizer software\n"
            "- webplotdigitizer: WebPlotDigitizer tool\n"
            "- other: Other method (specify in extraction_notes)"
        ),
    )

    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Additional context for figure extraction.\n"
            "Example: 'Read from y-axis at day 14 timepoint'\n"
            "Example: 'Digitized all points from survival curve'"
        ),
    )

    # Dispersion identification fields
    dispersion_type: Optional[UncertaintyType] = Field(
        None,
        description=(
            "Type of dispersion measure this input represents. REQUIRED when this input is a "
            "measure of spread/uncertainty (SD, SEM, CI bound, IQR bound, range).\n\n"
            "CRITICAL: Papers often report 'mean ± X' without specifying whether X is SD or SEM. "
            "Misidentifying SEM as SD underestimates uncertainty by a factor of sqrt(n), producing "
            "calibration targets that are far too constraining.\n\n"
            "How to distinguish SD vs SEM:\n"
            "1. Check methods section for explicit statement ('values are mean ± SD/SEM')\n"
            "2. Check table/figure legends for the error type\n"
            "3. Use sqrt(n) test: if '±X' scales as 1/sqrt(n) across subgroups with similar "
            "biology, it's SEM (SD = SEM × sqrt(n))\n"
            "4. Biological plausibility: SD should give CV typical for the measurement type "
            "(immune cell densities: CV 50-200%; tumor volumes: CV 50-100%). "
            "If treating ± as SD gives CV < 20%, it's almost certainly SEM.\n\n"
            "Set to None for inputs that are NOT dispersion measures (e.g., sample sizes, "
            "medians, means, quartile values)."
        ),
    )
    dispersion_type_rationale: Optional[str] = Field(
        None,
        description=(
            "REQUIRED when dispersion_type is set. Explain how you determined the dispersion type.\n\n"
            "Examples:\n"
            "- 'Methods section states: all values presented as mean ± standard deviation'\n"
            "- 'Table legend specifies SEM. SD = SEM × sqrt(n) = 15.0 × sqrt(368) = 287.7'\n"
            "- 'Paper does not specify. ±15.0 with n=368 gives CV=6.6% if SD (implausibly low) "
            "vs CV=126% if SEM (typical for immune cell densities). Treating as SEM.'\n"
            "- 'IQR explicitly stated in table header as median (Q1-Q3)'"
        ),
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued inputs are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued input cannot be an empty list")
        return v

    @model_validator(mode="after")
    def require_one_snippet_form(self) -> "EstimateInput":
        """At least one of value_snippet, table_excerpt, or figure_excerpt must be provided."""
        if not any([self.value_snippet, self.table_excerpt, self.figure_excerpt]):
            raise ValueError(
                f"Input '{self.name}': at least one of value_snippet, table_excerpt, or "
                f"figure_excerpt must be provided for traceability."
            )
        return self

    @field_validator("conversion_formula")
    @classmethod
    def require_conversion_for_proxy(cls, v: Optional[str], info) -> Optional[str]:
        """Warn if proxy_measurement type lacks conversion_formula."""
        import warnings

        # Access input_type from the data being validated
        input_type = info.data.get("input_type", InputType.DIRECT_PARAMETER)

        if input_type == InputType.PROXY_MEASUREMENT and v is None:
            warnings.warn(
                "EstimateInput with input_type='proxy_measurement' should have conversion_formula "
                "documenting how to convert to model parameter (e.g., 'k = ln(2) / t_half').",
                UserWarning,
            )
        return v

    @model_validator(mode="after")
    def validate_figure_fields(self) -> "EstimateInput":
        """Ensure figure sources have required figure_id and extraction_method."""
        if self.source_type == SourceType.FIGURE:
            missing = []
            if not self.figure_id:
                missing.append("figure_id")
            if not self.extraction_method:
                missing.append("extraction_method")
            if missing:
                raise ValueError(
                    f"When source_type='figure', the following fields are required: {', '.join(missing)}"
                )
        return self

    @model_validator(mode="after")
    def validate_dispersion_type_fields(self) -> "EstimateInput":
        """Require dispersion_type on dispersion-like inputs; require rationale when type is set."""
        # If dispersion_type is set, rationale is required
        if self.dispersion_type is not None and not self.dispersion_type_rationale:
            raise ValueError(
                f"Input '{self.name}': dispersion_type_rationale is required when "
                f"dispersion_type is set ('{self.dispersion_type.value}'). "
                f"Explain how you determined this is {self.dispersion_type.value.upper()} "
                f"(e.g., paper states explicitly, inferred from sqrt(n) scaling, "
                f"biological plausibility of implied CV)."
            )

        # If input name looks like a dispersion measure, dispersion_type is required
        if self.dispersion_type is None:
            _DISPERSION_KEYWORDS = [
                "sd",
                "std",
                "stdev",
                "se",
                "sem",
                "stderr",
                "sigma",
                "dispersion",
                "error_bar",
            ]
            name_lower = self.name.lower()
            # Check for keyword matches (as whole tokens separated by _ or at boundaries)
            name_tokens = set(name_lower.replace("-", "_").split("_"))
            if name_tokens & set(_DISPERSION_KEYWORDS):
                raise ValueError(
                    f"Input '{self.name}' appears to be a dispersion measure but "
                    f"dispersion_type is not set. Set dispersion_type to 'sd', 'se', "
                    f"'ci95', 'iqr', or 'range' and provide dispersion_type_rationale "
                    f"explaining how you determined the type.\n\n"
                    f"CRITICAL: Misidentifying SEM as SD underestimates uncertainty by "
                    f"sqrt(n). Use the sqrt(n) test or check biological plausibility of "
                    f"the implied CV to distinguish them."
                )

        return self


# Backwards compatibility alias
LiteratureInput = EstimateInput


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

    model_config = ConfigDict(extra="forbid")

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


class SubmodelInput(BaseModel):
    """
    An experimental input used in submodel or observable code.

    Used in Submodel.inputs and Observable.inputs for values that appear directly
    in ODE or observable calculations. Unlike EstimateInput (which feeds
    distribution_code), these are inputs to the model simulation itself.

    Examples:
    - E:T ratio for killing assay
    - Drug concentration for dose-response
    - Culture duration for growth experiments

    All provenance fields are REQUIRED for traceability.
    """

    model_config = ConfigDict(extra="forbid")

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

    # Figure extraction fields
    source_type: SourceType = Field(
        default=SourceType.TEXT,
        description=(
            "Type of source from which the value was extracted:\n"
            "- text: Body text, results section, or abstract (default)\n"
            "- table: Table\n"
            "- figure: Figure (requires figure_id and extraction_method)"
        ),
    )

    figure_id: Optional[str] = Field(
        None,
        description=(
            "Figure identifier (e.g., 'Figure 2A', 'Fig. 3B'). "
            "Required when source_type='figure'."
        ),
    )

    extraction_method: Optional[ExtractionMethod] = Field(
        None,
        description=(
            "Method used to extract value from figure. Required when source_type='figure'.\n"
            "- manual: Manual reading from figure axes\n"
            "- digitizer: Generic digitizer software\n"
            "- webplotdigitizer: WebPlotDigitizer tool\n"
            "- other: Other method (specify in extraction_notes)"
        ),
    )

    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Additional context for figure extraction.\n"
            "Example: 'Read from y-axis at day 14 timepoint'\n"
            "Example: 'Digitized all points from survival curve'"
        ),
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued inputs are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued input cannot be an empty list")
        return v

    @model_validator(mode="after")
    def validate_figure_fields(self) -> "SubmodelInput":
        """Ensure figure sources have required figure_id and extraction_method."""
        if self.source_type == SourceType.FIGURE:
            missing = []
            if not self.figure_id:
                missing.append("figure_id")
            if not self.extraction_method:
                missing.append("extraction_method")
            if missing:
                raise ValueError(
                    f"When source_type='figure', the following fields are required: {', '.join(missing)}"
                )
        return self


class KeyAssumption(BaseModel):
    """
    A single key assumption with its number and text.

    Note: CalibrationTarget now uses a simpler `caveats: List[str]` field.
    This class is kept for backward compatibility with ParameterMetadata and TestStatistic.
    """

    model_config = ConfigDict(extra="forbid")

    number: int = Field(description="Assumption number (1, 2, 3, ...)")
    text: str = Field(description="Assumption text")


class WeightScore(BaseModel):
    """A rubric-based weight score with justification."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(description="Rubric value (0-1)")
    justification: str = Field(description="Justification for this value")


class Source(BaseModel):
    """A bibliographic source (primary data)."""

    model_config = ConfigDict(extra="forbid")

    source_tag: str = Field(description="Unique tag for referencing")
    title: str = Field(description="Full title")
    first_author: str = Field(description="First author last name")
    year: int = Field(description="Publication year")
    doi: Optional[str] = Field(None, description="DOI (or null)")
    source_relevance: "ClinicalSourceRelevance" = Field(
        description=(
            "Structured assessment of how well this source's data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
    )


class SecondarySource(BaseModel):
    """A secondary data source (reference values, textbooks)."""

    model_config = ConfigDict(extra="forbid")

    source_tag: str = Field(description="Unique tag for referencing")
    title: str = Field(description="Full title")
    first_author: str = Field(description="First author last name")
    year: int = Field(description="Publication year")
    doi_or_url: Optional[str] = Field(None, description="DOI or URL (or null)")
    source_relevance: "ClinicalSourceRelevance" = Field(
        description=(
            "Structured assessment of how well this source's data translates to the target model. "
            "Captures indication match, source quality, perturbation context, and TME compatibility."
        ),
    )


# ============================================================================
# Provenance Models
# ============================================================================


class Snippet(BaseModel):
    """A text snippet from a source paper."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Exact text from the paper")
    source_tag: str = Field(
        description="Reference to source (must match a source_tag in primary_data_source or secondary_data_sources)"
    )
    figure_or_table: Optional[str] = Field(
        None, description="Figure/table reference (e.g., 'Figure 3A', 'Table 2')"
    )


class Validation(BaseModel):
    """Validation metadata (auto-populated by validation suite)."""

    model_config = ConfigDict(extra="forbid")

    tags: List[str] = Field(default_factory=list, description="Validation tags")
    validated_at: Optional[str] = Field(None, description="ISO timestamp of validation")


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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    medium: Optional[str] = Field(None, description="Culture medium (e.g., 'RPMI-1640', 'DMEM')")
    duration_hours: Optional[float] = Field(None, description="Culture duration in hours")
    additional: Optional[dict] = Field(
        None,
        description="Additional conditions (serum, supplements, temperature, CO2, etc.)",
    )


# ============================================================================
# Multi-Point Data Models (for trajectories and dose-response)
# ============================================================================


class TrajectoryData(BaseModel):
    """
    Time-course data with multiple measurements over time.

    Use this for kinetic experiments where the same observable is measured
    at multiple time points (e.g., proliferation curves, cytokine kinetics).
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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


# =============================================================================
# SOURCE RELEVANCE ASSESSMENT
# =============================================================================


class ClinicalSourceRelevance(BaseModel):
    """
    Source relevance assessment for clinical/in vivo calibration targets.

    Captures how well the source data translates to the target model for
    clinical observables (biopsies, resections, blood draws). Omits the
    measurement_directness, temporal_resolution, and experimental_system
    axes used by SubmodelTarget (in vitro/preclinical data), since clinical
    CalibrationTargets are always direct patient measurements.

    Translation sigma is computed from the 5 core axes (indication, species,
    source quality, perturbation, TME compatibility) added in quadrature.
    """

    model_config = ConfigDict(extra="forbid")

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
    perturbation_relevance: str = Field(
        min_length=30,
        description=(
            "Explain relevance of the experimental perturbation to the physiological "
            "parameter being estimated. Required even for physiological_baseline to "
            "document the observational context."
        ),
    )

    # TME compatibility
    tme_compatibility: TMECompatibility = Field(
        description=(
            "Tumor microenvironment compatibility assessment.\n"
            "- high: Source TME similar to target (e.g., both desmoplastic)\n"
            "- moderate: Some TME differences that may affect values\n"
            "- low: Major differences (e.g., T cell-permissive model for T cell-excluded tumor)"
        ),
    )
    tme_compatibility_notes: str = Field(
        min_length=30,
        description=(
            "Notes on TME differences and their expected impact on the observable. "
            "E.g., 'Treatment-naive PDAC with characteristic desmoplastic, "
            "immune-excluded TME matching model assumptions.'"
        ),
    )

    # Computed (machine-populated, not user-facing)
    validation_warnings: Optional[List[str]] = Field(
        default=None,
        description="Validation warnings generated by automated checks (populated by validators)",
    )


class SourceRelevanceAssessment(BaseModel):
    """
    Structured assessment of source-to-target relevance for calibration targets.

    This model captures how well the source data translates to the target model,
    including indication match, source quality, perturbation context, and TME
    compatibility. Validators use this information to flag potential issues.
    """

    model_config = ConfigDict(extra="forbid")

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
    perturbation_relevance: str = Field(
        min_length=30,
        description=(
            "Explain relevance of the experimental perturbation to the physiological "
            "parameter being estimated. E.g., if using drug-induced death rates, explain "
            "whether this represents an upper bound, typical value, or requires scaling."
        ),
    )

    # TME compatibility
    tme_compatibility: TMECompatibility = Field(
        description=(
            "Tumor microenvironment compatibility assessment.\n"
            "- high: Source TME similar to target (e.g., both desmoplastic)\n"
            "- moderate: Some TME differences that may affect values\n"
            "- low: Major differences (e.g., T cell-permissive model for T cell-excluded tumor)"
        ),
    )
    tme_compatibility_notes: str = Field(
        min_length=30,
        description=(
            "Notes on TME differences and their expected impact on the parameter. "
            "E.g., 'EG7 thymoma is highly T cell-permissive; PDAC is T cell-excluded. "
            "Expect 10-100x overestimation of infiltration rates.'"
        ),
    )

    # Measurement and system characterization (for translation sigma rubric)
    measurement_directness: MeasurementDirectness = Field(
        description=(
            "How many inferential steps between the raw measurement and the model parameter.\n"
            "- direct: trivial transform (unit conversion, ln2/x)\n"
            "- single_inversion: one kinetic/mechanistic model assumption\n"
            "- steady_state_inversion: inferred via balance equations with assumed rates\n"
            "- proxy_observable: surrogate measurement (RNA for protein, IHC score for concentration)"
        ),
    )
    temporal_resolution: TemporalResolution = Field(
        description=(
            "How well the temporal or dose structure constrains the parameter.\n"
            "- timecourse: >=3 timepoints/doses spanning the dynamic range\n"
            "- endpoint_pair: 2 timepoints or conditions\n"
            "- snapshot_or_equilibrium: single timepoint or assumed steady state"
        ),
    )
    experimental_system: ExperimentalSystem = Field(
        description=(
            "Biological fidelity of the experimental system. Distinct from source_quality "
            "(evidence reliability). Captures how well the experimental conditions "
            "recapitulate the in vivo tumor biology being modeled.\n"
            "- clinical_in_vivo: direct patient measurements\n"
            "- animal_in_vivo: animal model (intact system, species gap)\n"
            "- ex_vivo: freshly isolated tissue\n"
            "- in_vitro_coculture: multi-cell-type culture\n"
            "- in_vitro_primary: primary cells in standard culture\n"
            "- in_vitro_cell_line: immortalized cell lines"
        ),
    )

    # Computed (machine-populated, not user-facing)
    validation_warnings: Optional[List[str]] = Field(
        default=None,
        description="Validation warnings generated by automated checks (populated by validators)",
    )


# Rebuild models that use forward references to ClinicalSourceRelevance
Source.model_rebuild()
SecondarySource.model_rebuild()
