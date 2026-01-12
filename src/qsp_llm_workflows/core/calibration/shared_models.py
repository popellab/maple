#!/usr/bin/env python3
"""
Shared Pydantic models used across different workflows.

These models are defined here to avoid circular imports between
calibration_target_models.py and pydantic_models.py.
"""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class Input(BaseModel):
    """
    An input value used in calibration target derivation.

    Supports both scalar and vector-valued inputs:
    - Scalar: value=42.0 (single measurement, or constant applied to all index points)
    - Vector: value=[10.0, 20.0, 30.0] (measurements at each index point)

    Vector-valued inputs must have the same length as the parent's index_values.
    Scalar inputs are broadcast (used for all index points).
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
    description: str = Field(description="What this input represents")
    source_ref: Optional[str] = Field(
        None,
        description="Source reference tag (must match a defined source, or 'modeling_assumption')",
    )
    value_table_or_section: Optional[str] = Field(
        None, description="Location of value in source (e.g., 'Table 2', 'Results section')"
    )
    value_snippet: Optional[str] = Field(
        None, description="Exact text snippet from source containing the value(s)"
    )

    @field_validator("value")
    @classmethod
    def ensure_list_not_empty(cls, v: Union[float, List[float]]) -> Union[float, List[float]]:
        """Ensure vector-valued inputs are not empty lists."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Vector-valued input cannot be an empty list")
        return v


class KeyAssumption(BaseModel):
    """A single key assumption with its number and text."""

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
