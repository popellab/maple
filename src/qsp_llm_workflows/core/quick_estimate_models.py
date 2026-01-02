#!/usr/bin/env python3
"""
Pydantic models for quick calibration target estimates.

Simple batch workflow: CSV in -> single LLM request -> CSV out.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QuickTargetEstimate(BaseModel):
    """Single quick estimate for a calibration target."""

    calibration_target_id: str = Field(description="Calibration target ID from input CSV")
    estimate: float = Field(description="Numeric estimate value from paper")
    units: str = Field(
        description="Units of the estimate (Pint-parseable, e.g., 'cell / millimeter**2', 'nanomolarity', 'day')"
    )
    uncertainty: Optional[float] = Field(
        None,
        description="Uncertainty value if reported (e.g., standard error, standard deviation, range width)",
    )
    uncertainty_type: Optional[str] = Field(
        None,
        description="Type of uncertainty: 'se' (standard error), 'sd' (standard deviation), 'ci95' (95% confidence interval), 'range' (min-max range), 'iqr' (interquartile range), or 'other'",
    )
    value_snippet: str = Field(description="Exact text snippet from paper containing the value")
    paper_name: str = Field(description="Full paper title")
    doi: str = Field(description="Paper DOI")
    threshold_description: str = Field(
        description="Human-readable description of calibration target threshold (e.g., 'samples taken at resection, average tumor volume 500 mm³')"
    )


class QuickEstimateResponse(BaseModel):
    """Response containing estimates for all calibration targets."""

    estimates: List[QuickTargetEstimate] = Field(
        description="List of estimates, one per calibration target from input CSV"
    )
