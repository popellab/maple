#!/usr/bin/env python3
"""
Shared Pydantic models used across different workflows.

These models are defined here to avoid circular imports between
calibration_target_models.py and pydantic_models.py.
"""

from typing import Optional

from pydantic import BaseModel, Field


class Input(BaseModel):
    """An input value used in parameter or test statistic derivation."""

    name: str = Field(description="Input name")
    value: float = Field(description="Input value")
    units: str = Field(
        description="Input units (must be Pint-parseable, e.g., 'pg/mL', 'cells/mm^2', 'dimensionless')"
    )
    description: str = Field(description="Input description")
    source_ref: Optional[str] = Field(description="Source reference tag (or null)")
    value_table_or_section: Optional[str] = Field(description="Location of value in source")
    value_snippet: Optional[str] = Field(description="Text snippet containing value")


class KeyAssumption(BaseModel):
    """A single key assumption with its number and text."""

    number: int = Field(description="Assumption number (1, 2, 3, ...)")
    text: str = Field(description="Assumption text")


class WeightScore(BaseModel):
    """A rubric-based weight score with justification."""

    value: float = Field(description="Rubric value (0-1)")
    justification: str = Field(description="Justification for this value")
