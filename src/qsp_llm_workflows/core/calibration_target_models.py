#!/usr/bin/env python3
"""
Pydantic models for Calibration Targets.

Calibration targets are raw observables extracted from literature, used to calibrate
QSP model parameters via Bayesian inference. Each observable has an experimental
context that may differ from the model context, requiring formal mismatch handling.

See docs/calibration_target_design.md for full specification.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Context Dimension Enums
# ============================================================================


class Species(str, Enum):
    """Species for observable context."""

    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    NON_HUMAN_PRIMATE = "non_human_primate"
    OTHER = "other"


class MouseSubspecifier(str, Enum):
    """Optional mouse subspecifier."""

    WILD_TYPE = "wild_type"
    IMMUNOCOMPROMISED = "immunocompromised"
    TRANSGENIC = "transgenic"


class TreatmentHistory(str, Enum):
    """Treatment history options (multi-select)."""

    TREATMENT_NAIVE = "treatment_naive"
    PRIOR_CHEMOTHERAPY = "prior_chemotherapy"
    PRIOR_RADIATION = "prior_radiation"
    PRIOR_IMMUNOTHERAPY = "prior_immunotherapy"
    PRIOR_TARGETED_THERAPY = "prior_targeted_therapy"
    PRIOR_SURGERY = "prior_surgery"


class TreatmentStatus(str, Enum):
    """Current treatment status (single select)."""

    OFF_TREATMENT = "off_treatment"
    ON_TREATMENT = "on_treatment"


class StageExtent(str, Enum):
    """Disease extent."""

    RESECTABLE = "resectable"
    BORDERLINE_RESECTABLE = "borderline_resectable"
    LOCALLY_ADVANCED = "locally_advanced"
    METASTATIC = "metastatic"


class StageBurden(str, Enum):
    """Disease burden."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ObservableClass(str, Enum):
    """Classification of observable types."""

    CELL_DENSITY = "cell_density"
    CELL_FRACTION = "cell_fraction"
    CONCENTRATION = "concentration"
    KINETIC_RATE = "kinetic_rate"
    FUNCTIONAL_READOUT = "functional_readout"
    TUMOR_MEASUREMENT = "tumor_measurement"
    SURVIVAL = "survival"


class UncertaintyType(str, Enum):
    """Type of reported uncertainty."""

    SD = "sd"  # Standard deviation
    SE = "se"  # Standard error
    CI_95 = "ci_95"  # 95% confidence interval
    IQR = "iqr"  # Interquartile range
    RANGE = "range"  # Min-max range
    ASSUMED = "assumed"  # No uncertainty reported; CV=30% assumed


# ============================================================================
# Context Models
# ============================================================================


class Stage(BaseModel):
    """Disease stage with extent and burden."""

    extent: StageExtent = Field(description="Disease extent")
    burden: StageBurden = Field(description="Disease burden")


class TreatmentContext(BaseModel):
    """Treatment context with history and current status."""

    history: List[TreatmentHistory] = Field(description="Treatment history (select all that apply)")
    status: TreatmentStatus = Field(description="Current treatment status")
    specifier: Optional[str] = Field(None, description="Optional drug name or class specifier")


class ExperimentalContext(BaseModel):
    """
    Experimental context for an observable.

    Uses hierarchical string notation for indication, compartment, and system.
    E.g., "tumor.primary", "clinical.resection", "gi_adenocarcinoma.PDAC"
    """

    species: Species = Field(description="Species")
    mouse_subspecifier: Optional[MouseSubspecifier] = Field(
        None, description="Optional mouse subspecifier (only if species is mouse)"
    )
    indication: str = Field(
        description="Indication (hierarchical, e.g., 'PDAC', 'gi_adenocarcinoma.colorectal')"
    )
    compartment: str = Field(
        description="Compartment (hierarchical, e.g., 'tumor.primary', 'blood.PBMC')"
    )
    system: str = Field(
        description="Experimental system (hierarchical, e.g., 'clinical.resection', 'animal_in_vivo.orthotopic')"
    )
    treatment: TreatmentContext = Field(description="Treatment context")
    stage: Stage = Field(description="Disease stage")


class ModelContext(BaseModel):
    """
    Model's assumed context for calibration.

    This defines the reference context against which observables are compared.
    """

    species: Species = Field(description="Model species assumption")
    indication: str = Field(description="Model indication (e.g., 'PDAC')")
    compartment: str = Field(description="Model compartment (e.g., 'tumor.primary')")
    system: str = Field(description="Model system (e.g., 'clinical')")
    treatment: List[TreatmentHistory] = Field(description="Model treatment assumption")
    treatment_specifier: Optional[str] = Field(None, description="Optional treatment specifier")
    stage: Stage = Field(description="Model stage assumption")


# ============================================================================
# Observable Model
# ============================================================================


class Observable(BaseModel):
    """
    A raw observable value extracted from literature.

    This represents what was actually measured, before any context adjustments.
    """

    observable_class: ObservableClass = Field(description="Classification of observable type")
    description: str = Field(description="Human-readable description of the observable")
    value: float = Field(description="Observed value")
    uncertainty: float = Field(
        description="Uncertainty (interpretation depends on uncertainty_type)"
    )
    uncertainty_type: UncertaintyType = Field(description="Type of reported uncertainty")
    units: str = Field(
        description="Units of the observable (Pint-parseable, e.g., 'cells/mm^2', 'pg/mL', 'dimensionless')"
    )
    sample_size: Optional[int] = Field(None, description="Sample size (n) if reported")


# ============================================================================
# Source Models (reused from pydantic_models.py pattern)
# ============================================================================


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
# Input Model (for derivation code)
# ============================================================================


class Input(BaseModel):
    """An input value used in observable derivation."""

    name: str = Field(description="Input name")
    value: float = Field(description="Input value")
    units: str = Field(
        description="Input units (Pint-parseable, e.g., 'pg/mL', 'cells/mm^2', 'dimensionless')"
    )
    description: str = Field(description="Input description")
    source_ref: Optional[str] = Field(description="Source reference tag (or null)")
    value_table_or_section: Optional[str] = Field(description="Location of value in source")
    value_snippet: Optional[str] = Field(description="Text snippet containing value")


# ============================================================================
# Key Assumption Model
# ============================================================================


class KeyAssumption(BaseModel):
    """A single key assumption with its number and text."""

    number: int = Field(description="Assumption number (1, 2, 3, ...)")
    text: str = Field(description="Assumption text")


# ============================================================================
# Calibration Target Estimates (LLM-generated)
# ============================================================================


class CalibrationTargetEstimates(BaseModel):
    """
    Estimates for a calibration target with structured inputs and derivation.

    Similar to TestStatisticEstimates but tailored for calibration targets.
    """

    inputs: List[Input] = Field(description="List of inputs used in derivation")
    derivation_code: str = Field(
        description=(
            "Python code defining a derive_observable(inputs, ureg) function. "
            "inputs is a dict mapping input names to Pint Quantities. "
            "Access values like: inputs['concentration'] (already a Pint Quantity). "
            "Must return dict with Pint Quantities: median_obs, iqr_obs, "
            "ci95_obs ([lower, upper])."
        )
    )
    median: float = Field(description="Median value")
    iqr: float = Field(description="Interquartile range")
    ci95: List[float] = Field(description="95% confidence interval [lower, upper]")
    units: str = Field(description="Units of the estimate")


# ============================================================================
# Header Models (not LLM-generated)
# ============================================================================


class CalibrationTargetHeaders(BaseModel):
    """
    Header fields for calibration target files.

    These are metadata about the file itself (not generated by LLM):
    - What calibration target is being estimated
    - Which cancer type and model context
    - When and how it was derived
    """

    schema_version: str = Field(description="Schema version (e.g., 'v1')")
    calibration_target_id: str = Field(description="Unique calibration target identifier")
    cancer_type: str = Field(description="Cancer type (e.g., 'PDAC')")
    model_context: ModelContext = Field(description="Model's assumed context for calibration")
    tags: List[str] = Field(default_factory=list, description="Metadata tags")
    context_hash: str = Field(description="Hash of model context for provenance")
    derivation_id: Optional[str] = Field(None, description="Unique derivation identifier")
    derivation_timestamp: Optional[str] = Field(None, description="ISO timestamp of derivation")


# ============================================================================
# Complete Calibration Target Model (LLM-generated content)
# ============================================================================


class CalibrationTarget(BaseModel):
    """
    Complete calibration target generated by LLM.

    This contains the LLM-generated content for a calibration target.
    Header fields are added during unpacking.

    Key differences from TestStatistic:
    - Has experimental_context (6 dimensions)
    - Has observable with class, value, uncertainty
    - No model_output_code (computed externally)
    """

    # Observable specification
    observable: Observable = Field(description="Raw observable value extracted from literature")
    experimental_context: ExperimentalContext = Field(
        description="Experimental context of the observable"
    )

    # Study information
    study_overview: str = Field(
        description="High-level biological context (WHAT and WHY) in 1-2 sentences"
    )
    study_design: str = Field(description="Concrete experimental details (HOW) in 1-2 sentences")

    # Derivation
    calibration_target_estimates: CalibrationTargetEstimates = Field(
        description="Calibration target estimates with inputs and derivation"
    )
    key_assumptions: List[KeyAssumption] = Field(
        description="List of key assumptions with numbers and text"
    )
    derivation_explanation: str = Field(
        description="Step-by-step explanation of derivation code with assumption justifications"
    )
    key_study_limitations: str = Field(
        description="Important limitations and their impact on reliability"
    )

    # Sources
    primary_data_sources: List[Source] = Field(
        description="Primary data sources (original measurements)"
    )
    secondary_data_sources: List[SecondarySource] = Field(
        description="Secondary data sources (reference values, constants)"
    )

    @classmethod
    def get_header_fields(cls) -> set[str]:
        """Get set of field names that are headers (from CalibrationTargetHeaders)."""
        return set(CalibrationTargetHeaders.model_fields.keys())


# ============================================================================
# Full Calibration Target (Headers + Content)
# ============================================================================


class FullCalibrationTarget(BaseModel):
    """
    Complete calibration target with headers and LLM-generated content.

    This is the full structure written to YAML files.
    """

    # Header fields
    schema_version: str = Field(description="Schema version (e.g., 'v1')")
    calibration_target_id: str = Field(description="Unique calibration target identifier")
    cancer_type: str = Field(description="Cancer type (e.g., 'PDAC')")
    model_context: ModelContext = Field(description="Model's assumed context for calibration")
    tags: List[str] = Field(default_factory=list, description="Metadata tags")
    context_hash: str = Field(description="Hash of model context for provenance")
    derivation_id: Optional[str] = Field(None, description="Unique derivation identifier")
    derivation_timestamp: Optional[str] = Field(None, description="ISO timestamp of derivation")

    # LLM-generated content
    observable: Observable = Field(description="Raw observable value extracted from literature")
    experimental_context: ExperimentalContext = Field(
        description="Experimental context of the observable"
    )
    study_overview: str = Field(
        description="High-level biological context (WHAT and WHY) in 1-2 sentences"
    )
    study_design: str = Field(description="Concrete experimental details (HOW) in 1-2 sentences")
    calibration_target_estimates: CalibrationTargetEstimates = Field(
        description="Calibration target estimates with inputs and derivation"
    )
    key_assumptions: List[KeyAssumption] = Field(
        description="List of key assumptions with numbers and text"
    )
    derivation_explanation: str = Field(
        description="Step-by-step explanation of derivation code with assumption justifications"
    )
    key_study_limitations: str = Field(
        description="Important limitations and their impact on reliability"
    )
    primary_data_sources: List[Source] = Field(
        description="Primary data sources (original measurements)"
    )
    secondary_data_sources: List[SecondarySource] = Field(
        description="Secondary data sources (reference values, constants)"
    )

    @classmethod
    def get_header_fields(cls) -> set[str]:
        """Get set of field names that are headers."""
        return set(CalibrationTargetHeaders.model_fields.keys())

    def split(self) -> tuple[CalibrationTargetHeaders, CalibrationTarget]:
        """
        Split into headers and content.

        Returns:
            Tuple of (headers_model, content_model)
        """
        header_fields = self.get_header_fields()
        all_data = self.model_dump()

        # Extract headers
        header_data = {k: v for k, v in all_data.items() if k in header_fields}
        headers = CalibrationTargetHeaders(**header_data)

        # Extract content
        content_data = {k: v for k, v in all_data.items() if k not in header_fields}
        content = CalibrationTarget(**content_data)

        return headers, content

    @classmethod
    def from_split(
        cls, headers: CalibrationTargetHeaders, content: CalibrationTarget
    ) -> "FullCalibrationTarget":
        """
        Create FullCalibrationTarget from headers and content.

        Args:
            headers: CalibrationTargetHeaders instance
            content: CalibrationTarget instance (LLM-generated fields)

        Returns:
            Complete FullCalibrationTarget instance
        """
        return cls(**{**headers.model_dump(), **content.model_dump()})
