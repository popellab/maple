#!/usr/bin/env python3
"""
Pydantic models for Calibration Targets.

Calibration targets are raw observables extracted from literature, used to calibrate
QSP model parameters via Bayesian inference. Each observable has an experimental
context that may differ from the model context, requiring formal mismatch handling.

See docs/calibration_target_design.md for full specification.
"""

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from qsp_llm_workflows.core.shared_models import (
    Input,
    KeyAssumption,
    Source,
    SecondarySource,
)


# ============================================================================
# Scenario Models (Interventions and Measurements)
# ============================================================================


class DrugDosing(BaseModel):
    """
    Drug dosing intervention.

    Maps to schedule_dosing() and SimBiology dosing schedules.
    """

    agent: str = Field(
        description=(
            "Drug name (e.g., 'anti_PD1', 'gemcitabine', 'nivolumab'). "
            "Maps to '{agent}_dose' and '{agent}_schedule' in dosing config."
        )
    )
    dose: float = Field(description="Dose amount per administration")
    dose_units: str = Field(
        description=(
            "Dose units (e.g., 'mg/kg', 'mg/m2', 'mg', 'cells'). "
            "For mg/kg or mg/m2, patient_weight or patient_bsa must be provided."
        )
    )
    schedule: List[float] = Field(
        description=(
            "Dosing timepoints in days (e.g., [0, 7, 14] for doses on days 0, 7, and 14). "
            "Explicit timepoint list for maximum flexibility. "
            "Converts to SimBiology dose schedule; "
            "for qspio: can generate from [start, interval, repeat] pattern if regular."
        )
    )
    patient_weight: Optional[float] = Field(
        None, description="Patient weight in kg (required for mg/kg dosing)"
    )
    patient_bsa: Optional[float] = Field(
        None, description="Patient body surface area in m^2 (required for mg/m2 dosing)"
    )


class SurgicalResection(BaseModel):
    """
    Surgical resection intervention.

    Fractional removal of cells from specified compartments at a single timepoint.
    """

    timing: float = Field(description="Day when resection occurs (e.g., 14.0 for day 14)")
    fraction_removed: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of cells removed (0.0 to 1.0). "
            "E.g., 0.9 for 90% debulking, leaving 10% residual disease. "
            "Applied multiplicatively: new_count = old_count * (1 - fraction_removed)."
        ),
    )
    affected_species: List[str] = Field(
        description=(
            "List of model species affected by resection (e.g., ['V_T.C1', 'V_T.Treg']). "
            "Typically includes tumor cells and immune cells in tumor compartment."
        )
    )


# TODO: Add RadiationTherapy intervention schema
# TODO: Add CellTransferTherapy intervention schema
# TODO: Add TumorInoculation intervention schema


class Measurement(BaseModel):
    """
    Measurement specification: when and what to measure from the model.

    Defines the timing (relative to a biomarker threshold) and computation
    for extracting a model-derived observable that can be compared to the
    literature-derived calibration target.

    All measurements are biomarker-triggered to ensure biological interpretability
    and simulation reproducibility. There is no absolute "time zero" in biology.
    """

    timing_type: Literal["biomarker_triggered"] = "biomarker_triggered"

    required_species: List[str] = Field(
        description=(
            "Model species required for measurement (SimBiology format). "
            "E.g., ['V_T.CD8', 'V_T.C1'] for CD8-to-tumor ratio measurement."
        )
    )

    computation_code: str = Field(
        description=(
            "Python code defining compute_measurement(time, species_dict, ureg). "
            "Converts model species to observable matching calibration target. "
            "Must return Pint Quantity for comparison. "
            "Example:\n"
            "def compute_measurement(time, species_dict, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )
    )

    biomarker_species: str = Field(
        description=(
            "Model species to monitor for trigger (SimBiology format). "
            "E.g., 'V_T.C1' for tumor cells, 'V_T.TGFb' for TGF-beta concentration. "
            "Common triggers: tumor burden (V_T.C1), circulating biomarkers, immune cell counts."
        )
    )

    threshold: float = Field(
        description=(
            "Threshold value that triggers measurement when crossed. "
            "Value should be in the natural units of biomarker_species (from species_units.json). "
            "E.g., if biomarker is 'V_T.C1' (cells), threshold might be 5e8 for resectable tumor. "
            "Extract from paper when possible, otherwise document as modeling assumption in inputs."
        )
    )

    comparison: Literal[">", "<"] = Field(
        description=(
            "Comparison operator: '>' (greater than) or '<' (less than). "
            "E.g., '>' triggers when biomarker exceeds threshold, '<' triggers when it falls below. "
            "For tumor burden: '>' = tumor reaches size. For response: '<' = tumor shrinks below."
        )
    )

    timepoints: List[float] = Field(
        description=(
            "Days relative to trigger event to sample model output. "
            "For single-point: [0.0] = at trigger. "
            "For post-trigger tracking: [0.0, 7.0, 14.0] = at trigger, +7d, +14d. "
            "For derivatives: [-1.0, 0.0, 1.0] = window around trigger for finite differences."
        )
    )


class Scenario(BaseModel):
    """
    Experimental scenario: sequence of interventions and measurements.

    Defines all exogenous events during the experiment (treatments, measurements, etc.).
    """

    description: str = Field(
        description="Human-readable description of the scenario (e.g., 'Anti-PD-1 monotherapy with resection')"
    )
    interventions: List[Union[DrugDosing, SurgicalResection]] = Field(
        description="List of interventions applied during the experiment"
    )
    measurements: List[Measurement] = Field(
        description=(
            "List of measurement events specifying when and what to measure from model. "
            "All measurements are biomarker-triggered (relative to observable biological state). "
            "Creates model-derived observables for comparison with calibration target. "
            "Must contain at least one measurement."
        )
    )

    @field_validator("measurements")
    @classmethod
    def validate_at_least_one_measurement(cls, v: List[Measurement]) -> List[Measurement]:
        """Ensure at least one measurement is specified."""
        if len(v) < 1:
            raise ValueError("Scenario must include at least one measurement")
        return v


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


class Indication(str, Enum):
    """Cancer indication (hierarchical leaf nodes)."""

    # GI Adenocarcinoma
    PDAC = "PDAC"
    COLORECTAL = "colorectal"
    GASTRIC = "gastric"
    ESOPHAGEAL = "esophageal"

    # Hepatobiliary
    HEPATOCELLULAR = "hepatocellular"
    CHOLANGIOCARCINOMA = "cholangiocarcinoma"

    # Lung
    LUNG_ADENO = "lung_adeno"
    LUNG_SQUAMOUS = "lung_squamous"
    SMALL_CELL_LUNG = "small_cell_lung"

    # Immunogenic Solid
    MELANOMA = "melanoma"
    RENAL_CELL = "renal_cell"
    HEAD_AND_NECK = "head_and_neck"

    # Other Solid
    BREAST = "breast"
    OVARIAN = "ovarian"
    PROSTATE = "prostate"
    GLIOBLASTOMA = "glioblastoma"

    # Heme Malignancy
    LYMPHOMA = "lymphoma"
    LEUKEMIA = "leukemia"
    MYELOMA = "myeloma"

    # Non-cancer
    HEALTHY = "healthy"
    OTHER_DISEASE = "other_disease"


class Compartment(str, Enum):
    """Anatomical compartment (hierarchical notation with dot separator)."""

    # Tumor
    TUMOR_PRIMARY = "tumor.primary"
    TUMOR_METASTATIC = "tumor.metastatic"
    TUMOR_UNSPECIFIED = "tumor.unspecified"

    # Blood
    BLOOD_WHOLE_BLOOD = "blood.whole_blood"
    BLOOD_PBMC = "blood.PBMC"
    BLOOD_PLASMA_SERUM = "blood.plasma_serum"

    # Lymphoid
    LYMPHOID_TUMOR_DRAINING_LN = "lymphoid.tumor_draining_LN"
    LYMPHOID_OTHER_LN = "lymphoid.other_LN"
    LYMPHOID_SPLEEN = "lymphoid.spleen"
    LYMPHOID_BONE_MARROW = "lymphoid.bone_marrow"

    # Other
    OTHER_TISSUE = "other_tissue"
    IN_VITRO = "in_vitro"


class System(str, Enum):
    """Experimental system (hierarchical notation with dot separator)."""

    # Clinical
    CLINICAL_BIOPSY = "clinical.biopsy"
    CLINICAL_RESECTION = "clinical.resection"
    CLINICAL_LIQUID_BIOPSY = "clinical.liquid_biopsy"

    # Ex vivo
    EX_VIVO_FRESH = "ex_vivo.fresh"
    EX_VIVO_CULTURED = "ex_vivo.cultured"

    # Animal in vivo
    ANIMAL_IN_VIVO_ORTHOTOPIC = "animal_in_vivo.orthotopic"
    ANIMAL_IN_VIVO_SUBCUTANEOUS = "animal_in_vivo.subcutaneous"
    ANIMAL_IN_VIVO_PDX = "animal_in_vivo.PDX"
    ANIMAL_IN_VIVO_GEM = "animal_in_vivo.GEM"
    ANIMAL_IN_VIVO_SYNGENEIC = "animal_in_vivo.syngeneic"

    # In vitro
    IN_VITRO_ORGANOID = "in_vitro.organoid"
    IN_VITRO_PRIMARY_CELLS = "in_vitro.primary_cells"
    IN_VITRO_CELL_LINE = "in_vitro.cell_line"


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

    Uses typed enums for all context dimensions with hierarchical notation
    encoded in enum values (e.g., Compartment.TUMOR_PRIMARY = "tumor.primary").
    """

    species: Species = Field(description="Species")
    mouse_subspecifier: Optional[MouseSubspecifier] = Field(
        None, description="Optional mouse subspecifier (only if species is mouse)"
    )
    indication: Indication = Field(description="Cancer indication")
    compartment: Compartment = Field(description="Anatomical compartment")
    system: System = Field(description="Experimental system")
    treatment: TreatmentContext = Field(description="Treatment context")
    stage: Stage = Field(description="Disease stage")


# ============================================================================
# Calibration Target
# ============================================================================


class CalibrationTargetFooters(BaseModel):
    """
    Footer fields for calibration target files.

    These are metadata about the file itself (not generated by LLM):
    - What observable is being tracked
    - Which cancer type
    - When and how it was derived
    """

    calibration_target_id: str = Field(description="Unique calibration target identifier")
    cancer_type: str = Field(description="Cancer type (e.g., 'PDAC')")
    tags: List[str] = Field(default_factory=list, description="Metadata tags")
    derivation_id: Optional[str] = Field(None, description="Unique derivation identifier")
    derivation_timestamp: Optional[str] = Field(None, description="ISO timestamp of derivation")


class CalibrationTargetEstimates(BaseModel):
    """Calibration target estimates with structured inputs and derivation."""

    median: float = Field(description="Median value")
    iqr: float = Field(description="Interquartile range")
    ci95: List[float] = Field(description="95% confidence interval [lower, upper]")
    units: str = Field(description="Units of the observable")
    inputs: List[Input] = Field(description="List of inputs used in derivation")
    distribution_code: str = Field(
        description=(
            "Python code defining a derive_distribution(inputs, ureg) function. "
            "inputs is a dict mapping input names to Pint Quantities. "
            "Must return dict with Pint Quantities: median_obs, iqr_obs, ci95_obs ([lower, upper]). "
            "CRITICAL: NO magic numbers allowed - every coefficient, multiplier, or constant must come from inputs. "
            "CRITICAL: Use MC methods (parametric bootstrap) for distribution estimates, NOT analytical approximations.\n"
            "Example (parametric bootstrap from literature):\n"
            "def derive_distribution(inputs, ureg):\n"
            "    import numpy as np\n"
            "    mean = inputs['cd8_density_mean']  # From Table 2\n"
            "    sd = inputs['cd8_density_sd']  # From Table 2\n"
            "    n_samples = inputs['n_mc_samples']  # e.g., 10000\n"
            "    # Parametric bootstrap: sample from reported distribution\n"
            "    samples = np.random.normal(mean.magnitude, sd.magnitude, int(n_samples.magnitude)) * mean.units\n"
            "    median_obs = np.median(samples)\n"
            "    iqr_obs = np.percentile(samples, 75) - np.percentile(samples, 25)\n"
            "    ci95_obs = [np.percentile(samples, 2.5), np.percentile(samples, 97.5)]\n"
            "    return {'median_obs': median_obs, 'iqr_obs': iqr_obs, 'ci95_obs': ci95_obs}"
        )
    )


class CalibrationTarget(BaseModel):
    """
    A calibration target: a raw observable extracted from literature.

    Used to calibrate QSP model parameters via Bayesian inference.

    Excludes header fields that are added during unpacking:
    - calibration_target_id, cancer_type
    - tags, derivation_id, derivation_timestamp
    """

    # --- Observable fields (LLM-generated) ---
    calibration_target_estimates: CalibrationTargetEstimates = Field(
        description="Observable estimates with inputs and derivation"
    )
    description: str = Field(description="Human-readable description of the observable")

    # --- Scenario specification (LLM-generated) ---
    scenario: Scenario = Field(
        description=(
            "Experimental scenario: sequence of interventions and measurements. "
            "Must include at least one measurement to specify what observable is being extracted. "
            "Interventions list may be empty for natural/untreated state measurements."
        )
    )

    # --- Experimental context (LLM-generated, grouped for distance calculations) ---
    experimental_context: ExperimentalContext = Field(
        description="Experimental context of the observable"
    )

    # --- Study information (LLM-generated) ---
    study_overview: str = Field(
        description="High-level biological context (WHAT and WHY) in 1-2 sentences"
    )
    study_design: str = Field(description="Concrete experimental details (HOW) in 1-2 sentences")
    derivation_explanation: str = Field(
        description="Step-by-step explanation of derivation code with assumption justifications"
    )
    key_assumptions: List[KeyAssumption] = Field(
        description="List of key assumptions with numbers and text"
    )
    key_study_limitations: str = Field(
        description="Important limitations and their impact on reliability"
    )

    # --- Sources (LLM-generated) ---
    primary_data_source: Source = Field(
        description="Primary data source (the paper where this observable was measured)"
    )
    secondary_data_sources: List[SecondarySource] = Field(
        description="Secondary data sources (reference values, constants)"
    )

    @classmethod
    def get_header_fields(cls) -> set[str]:
        """Get set of field names that are headers (from CalibrationTargetFooters)."""
        return set(CalibrationTargetFooters.model_fields.keys())

    def split(self) -> tuple[CalibrationTargetFooters, dict]:
        """
        Split into headers and content.

        Returns:
            Tuple of (headers_model, content_dict)
        """
        header_fields = self.get_header_fields()
        all_data = self.model_dump()

        # Extract headers
        header_data = {k: v for k, v in all_data.items() if k in header_fields}
        headers = CalibrationTargetFooters(**header_data)

        # Extract content
        content = {k: v for k, v in all_data.items() if k not in header_fields}

        return headers, content

    @classmethod
    def from_split(cls, headers: CalibrationTargetFooters, content: dict) -> "CalibrationTarget":
        """
        Create CalibrationTarget from headers and content.

        Args:
            headers: CalibrationTargetFooters instance
            content: Content dictionary (LLM-generated fields)

        Returns:
            Complete CalibrationTarget instance
        """
        return cls(**{**headers.model_dump(), **content})
