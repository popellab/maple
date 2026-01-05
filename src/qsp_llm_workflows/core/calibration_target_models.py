#!/usr/bin/env python3
"""
Pydantic models for Calibration Targets.

Calibration targets are raw observables extracted from literature, used to calibrate
QSP model parameters via Bayesian inference. Each observable has an experimental
context that may differ from the model context, requiring formal mismatch handling.

See docs/calibration_target_design.md for full specification.
"""

import ast
from difflib import SequenceMatcher
from enum import Enum
from typing import List, Literal, Optional, Union

import numpy as np
import requests
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

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

    measurement_species: List[str] = Field(
        description=(
            "Model species needed to compute the measurement (SimBiology format). "
            "E.g., ['V_T.CD8', 'V_T.C1'] for CD8-to-tumor ratio measurement."
        )
    )

    measurement_code: str = Field(
        description=(
            "📊 WHAT TO MEASURE: Python code defining compute_measurement(time, species_dict, ureg). "
            "Converts model species to observable matching calibration target. "
            "This is what gets compared to literature values. "
            "Must return Pint Quantity for comparison. "
            "Example:\n"
            "def compute_measurement(time, species_dict, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    ratio = cd8 / tumor\n"
            "    return ratio.to(ureg.dimensionless)"
        )
    )

    trigger_species: str = Field(
        description=(
            "⏰ WHEN TO MEASURE: Model species that triggers the measurement (SimBiology format). "
            "NOT the same as what you're measuring! "
            "E.g., 'V_T.C1' (tumor burden) triggers measurement of CD8 density. "
            "Common triggers: tumor size at resection, treatment timepoint, clinical milestone."
        )
    )

    threshold_conversion_code: str = Field(
        description=(
            "Python code to convert trigger_species to threshold comparison space. "
            "Function signature: compute_threshold_value(species_dict, inputs, ureg) -> Pint Quantity.\n\n"
            "COMMON CASE (identity mapping - trigger in natural units):\n"
            "def compute_threshold_value(species_dict, inputs, ureg):\n"
            "    return species_dict['V_T.TGFb']  # Direct concentration comparison\n\n"
            "CONVERSION CASE (cells → volume):\n"
            "def compute_threshold_value(species_dict, inputs, ureg):\n"
            "    tumor_cells = species_dict['V_T.C1']\n"
            "    cell_density = inputs['cell_packing_density']  # From inputs list\n"
            "    volume = tumor_cells / cell_density\n"
            "    return volume.to(ureg.mm**3)\n\n"
            "All conversion factors must come from inputs list with source tracking."
        )
    )

    threshold: float = Field(
        description=(
            "Threshold value that triggers measurement when crossed. "
            "Units specified by threshold_units field. "
            "Extract from paper when possible ('resection at 500 mm³'), "
            "otherwise document as modeling assumption in inputs."
        )
    )

    threshold_units: str = Field(
        description=(
            "Units of threshold value (must be Pint-parseable). "
            "If threshold_conversion_code provided: units of that code's output. "
            "If threshold_conversion_code is None: must match natural units of trigger_species. "
            "Examples: 'millimeter**3' (volume), 'cell' (count), 'nanomolarity' (concentration)."
        )
    )

    threshold_input_name: str = Field(
        description=(
            "Name of input that provides the threshold value with source tracking. "
            "Must reference an input in the inputs list with matching value/units. "
            "⚠️ CRITICAL: Threshold value MUST come from primary source, NOT modeling_assumption. "
            "Extract from paper (e.g., 'resected at 500 mm³', 'enrolled at 1 cm diameter'). "
            "Conversion factors (cell_density, geometric constants) CAN use modeling_assumption."
        )
    )

    comparison: Literal[">", "<"] = Field(
        description=(
            "Comparison operator: '>' (greater than) or '<' (less than). "
            "E.g., '>' triggers when computed value exceeds threshold, '<' when it falls below. "
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

    # ========================================================================
    # Static helper methods for validation
    # ========================================================================

    @staticmethod
    def resolve_doi(doi: str) -> Optional[dict]:
        """
        Resolve DOI and get metadata from CrossRef.

        Returns:
            Dict with title, first_author, year, doi or None if resolution fails
        """
        if not doi:
            return None

        # Clean DOI
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

        try:
            # Query CrossRef API
            url = f"https://doi.org/{doi_clean}"
            headers = {"Accept": "application/vnd.citationstyles.csl+json"}
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                return None

            metadata = response.json()

            # Extract title
            title = metadata.get("title", "")
            if isinstance(title, list) and len(title) > 0:
                title = title[0]

            # Extract first author
            authors = metadata.get("author", [])
            first_author = None
            if authors and len(authors) > 0:
                first_author = authors[0].get("family", "")

            # Extract year
            date_parts = metadata.get("issued", {}).get("date-parts", [[]])
            year = None
            if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
                year = date_parts[0][0]

            return {"title": title, "first_author": first_author, "year": year, "doi": doi_clean}

        except Exception:
            return None

    @staticmethod
    def fuzzy_match(str1: str, str2: str, threshold: float = 0.75) -> bool:
        """
        Fuzzy string matching using SequenceMatcher.

        Returns:
            True if similarity >= threshold
        """
        if not str1 or not str2:
            return False

        s1 = str1.lower().strip()
        s2 = str2.lower().strip()

        similarity = SequenceMatcher(None, s1, s2).ratio()
        return similarity >= threshold

    @staticmethod
    def check_value_in_text(text: str, value: float) -> bool:
        """
        Check if numeric value appears in text.
        Handles different formats: scientific notation, percentages, etc.

        Returns:
            True if value found in text
        """
        if not text:
            return False

        text_norm = text.lower().replace(",", "")

        # Generate search patterns for the value
        patterns = []

        # Direct value
        patterns.append(str(value))

        # Scientific notation variations
        if abs(value) < 0.01 or abs(value) > 10000:
            patterns.append(f"{value:e}")
            patterns.append(f"{value:.2e}")
            patterns.append(f"{value:.3e}")

        # Percentage format (if value is between 0 and 1)
        if 0 < value < 1:
            pct = value * 100
            patterns.append(f"{pct}%")
            patterns.append(f"{pct:.1f}%")
            patterns.append(f"{pct:.2f}%")

        # Rounded variations
        patterns.append(f"{value:.1f}")
        patterns.append(f"{value:.2f}")
        patterns.append(f"{value:.3f}")

        # Check each pattern
        for pattern in patterns:
            if str(pattern).lower() in text_norm:
                return True

        return False

    @staticmethod
    def create_mock_species(species_units: dict, ureg) -> dict:
        """
        Create mock species data from species_units dict.

        Args:
            species_units: Dict mapping species names to unit info (str or dict with 'units' key)
            ureg: Pint UnitRegistry

        Returns:
            Dict mapping species names to mock Pint quantities
        """
        mock_species = {}
        for species, unit_info in species_units.items():
            # Handle both old format (string) and new format (dict with 'units' key)
            if isinstance(unit_info, dict):
                unit_str = unit_info.get("units", "dimensionless")
            else:
                unit_str = unit_info

            # Infer reasonable mock values based on unit type
            if "cell" in unit_str:
                value = 1e6
            elif "molarity" in unit_str:
                value = 1.0
            elif "gram" in unit_str:
                value = 100.0
            else:
                value = 1.0
            mock_species[species] = np.ones(10) * value * ureg(unit_str)
        return mock_species

    # ========================================================================
    # Field definitions
    # ========================================================================

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

    @model_validator(mode="after")
    def validate_threshold_inputs(self) -> "CalibrationTarget":
        """
        Validate that threshold_input_name references exist and match threshold values.

        Checks:
        1. Each measurement's threshold_input_name exists in inputs list
        2. Referenced input has matching value and units
        3. Warns if threshold input uses modeling_assumption
        """
        inputs_dict = {inp.name: inp for inp in self.calibration_target_estimates.inputs}

        for measurement in self.scenario.measurements:
            threshold_input_name = measurement.threshold_input_name

            # Check 1: Input exists
            if threshold_input_name not in inputs_dict:
                raise ValueError(
                    f"Measurement threshold_input_name '{threshold_input_name}' not found in inputs list. "
                    f"Available inputs: {list(inputs_dict.keys())}"
                )

            threshold_input = inputs_dict[threshold_input_name]

            # Check 2: Value and units match
            if abs(threshold_input.value - measurement.threshold) > 1e-9:
                raise ValueError(
                    f"Threshold input '{threshold_input_name}' value ({threshold_input.value}) "
                    f"does not match measurement threshold ({measurement.threshold})"
                )

            if threshold_input.units != measurement.threshold_units:
                raise ValueError(
                    f"Threshold input '{threshold_input_name}' units ('{threshold_input.units}') "
                    f"do not match measurement threshold_units ('{measurement.threshold_units}')"
                )

            # Check 3: Reject if modeling_assumption
            if threshold_input.source_ref == "modeling_assumption":
                raise ValueError(
                    f"Threshold input '{threshold_input_name}' uses 'modeling_assumption' as source_ref. "
                    f"Threshold values MUST be extracted from the same paper as the calibration target, not assumed. "
                    f"Search the paper for patient demographics, cohort characteristics, or enrollment criteria."
                )

        return self

    @model_validator(mode="after")
    def validate_doi_resolution(self) -> "CalibrationTarget":
        """Validator: Check that primary DOI resolves via CrossRef."""
        if self.primary_data_source.doi:
            metadata = self.resolve_doi(self.primary_data_source.doi)
            if metadata is None:
                raise ValueError(
                    f"DOI '{self.primary_data_source.doi}' failed to resolve via CrossRef. "
                    "Verify the DOI exists and is correctly formatted (e.g., '10.1234/journal.2023.123'). "
                    "Search for the paper on Google Scholar or PubMed to find the correct DOI."
                )
        return self

    @model_validator(mode="after")
    def validate_title_match(self) -> "CalibrationTarget":
        """Validator: Check that paper title matches CrossRef metadata."""
        if self.primary_data_source.doi:
            metadata = self.resolve_doi(self.primary_data_source.doi)
            if metadata:
                crossref_title = metadata.get("title", "")
                if not self.fuzzy_match(
                    crossref_title, self.primary_data_source.title, threshold=0.75
                ):
                    raise ValueError(
                        f"Paper title mismatch:\n"
                        f"  CrossRef: '{crossref_title}'\n"
                        f"  Provided: '{self.primary_data_source.title}'\n"
                        f"Use the exact title from the DOI. Copy the title from CrossRef or the paper itself."
                    )
        return self

    @model_validator(mode="after")
    def validate_measurement_code_units(self, info: ValidationInfo) -> "CalibrationTarget":
        """
        Validator: Check that measurement_code returns correct units.

        Requires context:
            species_units: Dict mapping species names to unit strings (from species_units.json)
        """
        from qsp_llm_workflows.core.unit_registry import ureg

        # Get species_units from context
        if not info.context:
            raise ValueError(
                "Validation context is required. Pass context={'species_units': {...}}"
            )
        species_units = info.context["species_units"]

        for measurement in self.scenario.measurements:
            # Parse the code
            try:
                tree = ast.parse(measurement.measurement_code)
            except SyntaxError as e:
                raise ValueError(f"measurement_code has syntax error: {e}")

            # Find the function definition
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "compute_measurement":
                    func_def = node
                    break

            if not func_def:
                raise ValueError(
                    "measurement_code must define a function named 'compute_measurement'"
                )

            # Check signature
            args = [arg.arg for arg in func_def.args.args]
            if args != ["time", "species_dict", "ureg"]:
                raise ValueError(
                    f"Function signature must be (time, species_dict, ureg), got ({', '.join(args)})"
                )

            # Execute with mock data to test output units
            try:
                # Create mock data from actual model species
                time = np.linspace(0, 14, 10) * ureg.day
                mock_species = self.create_mock_species(species_units, ureg)

                # Execute function
                local_scope = {"ureg": ureg, "np": np}
                exec(measurement.measurement_code, local_scope)
                compute_fn = local_scope["compute_measurement"]

                result = compute_fn(time, mock_species, ureg)

                # Check result has units
                if not hasattr(result, "units"):
                    raise ValueError("Function must return a Pint Quantity with units")

                # Check dimensionality matches
                expected_quantity = 1.0 * ureg(self.calibration_target_estimates.units)
                if not result.dimensionality == expected_quantity.dimensionality:
                    raise ValueError(
                        f"Unit dimensionality mismatch:\n"
                        f"  Expected: {self.calibration_target_estimates.units} ({expected_quantity.dimensionality})\n"
                        f"  Got: {result.units} ({result.dimensionality})\n"
                        f"Ensure measurement_code returns the same units as the calibration target estimate."
                    )

            except Exception as e:
                if "dimensionality mismatch" in str(e) or "Unit" in str(e):
                    raise  # Re-raise unit errors
                # Other execution errors might be due to missing species - be lenient
                pass

        return self

    @model_validator(mode="after")
    def validate_derivation_code(self) -> "CalibrationTarget":
        """
        Validator: Check that derivation_code executes, returns correct structure and units,
        and computed values match reported values.
        """
        from qsp_llm_workflows.core.unit_registry import ureg

        # Parse the code
        try:
            tree = ast.parse(self.calibration_target_estimates.distribution_code)
        except SyntaxError as e:
            raise ValueError(f"distribution_code has syntax error: {e}")

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "derive_distribution":
                func_def = node
                break

        if not func_def:
            raise ValueError("distribution_code must define a function named 'derive_distribution'")

        # Check signature
        args = [arg.arg for arg in func_def.args.args]
        if args != ["inputs", "ureg"]:
            raise ValueError(f"Function signature must be (inputs, ureg), got ({', '.join(args)})")

        # Execute with mock inputs to test structure and units
        try:
            # Create mock inputs dict with Pint quantities
            mock_inputs = {}
            for inp in self.calibration_target_estimates.inputs:
                mock_inputs[inp.name] = inp.value * ureg(inp.units)

            # Execute function
            local_scope = {"ureg": ureg, "np": np}
            exec(self.calibration_target_estimates.distribution_code, local_scope)
            derive_fn = local_scope["derive_distribution"]

            result = derive_fn(mock_inputs, ureg)

            # Check result is dict with required keys
            if not isinstance(result, dict):
                raise ValueError("Function must return a dict")

            required_keys = {"median_obs", "iqr_obs", "ci95_obs"}
            if not required_keys.issubset(result.keys()):
                raise ValueError(
                    f"Function must return dict with keys: {required_keys}. Got: {result.keys()}"
                )

            # Check all values have units
            for key in required_keys:
                if not hasattr(result[key], "units"):
                    raise ValueError(f"Result['{key}'] must be a Pint Quantity with units")

            # Check dimensionality matches expected units
            expected_quantity = 1.0 * ureg(self.calibration_target_estimates.units)

            # Check median
            if result["median_obs"].dimensionality != expected_quantity.dimensionality:
                raise ValueError(
                    f"median_obs unit dimensionality mismatch:\n"
                    f"  Expected: {self.calibration_target_estimates.units} ({expected_quantity.dimensionality})\n"
                    f"  Got: {result['median_obs'].units} ({result['median_obs'].dimensionality})"
                )

            # Check iqr
            if result["iqr_obs"].dimensionality != expected_quantity.dimensionality:
                raise ValueError(
                    f"iqr_obs unit dimensionality mismatch:\n"
                    f"  Expected: {self.calibration_target_estimates.units}\n"
                    f"  Got: {result['iqr_obs'].units}"
                )

            # Check ci95 is list/array of 2 elements
            if not hasattr(result["ci95_obs"], "__len__") or len(result["ci95_obs"]) != 2:
                raise ValueError("ci95_obs must be a list/array with 2 elements [lower, upper]")

            # Convert computed values to expected units for comparison
            median_computed = result["median_obs"].to(self.calibration_target_estimates.units)
            iqr_computed = result["iqr_obs"].to(self.calibration_target_estimates.units)
            ci95_computed = [
                result["ci95_obs"][0].to(self.calibration_target_estimates.units),
                result["ci95_obs"][1].to(self.calibration_target_estimates.units),
            ]

            # Check computed values match reported values (with tolerance)
            median_reported = self.calibration_target_estimates.median
            iqr_reported = self.calibration_target_estimates.iqr
            ci95_reported = self.calibration_target_estimates.ci95

            # Use relative tolerance of 1% for comparison
            rel_tol = 0.01

            if abs(median_computed.magnitude - median_reported) > rel_tol * abs(median_reported):
                raise ValueError(
                    f"Computed median ({median_computed.magnitude:.4g}) does not match "
                    f"reported median ({median_reported:.4g}) within 1% tolerance"
                )

            if abs(iqr_computed.magnitude - iqr_reported) > rel_tol * abs(iqr_reported):
                raise ValueError(
                    f"Computed IQR ({iqr_computed.magnitude:.4g}) does not match "
                    f"reported IQR ({iqr_reported:.4g}) within 1% tolerance"
                )

            # Check CI95 bounds
            if abs(ci95_computed[0].magnitude - ci95_reported[0]) > rel_tol * abs(ci95_reported[0]):
                raise ValueError(
                    f"Computed CI95 lower ({ci95_computed[0].magnitude:.4g}) does not match "
                    f"reported CI95 lower ({ci95_reported[0]:.4g}) within 1% tolerance"
                )

            if abs(ci95_computed[1].magnitude - ci95_reported[1]) > rel_tol * abs(ci95_reported[1]):
                raise ValueError(
                    f"Computed CI95 upper ({ci95_computed[1].magnitude:.4g}) does not match "
                    f"reported CI95 upper ({ci95_reported[1]:.4g}) within 1% tolerance"
                )

        except Exception as e:
            if "mismatch" in str(e) or "does not match" in str(e):
                raise  # Re-raise validation errors
            # Other execution errors - provide context
            raise ValueError(f"Error executing derivation_code: {e}")

        return self

    @model_validator(mode="after")
    def validate_source_refs(self) -> "CalibrationTarget":
        """Validator: Check all source_refs in inputs point to defined sources."""
        # Build set of valid source tags
        valid_tags = {self.primary_data_source.source_tag}
        valid_tags.update(s.source_tag for s in self.secondary_data_sources)
        valid_tags.add("modeling_assumption")  # Special tag

        # Check each input's source_ref
        for inp in self.calibration_target_estimates.inputs:
            if inp.source_ref not in valid_tags:
                raise ValueError(
                    f"Input '{inp.name}' has source_ref '{inp.source_ref}' which is not defined.\n"
                    f"Valid source tags: {sorted(valid_tags)}\n"
                    f"Add the source to primary_data_source or secondary_data_sources, "
                    f"or use 'modeling_assumption' if appropriate."
                )

        return self

    @model_validator(mode="after")
    def validate_species_exist(self, info: ValidationInfo) -> "CalibrationTarget":
        """
        Validator: Check that trigger_species and measurement_species exist in model.

        Requires context:
            species_units: Dict mapping species names to unit strings (from species_units.json)
        """
        # Get species_units from context
        if not info.context:
            raise ValueError(
                "Validation context is required. Pass context={'species_units': {...}}"
            )
        species_units = info.context["species_units"]
        available_species = set(species_units.keys())

        for measurement in self.scenario.measurements:
            # Check trigger_species exists
            if measurement.trigger_species not in available_species:
                raise ValueError(
                    f"trigger_species '{measurement.trigger_species}' not found in model.\n"
                    f"Available species: {sorted(available_species)}\n"
                    f"Check species name format (should be compartment.species, e.g., 'V_T.C1', 'V_T.CD8')"
                )

            # Check all measurement_species exist
            for species in measurement.measurement_species:
                if species not in available_species:
                    raise ValueError(
                        f"measurement_species '{species}' not found in model.\n"
                        f"Available species: {sorted(available_species)}\n"
                        f"Check species name format (should be compartment.species, e.g., 'V_T.C1', 'V_T.CD8')"
                    )

        return self

    @model_validator(mode="after")
    def validate_threshold_conversion_code(self, info: ValidationInfo) -> "CalibrationTarget":
        """
        Validator: Check that threshold_conversion_code executes and returns correct units.

        Requires context:
            species_units: Dict mapping species names to unit strings (from species_units.json)
        """
        from qsp_llm_workflows.core.unit_registry import ureg

        # Get species_units from context
        if not info.context:
            raise ValueError(
                "Validation context is required. Pass context={'species_units': {...}}"
            )
        species_units = info.context["species_units"]

        for measurement in self.scenario.measurements:
            if not measurement.threshold_conversion_code:
                continue  # Identity mapping, no code to validate

            # Parse the code
            try:
                tree = ast.parse(measurement.threshold_conversion_code)
            except SyntaxError as e:
                raise ValueError(f"threshold_conversion_code has syntax error: {e}")

            # Find the function definition
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "compute_threshold_value":
                    func_def = node
                    break

            if not func_def:
                raise ValueError(
                    "threshold_conversion_code must define a function named 'compute_threshold_value'"
                )

            # Check signature
            args = [arg.arg for arg in func_def.args.args]
            if args != ["species_dict", "inputs", "ureg"]:
                raise ValueError(
                    f"Function signature must be (species_dict, inputs, ureg), got ({', '.join(args)})"
                )

            # Execute with mock data
            try:
                # Create mock species from actual model
                mock_species = self.create_mock_species(species_units, ureg)

                mock_inputs = {}
                for inp in self.calibration_target_estimates.inputs:
                    mock_inputs[inp.name] = inp.value * ureg(inp.units)

                # Execute function
                local_scope = {"ureg": ureg, "np": np}
                exec(measurement.threshold_conversion_code, local_scope)
                compute_fn = local_scope["compute_threshold_value"]

                result = compute_fn(mock_species, mock_inputs, ureg)

                # Check result has units
                if not hasattr(result, "units"):
                    raise ValueError("Function must return a Pint Quantity with units")

                # Check dimensionality matches threshold units
                expected_quantity = 1.0 * ureg(measurement.threshold_units)
                if result.dimensionality != expected_quantity.dimensionality:
                    raise ValueError(
                        f"Threshold computation unit dimensionality mismatch:\n"
                        f"  Expected: {measurement.threshold_units} ({expected_quantity.dimensionality})\n"
                        f"  Got: {result.units} ({result.dimensionality})"
                    )

            except Exception as e:
                if "dimensionality mismatch" in str(e):
                    raise
                # Other errors might be due to missing species/inputs - be lenient
                pass

        return self

    @model_validator(mode="after")
    def validate_inputs_used(self) -> "CalibrationTarget":
        """Validator: Warn if inputs are defined but not used in derivation_code."""
        # Parse derivation_code to find variable references
        try:
            tree = ast.parse(self.calibration_target_estimates.distribution_code)
        except SyntaxError:
            # Already caught by validate_derivation_code
            return self

        # Extract all names accessed from 'inputs' dict
        used_input_names = set()

        class InputAccessVisitor(ast.NodeVisitor):
            def visit_Subscript(self, node):
                # Look for inputs['name'] or inputs["name"]
                if isinstance(node.value, ast.Name) and node.value.id == "inputs":
                    if isinstance(node.slice, ast.Constant):
                        used_input_names.add(node.slice.value)
                self.generic_visit(node)

        visitor = InputAccessVisitor()
        visitor.visit(tree)

        # Check which defined inputs are not used
        defined_inputs = {inp.name for inp in self.calibration_target_estimates.inputs}
        unused_inputs = defined_inputs - used_input_names

        if unused_inputs:
            # This is a warning, not an error - could be intentional
            import warnings

            warnings.warn(
                f"The following inputs are defined but not used in derivation_code: {sorted(unused_inputs)}. "
                f"If these inputs are not needed, consider removing them.",
                UserWarning,
            )

        return self
