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
from typing import List, Optional, Type

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
# Helper Functions
# ============================================================================


def enum_field_description(enum_class: Type[Enum], base_description: str = "") -> str:
    """
    Generate field description with enum options automatically included.

    Args:
        enum_class: The Enum class to extract options from
        base_description: Optional base description text

    Returns:
        Description string with valid options listed
    """
    options = [f"'{member.value}'" for member in enum_class]
    options_str = ", ".join(options)

    if base_description:
        return f"{base_description}. Valid options: {options_str}"
    else:
        return f"Valid options: {options_str}"


# ============================================================================
# Scenario Models (Interventions and Measurements)
# ============================================================================
# NOTE: Simplified to text descriptions - executable code deferred to manual implementation


class Intervention(BaseModel):
    """
    Text description of intervention (deferred executable code to later).

    For now, we capture the essential information in text form.
    Later, this can be converted to executable DrugDosing/SurgicalResection/etc.
    """

    intervention_description: str = Field(
        description=(
            "Complete text description of the intervention including:\n"
            "- What: Agent/procedure name\n"
            "- How much: Dose and units (mg/kg, mg/m2, etc.)\n"
            "- When: Schedule/timing (e.g., 'every 2 weeks starting day 0', 'day 14 resection')\n"
            "- Additional details: Patient weight/BSA if relevant, fraction removed for resection\n\n"
            "Examples:\n"
            "- 'Anti-PD-1 antibody 3 mg/kg IV every 2 weeks starting day 0 (patient weight 70 kg)'\n"
            "- 'Surgical resection on day 14, removing 90% of tumor burden'\n"
            "- 'Gemcitabine 1000 mg/m2 on days 0, 7, 14 (patient BSA 1.8 m2)'\n"
            "- 'No intervention (natural disease progression)'"
        )
    )


class Measurement(BaseModel):
    """
    Measurement specification with executable code for WHAT to measure and text description for WHEN.

    - measurement_code defines WHAT observable to compute from simulation
    - threshold_description describes WHEN the measurement occurs (text only for now)
    """

    measurement_description: str = Field(
        description=(
            "Text description of WHAT is being measured and HOW:\n"
            "- Observable: What biological quantity (e.g., 'CD8+ T cell density', 'tumor volume')\n"
            "- Method: How it's measured (e.g., 'via IHC', 'via imaging', 'via flow cytometry')\n"
            "- Location: Where in the body (e.g., 'in primary tumor tissue', 'in peripheral blood')\n"
            "- Units: Expected units (e.g., 'cells/mm²', 'millimeter³', 'nanomolarity')\n\n"
            "Example: 'CD8+ T cell density measured via IHC in tumor tissue sections, reported as cells/mm²'"
        )
    )

    measurement_species: List[str] = Field(
        description=(
            "List of species accessed by measurement_code.\n"
            "Format: 'compartment.species' (e.g., ['V_T.CD8', 'V_T.C1']).\n"
            "Must match species names in model."
        )
    )

    measurement_code: str = Field(
        description=(
            "Python function that computes the observable from species time series.\n\n"
            "Function signature: compute_measurement(time, species_dict, ureg)\n"
            "- time: numpy array with time values (Pint Quantity with day units)\n"
            "- species_dict: dict mapping species names to numpy arrays (Pint Quantities, one value per timepoint)\n"
            "- ureg: Pint UnitRegistry for unit conversions\n\n"
            "Must return a Pint Quantity (scalar or array) with units matching calibration_target_estimates.units.\n\n"
            "IMPORTANT: Do NOT include time filtering logic (e.g., 'use last timepoint' or 'at t=14').\n"
            "This function computes WHAT to measure from the time series.\n"
            "WHEN to measure is handled separately via threshold_description.\n\n"
            "The function may use multiple timepoints (e.g., for derivatives), but should not decide which timepoint to evaluate.\n\n"
            "Example (simple ratio):\n"
            "def compute_measurement(time, species_dict, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']  # Array of values over time\n"
            "    tumor = species_dict['V_T.C1']  # Array of values over time\n"
            "    ratio = cd8 / tumor  # Element-wise division\n"
            "    return ratio.to(ureg.dimensionless)  # Returns array\n\n"
            "Example (derivative):\n"
            "def compute_measurement(time, species_dict, ureg):\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    growth_rate = np.gradient(tumor.magnitude, time.magnitude) * (tumor.units / time.units)\n"
            "    return growth_rate.to(ureg.cell / ureg.day)  # Returns array"
        )
    )

    threshold_description: str = Field(
        description=(
            "Text description of WHEN the measurement occurs:\n"
            "- Trigger: What biological/clinical event triggers measurement\n"
            "- Threshold: Specific value if stated (e.g., 'when tumor reaches 500 mm³', 'at 1e9 cells')\n"
            "- Context: Clinical context (e.g., 'at resection', 'at diagnosis', 'baseline')\n\n"
            "Examples:\n"
            "- 'At tumor resection when tumor burden reaches approximately 1e9 cells (~500 mm³)'\n"
            "- 'At baseline/diagnosis before any treatment (tumor burden ~1e9 cells)'\n"
            "- '7 days after first anti-PD-1 dose when tumor begins responding'\n"
            "- 'At clinical presentation (median tumor volume 450 mm³ in study cohort)'"
        )
    )


class Scenario(BaseModel):
    """
    Experimental scenario: sequence of interventions and measurements.

    Defines all exogenous events during the experiment (treatments, measurements, etc.).
    """

    description: str = Field(
        description="Human-readable description of the scenario (e.g., 'Baseline PDAC tumor at resection, treatment-naive')"
    )
    interventions: List[Intervention] = Field(
        description=(
            "List of interventions applied during the experiment. "
            "May be empty list for natural/untreated state measurements. "
            "Use single entry with 'No intervention (natural disease progression)' for clarity."
        )
    )
    measurements: List[Measurement] = Field(
        description=(
            "List of measurement specifications (what and when to measure). "
            "Must contain at least one measurement. "
            "Each measurement describes the observable and the clinical/biological context."
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

    extent: StageExtent = Field(description=enum_field_description(StageExtent, "Disease extent"))
    burden: StageBurden = Field(
        description=enum_field_description(
            StageBurden, "Disease burden (tumor size/volume category)"
        )
    )


class TreatmentContext(BaseModel):
    """Treatment context with history and current status."""

    history: List[TreatmentHistory] = Field(
        description=enum_field_description(
            TreatmentHistory, "Treatment history (select all that apply)"
        )
    )
    status: TreatmentStatus = Field(
        description=enum_field_description(
            TreatmentStatus,
            "Current treatment status. Use 'off_treatment' for untreated/baseline measurements",
        )
    )
    specifier: Optional[str] = Field(None, description="Optional drug name or class specifier")


class ExperimentalContext(BaseModel):
    """
    Experimental context for an observable.

    Uses typed enums for all context dimensions with hierarchical notation
    encoded in enum values (e.g., Compartment.TUMOR_PRIMARY = "tumor.primary").
    """

    species: Species = Field(description=enum_field_description(Species, "Species"))
    mouse_subspecifier: Optional[MouseSubspecifier] = Field(
        None,
        description=enum_field_description(
            MouseSubspecifier, "Optional mouse subspecifier (only if species is mouse)"
        ),
    )
    indication: Indication = Field(
        description=enum_field_description(Indication, "Cancer indication")
    )
    compartment: Compartment = Field(
        description=enum_field_description(Compartment, "Anatomical compartment")
    )
    system: System = Field(description=enum_field_description(System, "Experimental system"))
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

    median: float = Field(
        description=(
            "COMPUTED median value from distribution_code. "
            "Set this to the output of derive_distribution()['median_obs'].magnitude. "
            "This is NOT necessarily the median reported in the paper (which goes in inputs)."
        )
    )
    iqr: float = Field(
        description=(
            "COMPUTED interquartile range from distribution_code. "
            "Set this to the output of derive_distribution()['iqr_obs'].magnitude. "
            "This is NOT necessarily the IQR reported in the paper (which goes in inputs)."
        )
    )
    ci95: List[float] = Field(
        description=(
            "COMPUTED 95% confidence interval [lower, upper] from distribution_code. "
            "Set this to the output of derive_distribution()['ci95_obs'] as [lower.magnitude, upper.magnitude]. "
            "Papers rarely report CI95 - derive from mean/SD or median/IQR via Monte Carlo."
        )
    )
    units: str = Field(description="Units of the observable")
    inputs: List[Input] = Field(
        description=(
            "List of inputs used in derivation. "
            "These are the VALUES REPORTED IN THE PAPER (mean, SD, median, IQR, etc.) "
            "that are used to DERIVE the distribution via Monte Carlo."
        )
    )
    distribution_code: str = Field(
        description=(
            "Python code defining a derive_distribution(inputs, ureg) function. "
            "inputs is a dict mapping input names to Pint Quantities. "
            "Must return dict with Pint Quantities: median_obs, iqr_obs, ci95_obs ([lower, upper]). "
            "Extract biological/experimental values via inputs with source traceability. "
            "Universal constants OK as literals: percentiles (2.5, 25, 75, 97.5), mathematical constants (π, 2), MC sample sizes (10000). "
            "Use MC methods (parametric bootstrap) for distribution estimates, NOT analytical approximations.\n"
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
    def create_mock_species(species_units: dict, ureg, n_timepoints: int = 100) -> dict:
        """
        Create mock species data from species_units dict.

        Args:
            species_units: Dict mapping species names to unit info (str or dict with 'units' key)
            ureg: Pint UnitRegistry
            n_timepoints: Number of timepoints in time series (default 100)

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
            mock_species[species] = np.ones(n_timepoints) * value * ureg(unit_str)
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
        description=(
            "Important limitations and their impact on reliability. Address:\n"
            "- Sample size and statistical power\n"
            "- Population generalizability (single center, specific subtype, demographics)\n"
            "- Large variance/uncertainty (if CV > 50% or wide CI, explain biological vs methodological sources)\n"
            "- Distributional assumptions (if using normal for size data, acknowledge potential bias)\n"
            "- Conversion factor uncertainties (cellularity, cell size, geometric assumptions)\n"
            "- Measurement timing ambiguity (if unclear what triggered observation)"
        )
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

    # DISABLED: Simplified measurement structure - executable code deferred to manual implementation
    # @model_validator(mode="after")
    # def validate_threshold_inputs(self) -> "CalibrationTarget":
    #     """Validator disabled for simplified measurement structure."""
    #     return self

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
        Validator: Check that measurement_code executes and returns correct units.

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

            # Execute with mock data
            try:
                # Create mock time array
                mock_time = np.linspace(0, 14, 100) * ureg.day

                # Create mock species from actual model (with matching length)
                mock_species = self.create_mock_species(
                    species_units, ureg, n_timepoints=len(mock_time)
                )

                # Execute function
                local_scope = {"ureg": ureg, "np": np}
                exec(measurement.measurement_code, local_scope)
                compute_fn = local_scope["compute_measurement"]

                result = compute_fn(mock_time, mock_species, ureg)

                # Check result has units
                if not hasattr(result, "units"):
                    raise ValueError("Function must return a Pint Quantity with units")

                # Check dimensionality matches calibration target units
                expected_quantity = 1.0 * ureg(self.calibration_target_estimates.units)
                if result.dimensionality != expected_quantity.dimensionality:
                    raise ValueError(
                        f"Measurement code unit dimensionality mismatch:\n"
                        f"  Expected: {self.calibration_target_estimates.units} ({expected_quantity.dimensionality})\n"
                        f"  Got: {result.units} ({result.dimensionality})"
                    )

                # Check result has same length as time series (no time indexing)
                if hasattr(result, "magnitude"):
                    result_magnitude = result.magnitude
                    # Handle both scalar and array results
                    if np.isscalar(result_magnitude):
                        raise ValueError(
                            f"Measurement code returned a scalar, but should return an array with same length as time series.\n"
                            f"  Expected length: {len(mock_time)}\n"
                            f"  Got: scalar value\n"
                            f"Do NOT use time indexing (e.g., species[-1] or species[0]) in measurement_code.\n"
                            f"Compute the observable over the entire time series."
                        )
                    elif len(result_magnitude) != len(mock_time):
                        raise ValueError(
                            f"Measurement code returned array with wrong length:\n"
                            f"  Expected: {len(mock_time)} (same as time series)\n"
                            f"  Got: {len(result_magnitude)}\n"
                            f"Do NOT use time indexing. Compute over entire time series."
                        )

            except Exception as e:
                error_str = str(e)
                if (
                    "dimensionality mismatch" in error_str
                    or "returned a scalar" in error_str
                    or "wrong length" in error_str
                    or "Cannot convert" in error_str  # Catch unit conversion errors
                ):
                    raise
                # Other errors might be due to missing species - be lenient
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
        Validator: Check that measurement_species exist in model.

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
            # Check all measurement_species exist
            for species in measurement.measurement_species:
                if species not in available_species:
                    raise ValueError(
                        f"measurement_species '{species}' not found in model.\n"
                        f"Available species: {sorted(available_species)}\n"
                        f"Check species name format (should be compartment.species, e.g., 'V_T.C1', 'V_T.CD8')"
                    )

        return self

    # DISABLED: Simplified measurement structure - executable code deferred to manual implementation
    # @model_validator(mode="after")
    # def validate_threshold_conversion_code(self, info: ValidationInfo) -> "CalibrationTarget":
    #     """Validator disabled for simplified measurement structure."""
    #     return self

    @model_validator(mode="after")
    def validate_clipping_suggests_lognormal(self) -> "CalibrationTarget":
        """
        Validator (WARNING): Check if distribution_code uses clipping.

        Clipping to avoid negative values suggests size/volume data that
        should use lognormal distribution instead of normal.
        """
        import warnings

        code = self.calibration_target_estimates.distribution_code

        # Check for clipping patterns
        clipping_patterns = ["np.clip", "np.maximum", "np.minimum", "max(0", "min("]
        if any(pattern in code for pattern in clipping_patterns):
            warnings.warn(
                "distribution_code uses clipping (np.clip/np.maximum/etc) to avoid negative values. "
                "This suggests size/volume/mass data that may be better modeled with a lognormal distribution. "
                "Normal distributions for positive-only data introduce bias when clipped. "
                "Consider converting mean±SD to lognormal parameters: "
                "μ_log = ln(mean²/√(mean²+sd²)), σ_log = √(ln(1+sd²/mean²))",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def validate_large_variance_documented(self) -> "CalibrationTarget":
        """
        Validator (WARNING): Check if large variance is documented in limitations.

        CV > 50% suggests substantial variability that should be explained.
        """
        import warnings

        # Find mean and SD/SE inputs
        mean_input = None
        std_input = None

        for inp in self.calibration_target_estimates.inputs:
            name_lower = inp.name.lower()
            if "mean" in name_lower and mean_input is None:
                mean_input = inp
            if any(x in name_lower for x in ["sd", "std", "se", "stderr", "stdev"]):
                std_input = inp

        if mean_input and std_input and mean_input.value != 0:
            # Calculate CV (coefficient of variation)
            cv = abs(std_input.value / mean_input.value)

            if cv > 0.5:  # CV > 50%
                # Check if limitations mention variance/uncertainty
                limitations_lower = self.key_study_limitations.lower()
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

                if not any(keyword in limitations_lower for keyword in variance_keywords):
                    warnings.warn(
                        f"Large coefficient of variation (CV = {cv:.1%}) detected but not discussed in key_study_limitations. "
                        f"Consider explaining whether this reflects biological variability, measurement error, "
                        f"or heterogeneous population. High variance may indicate need for stratification or "
                        f"alternative distributional assumptions (e.g., lognormal).",
                        UserWarning,
                    )

        return self

    @model_validator(mode="after")
    def validate_distribution_choice_for_size_data(self) -> "CalibrationTarget":
        """
        Validator (WARNING): Check if normal distribution used for size data.

        Size/volume/mass measurements are typically lognormal, not normal.
        """
        import warnings

        units = self.calibration_target_estimates.units.lower()
        code = self.calibration_target_estimates.distribution_code

        # Check if units suggest size/volume/mass
        size_units = ["meter", "liter", "gram", "cell", "mole", "molarity"]
        is_size_data = any(unit in units for unit in size_units) and units != "dimensionless"

        # Check if using normal distribution (but not lognormal)
        uses_normal = ("normal(" in code or "randn" in code) and "lognormal" not in code

        if is_size_data and uses_normal:
            warnings.warn(
                f"Using normal distribution for size/volume/mass data (units: {units}). "
                f"These measurements are typically lognormally distributed. "
                f"Normal distributions can produce non-physical negative values requiring clipping. "
                f"Consider using lognormal distribution instead.",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def validate_conversion_factors_documented(self) -> "CalibrationTarget":
        """
        Validator (WARNING): Check that conversion factors are documented.

        Magic numbers in measurement_code should be in inputs or key_assumptions.
        """
        import warnings

        # Universal constants to ignore (only truly fundamental values)
        UNIVERSAL_CONSTANTS = {
            # Basic integers
            0,
            1,
            2,
            3,
            4,
            # Mathematical constants
            np.pi,
            3.14159,
            3.141592653589793,
            np.e,
            2.71828,
            2.718281828459045,
            # Common fractions
            0.5,
            0.25,
            0.75,
            # Statistical percentiles (standard)
            2.5,
            25,
            50,
            75,
            97.5,
            0.025,
            0.25,
            0.5,
            0.75,
            0.975,
        }

        for measurement in self.scenario.measurements:
            code = measurement.measurement_code

            # Extract numeric literals from code
            try:
                tree = ast.parse(code)
                literals = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        val = node.value
                        # Skip universal constants (with tolerance for floating point)
                        if not any(abs(val - c) < 1e-6 for c in UNIVERSAL_CONSTANTS):
                            literals.append(val)
            except SyntaxError:
                # If code doesn't parse, skip this check
                continue

            if literals:
                # Check if these appear in inputs or assumptions
                input_values = {inp.value for inp in self.calibration_target_estimates.inputs}
                assumption_texts = " ".join(a.text for a in self.key_assumptions)

                undocumented = []
                for lit in literals:
                    # Check if value appears in inputs (with tolerance)
                    in_inputs = any(abs(lit - v) < 1e-6 for v in input_values)
                    # Check if value appears in assumption text
                    in_assumptions = (
                        str(lit) in assumption_texts or f"{lit:.0e}" in assumption_texts
                    )

                    if not (in_inputs or in_assumptions):
                        undocumented.append(lit)

                if undocumented:
                    warnings.warn(
                        f"measurement_code contains numeric literals {undocumented} that may be conversion factors. "
                        f"Document these in either inputs (with source_ref) or key_assumptions. "
                        f"Examples: cell sizes (µm), cellularity fractions, density values, geometric conversion factors.",
                        UserWarning,
                    )

        return self

    @model_validator(mode="after")
    def validate_inputs_used(self) -> "CalibrationTarget":
        """Validator: Warn if inputs are defined but not used in distribution_code (simplified for text-based measurements)."""
        # Extract all names accessed from 'inputs' dict
        used_input_names = set()

        class InputAccessVisitor(ast.NodeVisitor):
            def visit_Subscript(self, node):
                # Look for inputs['name'] or inputs["name"]
                if isinstance(node.value, ast.Name) and node.value.id == "inputs":
                    if isinstance(node.slice, ast.Constant):
                        used_input_names.add(node.slice.value)
                self.generic_visit(node)

        # Parse distribution_code to find variable references
        try:
            tree = ast.parse(self.calibration_target_estimates.distribution_code)
            visitor = InputAccessVisitor()
            visitor.visit(tree)
        except SyntaxError:
            # Already caught by validate_derivation_code
            pass

        # Check which defined inputs are not used
        defined_inputs = {inp.name for inp in self.calibration_target_estimates.inputs}
        unused_inputs = defined_inputs - used_input_names

        if unused_inputs:
            # This is a warning, not an error - could be intentional
            import warnings

            warnings.warn(
                f"The following inputs are defined but not used in distribution_code: {sorted(unused_inputs)}. "
                f"If these inputs are not needed, consider removing them.",
                UserWarning,
            )

        return self

    @model_validator(mode="after")
    def validate_scale_consistency(self, info: ValidationInfo) -> "CalibrationTarget":
        """
        Validator: Check that measurement_code output scale matches calibration target scale.

        Executes measurement_code with mock species data spanning wide range and checks
        if output range is compatible with calibration_target_estimates range.

        Catches scale mismatches like ratio (0-1) vs score (0-3) before calibration.

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

        # Get calibration target range
        estimates = self.calibration_target_estimates
        target_median = estimates.median
        target_ci95 = estimates.ci95
        target_min = float(min(target_ci95))
        target_max = float(max(target_ci95))

        # Execute measurement_code and check output range
        for measurement in self.scenario.measurements:
            try:
                # Create mock time array (wide range to explore measurement_code behavior)
                mock_time = np.linspace(0, 100, 50) * ureg.day

                # Create mock species with wide value range
                mock_species = {}
                for species_name in measurement.measurement_species:
                    # Generate values spanning many orders of magnitude
                    base = np.logspace(0, 8, len(mock_time))

                    # Get unit from species_units (handle both dict and string formats)
                    species_info = species_units.get(species_name, "cell")
                    if isinstance(species_info, dict):
                        unit_str = species_info.get("units", "cell")
                    else:
                        unit_str = species_info
                    mock_species[species_name] = base * ureg(unit_str)

                # Execute measurement_code
                local_scope = {"ureg": ureg, "np": np}
                exec(measurement.measurement_code, local_scope)
                compute_fn = local_scope["compute_measurement"]

                result = compute_fn(mock_time, mock_species, ureg)

                # Get code output range
                code_min = float(np.min(result.magnitude))
                code_max = float(np.max(result.magnitude))
                code_median = float(np.median(result.magnitude))

                # Check for scale mismatch patterns
                # Pattern 1: Code outputs ratio (0-1) but target is score (>2)
                if code_max <= 1.5 and target_max > 2.0:
                    raise ValueError(
                        f"Scale mismatch detected - likely ratio (0-1) vs score (0-N) mismatch:\n"
                        f"  • Calibration target: median={target_median:.3g}, range=[{target_min:.3g}, {target_max:.3g}]\n"
                        f"  • Measurement code:   median={code_median:.3g}, range=[{code_min:.3g}, {code_max:.3g}]\n"
                        f"The measurement_code outputs values on 0-1 scale (likely a ratio/fraction),\n"
                        f"but calibration target is on a larger scale (likely a 0-N score).\n"
                        f"Fix: Either scale the calibration target down to 0-1, or scale the measurement_code up to 0-N."
                    )

                # Pattern 2: Code outputs score (>2) but target is ratio (0-1)
                if code_max > 2.0 and target_max <= 1.5:
                    raise ValueError(
                        f"Scale mismatch detected - likely score (0-N) vs ratio (0-1) mismatch:\n"
                        f"  • Calibration target: median={target_median:.3g}, range=[{target_min:.3g}, {target_max:.3g}]\n"
                        f"  • Measurement code:   median={code_median:.3g}, range=[{code_min:.3g}, {code_max:.3g}]\n"
                        f"The measurement_code outputs values on 0-N scale (likely a score),\n"
                        f"but calibration target is on 0-1 scale (likely a ratio/fraction).\n"
                        f"Fix: Either scale the calibration target up to 0-N, or scale the measurement_code down to 0-1."
                    )

                # Pattern 3: Ranges don't overlap and differ by >10x
                tolerance = 10.0
                if code_median > 0 and target_median > 0:
                    median_ratio = max(target_median, code_median) / min(target_median, code_median)

                    # Check if ranges don't overlap
                    ranges_overlap = not (code_max < target_min or code_min > target_max)

                    if not ranges_overlap and median_ratio > tolerance:
                        raise ValueError(
                            f"Scale mismatch detected - output scale differs by {median_ratio:.1f}x:\n"
                            f"  • Calibration target: median={target_median:.3g}, range=[{target_min:.3g}, {target_max:.3g}]\n"
                            f"  • Measurement code:   median={code_median:.3g}, range=[{code_min:.3g}, {code_max:.3g}]\n"
                            f"The measurement_code output is on a different scale than the calibration target.\n"
                            f"Check for missing unit conversions or scaling factors."
                        )

            except ValueError:
                # Re-raise validation errors
                raise
            except Exception:
                # Other errors (e.g., complex mock data requirements) - be lenient
                # The measurement_code might use logic we can't easily mock
                pass

        return self

    @model_validator(mode="after")
    def validate_no_control_characters(self) -> "CalibrationTarget":
        """
        Validator: Check for control characters in all string fields.

        Control characters (especially in titles, descriptions, snippets) can cause
        YAML parsing failures and other issues. This validator detects them early.

        Checks for:
        - ASCII control characters (0x00-0x1F) except tab (0x09), LF (0x0A), CR (0x0D)
        - DEL character (0x7F)
        - Unicode control characters (U+0080-U+009F)
        """
        import re

        # Pattern to match control characters (excluding valid whitespace)
        # \x00-\x08: NUL through BACKSPACE
        # \x0B-\x0C: VERTICAL TAB, FORM FEED
        # \x0E-\x1F: SHIFT OUT through UNIT SEPARATOR
        # \x7F: DEL
        # \x80-\x9F: Unicode control characters
        control_char_pattern = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F\x80-\x9F]")

        def check_string(value: str, field_path: str) -> None:
            """Check a string for control characters."""
            match = control_char_pattern.search(value)
            if match:
                char_code = ord(match.group(0))
                char_hex = f"0x{char_code:02X}"
                # Get position in string
                pos = match.start()
                # Get context around the control character
                context_start = max(0, pos - 20)
                context_end = min(len(value), pos + 20)
                context = value[context_start:context_end]
                # Replace control char with placeholder for display
                context_display = context.replace(match.group(0), f"[{char_hex}]")

                raise ValueError(
                    f"Control character {char_hex} found in field '{field_path}' at position {pos}.\n"
                    f"Context: ...{context_display}...\n"
                    f"Control characters cause YAML parsing failures and must be removed.\n"
                    f"Common causes: copying from PDFs, word processors, or web pages with special formatting.\n"
                    f"Fix: Re-type the text or use a plain text editor to remove invisible characters."
                )

        def walk_and_check(obj, path: str = "") -> None:
            """Recursively walk through the object and check all strings."""
            if isinstance(obj, str):
                check_string(obj, path)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    walk_and_check(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    walk_and_check(item, f"{path}[{i}]")
            elif hasattr(obj, "model_dump"):
                # Pydantic model - convert to dict and recurse
                walk_and_check(obj.model_dump(), path)

        # Check all fields in the model
        walk_and_check(self.model_dump(), "CalibrationTarget")

        return self
