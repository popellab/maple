#!/usr/bin/env python3
"""
Enum definitions for calibration targets.

Provides typed enums for experimental context dimensions including species,
indication, compartment, system, treatment status, and disease stage.
"""

from enum import Enum
from typing import Type


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


class SourceType(str, Enum):
    """Type of source from which a value was extracted."""

    TEXT = "text"
    """Value extracted from body text, results section, or abstract."""

    TABLE = "table"
    """Value extracted from a table."""

    FIGURE = "figure"
    """Value extracted from a figure (requires manual/digitizer extraction)."""


class ExtractionMethod(str, Enum):
    """Method used to extract values from figures."""

    MANUAL = "manual"
    """Manual reading from figure axes."""

    DIGITIZER = "digitizer"
    """Generic digitizer software."""

    WEBPLOTDIGITIZER = "webplotdigitizer"
    """WebPlotDigitizer (https://automeris.io/WebPlotDigitizer)."""

    OTHER = "other"
    """Other extraction method (specify in extraction_notes)."""


# =============================================================================
# SOURCE RELEVANCE ENUMS
# =============================================================================


class IndicationMatch(str, Enum):
    """How well the source indication matches the target model indication."""

    EXACT = "exact"
    """Same disease (e.g., PDAC data for PDAC model)."""

    RELATED = "related"
    """Same organ or disease class (e.g., other pancreatic diseases, other adenocarcinomas)."""

    PROXY = "proxy"
    """Different tissue/disease used as mechanistic proxy (e.g., melanoma for PDAC)."""

    UNRELATED = "unrelated"
    """No clear biological connection to target indication."""


class SourceQuality(str, Enum):
    """Quality tier of the primary data source."""

    PRIMARY_HUMAN_CLINICAL = "primary_human_clinical"
    """Peer-reviewed original research with human clinical data."""

    PRIMARY_HUMAN_IN_VITRO = "primary_human_in_vitro"
    """Peer-reviewed original research with human cells/tissue in vitro."""

    PRIMARY_ANIMAL_IN_VIVO = "primary_animal_in_vivo"
    """Peer-reviewed original research with animal in vivo data."""

    PRIMARY_ANIMAL_IN_VITRO = "primary_animal_in_vitro"
    """Peer-reviewed original research with animal cells in vitro."""

    REVIEW = "review_article"
    """Review article summarizing primary data (cite original source if possible)."""

    TEXTBOOK = "textbook"
    """Textbook or reference work."""

    NON_PEER_REVIEWED = "non_peer_reviewed"
    """Non-peer-reviewed source (Wikipedia, preprints, etc.). Avoid if possible."""


class PerturbationType(str, Enum):
    """Type of experimental perturbation in the source study."""

    PHYSIOLOGICAL_BASELINE = "physiological_baseline"
    """Normal/homeostatic conditions without perturbation."""

    PATHOLOGICAL_STATE = "pathological_state"
    """Disease-relevant perturbation or condition."""

    PHARMACOLOGICAL = "pharmacological"
    """Drug-induced perturbation (often supraphysiological concentrations)."""

    GENETIC = "genetic_perturbation"
    """Knockout, knockdown, or overexpression."""


class TMECompatibility(str, Enum):
    """Tumor microenvironment compatibility for immune/stromal parameters."""

    HIGH = "high"
    """Source TME has similar characteristics to target (e.g., both desmoplastic)."""

    MODERATE = "moderate"
    """Some TME differences that may affect parameter values."""

    LOW = "low"
    """Major TME differences (e.g., T cell-permissive model for T cell-excluded tumor)."""
