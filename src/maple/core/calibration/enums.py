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
    """Same disease. Examples:
    - PDAC patient tumor data for PDAC model
    - Human PDAC cell line secretion rates
    """

    RELATED = "related"
    """Same organ or disease class. Examples:
    - Chronic pancreatitis PSC data for PDAC PSC parameters
    - Colorectal adenocarcinoma growth rates for PDAC (both GI adenocarcinomas)
    - Other pancreatic cancers (neuroendocrine) for PDAC
    """

    PROXY = "proxy"
    """Different tissue/disease as mechanistic proxy. Examples:
    - Prostate stromal cells for PDAC CCL2 secretion (different organ, similar cell type)
    - Melanoma MDSC data for PDAC (different cancer, same immune cell type)
    - Lewis lung carcinoma TAM reprogramming for PDAC macrophages
    Requires 3-10x translation uncertainty.
    """

    UNRELATED = "unrelated"
    """No clear biological connection. Examples:
    - Healthy volunteer T cell division for tumor-infiltrating T cells
    - Generic fibroblast data for cancer-associated fibroblasts
    Requires 10-100x translation uncertainty.
    """


class SourceQuality(str, Enum):
    """Quality tier of the primary data source.

    Higher quality sources provide more reliable parameter estimates.
    Always prefer primary peer-reviewed literature over reviews or non-peer-reviewed sources.
    """

    PRIMARY_HUMAN_CLINICAL = "primary_human_clinical"
    """Peer-reviewed original research with human clinical data. Examples:
    - CT imaging of tumor growth in PDAC patients
    - Flow cytometry of human PDAC resections
    - Clinical trial pharmacokinetic data
    """

    PRIMARY_HUMAN_IN_VITRO = "primary_human_in_vitro"
    """Peer-reviewed original research with human cells/tissue in vitro. Examples:
    - Primary human PSC secretion assays
    - Human PDAC organoid drug response
    - Human PBMC T cell division tracking
    """

    PRIMARY_ANIMAL_IN_VIVO = "primary_animal_in_vivo"
    """Peer-reviewed original research with animal in vivo data. Examples:
    - KPC mouse tumor growth curves
    - Adoptive T cell transfer trafficking studies
    - Syngeneic tumor MDSC quantification
    """

    PRIMARY_ANIMAL_IN_VITRO = "primary_animal_in_vitro"
    """Peer-reviewed original research with animal cells in vitro. Examples:
    - Rat PSC activation timecourse
    - Mouse macrophage polarization assays
    - Murine T cell CFSE division tracking
    """

    REVIEW = "review_article"
    """Review article summarizing primary data. Examples:
    - Meta-analysis of tumor doubling times
    - Systematic review of macrophage half-lives
    Note: Cite original source if possible for better traceability.
    """

    TEXTBOOK = "textbook"
    """Textbook or reference work. Examples:
    - Janeway's Immunobiology for T cell biology parameters
    - Standard pharmacology references for drug PK
    """

    NON_PEER_REVIEWED = "non_peer_reviewed"
    """Non-peer-reviewed source. Examples:
    - Wikipedia articles
    - Preprints (bioRxiv, medRxiv)
    - Conference abstracts without full paper
    - Blog posts or grey literature
    AVOID if possible. If used, document rationale and increase uncertainty.
    """


class PerturbationType(str, Enum):
    """Type of experimental perturbation in the source study.

    Important for interpreting whether measured values represent
    physiological baselines, maximal responses, or perturbed states.
    """

    PHYSIOLOGICAL_BASELINE = "physiological_baseline"
    """Normal/homeostatic conditions without perturbation. Examples:
    - Unstimulated cell secretion rates
    - Untreated tumor growth rates
    - Baseline immune cell trafficking
    Best for estimating basal parameter values.
    """

    PATHOLOGICAL_STATE = "pathological_state"
    """Disease-relevant perturbation or condition. Examples:
    - Tumor-associated macrophages (vs normal tissue macrophages)
    - Hypoxia-exposed PSCs (tumor-like conditions)
    - Inflammatory cytokine milieu
    Appropriate when modeling disease state.
    """

    PHARMACOLOGICAL = "pharmacological"
    """Drug-induced perturbation. Examples:
    - 1 mM melatonin-induced PSC death (supraphysiological)
    - IL-12 complex-induced T cell expansion
    - High-dose cytokine stimulation
    Often represents UPPER BOUND on parameter values.
    Must document in perturbation_relevance how this relates to physiological values.
    """

    GENETIC = "genetic_perturbation"
    """Knockout, knockdown, or overexpression. Examples:
    - PDGFB knockout affecting pericyte coverage
    - CAR-T cells (engineered TCR)
    - Constitutively active signaling mutants
    Must document in perturbation_relevance how this relates to wild-type values.
    """


class TMECompatibility(str, Enum):
    """Tumor microenvironment compatibility for immune/stromal parameters.

    Critical for T cell trafficking, macrophage polarization, and stromal parameters.
    PDAC is characterized by: dense desmoplasia, T cell exclusion, high CXCL12,
    immunosuppressive myeloid infiltrate, and hypoxic regions.
    """

    HIGH = "high"
    """Source TME similar to target. Examples for PDAC:
    - KPC mouse model (desmoplastic, T cell-excluded)
    - Human PDAC tissue sections
    - 3D PDAC organoid co-cultures with CAFs
    """

    MODERATE = "moderate"
    """Some TME differences. Examples for PDAC:
    - 4T1 breast cancer (moderately excluded, some desmoplasia)
    - MC38 colon cancer (moderate immune infiltration)
    - 2D PSC cultures (missing 3D matrix context)
    """

    LOW = "low"
    """Major TME differences. Examples for PDAC:
    - EG7/B16-OVA thymoma (highly T cell-permissive) -> PDAC (T cell-excluded)
    - CT26 (immunogenic) -> PDAC (immunosuppressive)
    - Well-vascularized subcutaneous tumors -> hypoxic PDAC
    Requires 10-100x translation uncertainty for trafficking parameters.
    """


class MeasurementDirectness(str, Enum):
    """How many inferential steps between the raw measurement and the model parameter.

    Captures structural uncertainty from model assumptions needed to extract
    the parameter from observed data. Distinct from statistical uncertainty
    (captured by bootstrap) and translational uncertainty (captured by other
    source_relevance fields).
    """

    DIRECT = "direct"
    """Parameter is the measured quantity or a trivial transform (unit conversion, ln2/x).
    Examples:
    - EC50 from dose-response curve -> EC50_GMCSF
    - Tumor doubling time from CT -> k_C1_growth = ln(2)/VDT
    - Half-life from decay curve -> k_death = ln(2)/t_half
    - TCR clonotype count -> n_CD8_clones
    """

    SINGLE_INVERSION = "single_inversion"
    """One kinetic/mechanistic model assumption needed to extract parameter.
    Examples:
    - CD80% at 0h and 24h + first-order kinetics -> k_APC_mature_ID
    - Serial killing count + mass-action assumption -> k_NK_kill
    - Secretion rate from accumulation assay + linear model -> k_CCL2_sec
    - Serum PK half-life mapped to tissue degradation rate -> k_IL2_deg
    """

    STEADY_STATE_INVERSION = "steady_state_inversion"
    """Parameter inferred from steady-state observables via balance equations.
    Depends on assumed values of other rate constants in the submodel.
    Examples:
    - Macrophage density at diagnosis + assumed death rate -> k_Mac_rec
    - M1/M2 ratio + assumed polarization/death rates -> k_M1_pol
    - MDSC fraction at diagnosis + assumed clearance -> k_MDSC_rec
    - CD8 density + assumed death/proliferation -> q_CD8_T_in
    """

    PROXY_OBSERVABLE = "proxy_observable"
    """Measured quantity is a surrogate for the parameter, not mechanistically linked.
    Examples:
    - RNA expression as proxy for protein secretion rate
    - IHC score as proxy for absolute concentration
    - Proliferation marker (Ki67) as proxy for net growth rate
    """


class TemporalResolution(str, Enum):
    """How well the temporal or dose structure of the data constrains the parameter.

    More timepoints/conditions reduce model-form uncertainty by constraining
    the shape of the kinetic response, not just its endpoint.
    """

    TIMECOURSE = "timecourse"
    """>=3 timepoints or doses spanning the relevant dynamic range.
    Examples:
    - PK curve with multiple blood draws
    - Dose-response with 5+ concentrations
    - Tumor growth measured at monthly CT intervals
    - Time-lapse microscopy with continuous observation
    """

    ENDPOINT_PAIR = "endpoint_pair"
    """Two timepoints (baseline + endpoint) or two conditions.
    Constrains a rate but cannot distinguish kinetic model forms.
    Examples:
    - CD80% at 0h and 24h -> k_APC_mature_ID
    - Kill count at 4h and 16h -> k_NK_kill
    - EC50 from two functional assays -> EC50_GMCSF
    """

    SNAPSHOT_OR_EQUILIBRIUM = "snapshot_or_equilibrium"
    """Single timepoint, cross-sectional data, or assumed steady state.
    Parameter must be inferred from a single observation, possibly
    via balance equations assuming equilibrium.
    Examples:
    - Tumor biopsy cell densities at diagnosis
    - M1/M2 ratio from single resection
    - Macrophage density assumed at recruitment-death balance
    - Steady-state cytokine concentration
    """


class ExperimentalSystem(str, Enum):
    """Biological fidelity of the experimental system to the in vivo tumor context.

    Distinct from source_quality (evidence reliability / peer review status).
    This captures the system gap: how well the experimental conditions recapitulate
    the biology being modeled.
    """

    CLINICAL_IN_VIVO = "clinical_in_vivo"
    """Direct patient measurements (imaging, biopsies, blood draws).
    Minimal system gap -- this IS the target biology.
    Examples:
    - CT-measured tumor growth rates
    - IHC on resected PDAC tissue
    - Serum PK from clinical trials
    """

    ANIMAL_IN_VIVO = "animal_in_vivo"
    """Animal model measurements. Species gap but intact system.
    Examples:
    - KPC mouse tumor growth
    - Syngeneic tumor model immune infiltrates
    - Mouse PK studies
    """

    EX_VIVO = "ex_vivo"
    """Freshly isolated tissue/cells measured without extended culture.
    Preserves in vivo phenotype but loses systemic context.
    Examples:
    - Flow cytometry on freshly dissociated tumor
    - Tissue explant secretion measurements (< 24h)
    - Fresh TAM polarization assessment
    """

    IN_VITRO_COCULTURE = "in_vitro_coculture"
    """Multi-cell-type culture systems that partially recapitulate TME.
    Examples:
    - Organoid + CAF co-cultures
    - Transwell migration assays with conditioned media
    - Tumor spheroid killing assays
    """

    IN_VITRO_PRIMARY = "in_vitro_primary"
    """Primary cells in standard 2D/suspension culture.
    Examples:
    - Monocyte-derived DC maturation kinetics
    - Primary NK cell killing assays
    - PBMC stimulation assays
    """

    IN_VITRO_CELL_LINE = "in_vitro_cell_line"
    """Immortalized cell lines. Biological drift from in vivo phenotype.
    Examples:
    - TF-1 proliferation for GM-CSF EC50
    - 721.221 as NK killing target
    - Cancer cell line growth rates
    """
