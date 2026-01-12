#!/usr/bin/env python3
"""
Scenario models for calibration targets.

Provides models for interventions, measurements, and experimental scenarios
that define what is measured and under what conditions.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from qsp_llm_workflows.core.exceptions import EmptyScenarioError


# Support types for measurement output constraints
SupportType = Literal["positive", "non_negative", "unit_interval", "positive_unbounded", "real"]


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
            "- How much: Dose and units\n"
            "- When: Schedule/timing\n"
            "- Additional details: Any relevant context\n\n"
            "Clinical/in vivo examples:\n"
            "- 'Anti-PD-1 antibody 3 mg/kg IV every 2 weeks starting day 0 (patient weight 70 kg)'\n"
            "- 'Surgical resection on day 14, removing 90% of tumor burden'\n"
            "- 'Gemcitabine 1000 mg/m2 on days 0, 7, 14 (patient BSA 1.8 m2)'\n"
            "- 'No intervention (natural disease progression)'\n\n"
            "In vitro examples:\n"
            "- 'Add recombinant IL-2 at 10 ng/mL at t=0'\n"
            "- 'Co-culture tumor cells with CD8+ T cells at E:T ratio 5:1'\n"
            "- 'Stimulate with anti-CD3/CD28 beads (bead:cell ratio 1:1) for 48h'\n"
            "- 'Treat with 1 μM drug X at t=24h'\n"
            "- 'No treatment (unstimulated control)'"
        )
    )


class MeasurementMapping(BaseModel):
    """
    Documents how literature measurements map to model species.

    Forces explicit thinking about what the literature actually measures
    and how it corresponds to model species.
    """

    literature_measures: str = Field(
        description=(
            "What the literature measurement actually captures.\n"
            "Be specific about the experimental technique and what it detects.\n\n"
            "Examples:\n"
            "- 'Total CD8+ T cells via anti-CD8 antibody staining (detects all CD8+ regardless of exhaustion)'\n"
            "- 'PD-1+CD8+ T cells via dual-marker flow cytometry (detects exhausted subset only)'\n"
            "- 'Collagen I/III protein by mass spectrometry (does not distinguish collagen subtypes)'"
        )
    )

    model_species_included: List[str] = Field(
        description=(
            "Model species that should be summed/combined to match the literature measurement.\n"
            "Must match entries in measurement_species."
        )
    )

    model_species_excluded: List[str] = Field(
        default_factory=list,
        description=(
            "Model species deliberately excluded from the measurement and why.\n"
            "Document species that exist in the model but are NOT part of this measurement."
        ),
    )

    mapping_rationale: str = Field(
        description=(
            "Explanation of why this mapping is appropriate.\n"
            "Address any assumptions about what the experimental technique captures."
        )
    )


class MeasurementConstant(BaseModel):
    """
    A constant used in measurement_code with explicit biological justification.

    All numeric constants with units that appear in measurement_code must be declared
    here. This prevents arbitrary magic numbers and forces explicit documentation
    of biological assumptions.
    """

    name: str = Field(
        description=(
            "Variable name used in measurement_code to access this constant.\n"
            "Must be a valid Python identifier (e.g., 'area_per_cancer_cell', 'normal_ecm_concentration')."
        )
    )

    value: float = Field(description="Numeric value of the constant (without units).")

    units: str = Field(
        description=(
            "Pint-parseable unit string (e.g., 'mm**2/cell', 'mg/mL', 'dimensionless').\n"
            "The constant will be passed to measurement_code as a Pint Quantity."
        )
    )

    biological_basis: str = Field(
        description=(
            "Explanation of where this value comes from biologically.\n"
            "Must include the reasoning or calculation, not just the value.\n\n"
            "Examples:\n"
            "- 'Cancer cell diameter ~17 μm → cross-sectional area = π×(8.5 μm)² = 227 μm² = 2.27e-4 mm²'\n"
            "- 'Normal pancreas ECM concentration ~23 mg/mL, derived from PDAC being ~2.6× elevated (60 mg/mL baseline)'\n"
            "- 'T cell diameter ~7 μm → area = π×(3.5 μm)² = 38.5 μm² = 3.85e-5 mm²'"
        )
    )

    source_ref: str = Field(
        description=(
            "Reference for this constant value.\n"
            "Use 'modeling_assumption' for geometric calculations or well-established values.\n"
            "Use a source_tag (e.g., 'Smith2020_CellSize') for literature-derived values."
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

    measurement_constants: List[MeasurementConstant] = Field(
        default_factory=list,
        description=(
            "List of constants used in measurement_code.\n"
            "All numeric values with units (conversion factors, reference values, etc.) must be declared here.\n"
            "Each constant requires a biological_basis explaining where the value comes from.\n\n"
            "These are passed to measurement_code as a dict of Pint Quantities.\n"
            "Access via: constants['area_per_cancer_cell']"
        ),
    )

    measurement_code: str = Field(
        description=(
            "Python function that computes the observable from species time series.\n\n"
            "Function signature: compute_measurement(time, species_dict, ureg, constants)\n"
            "- time: numpy array with time values (Pint Quantity with day units)\n"
            "- species_dict: dict mapping species names to numpy arrays (Pint Quantities, one value per timepoint)\n"
            "- ureg: Pint UnitRegistry for unit conversions\n"
            "- constants: dict mapping constant names to Pint Quantities (from measurement_constants)\n\n"
            "Must return a Pint Quantity (scalar or array) with units matching calibration_target_estimates.units.\n\n"
            "IMPORTANT: Do NOT hardcode numeric values with units in the code.\n"
            "All conversion factors and reference values must come from the constants dict.\n"
            "Only universal constants (π, percentiles like 2.5/97.5) may be inline.\n\n"
            "IMPORTANT: Do NOT include time filtering logic (e.g., 'use last timepoint' or 'at t=14').\n"
            "This function computes WHAT to measure from the time series.\n"
            "WHEN to measure is handled separately via threshold_description.\n\n"
            "Example (density with conversion factor):\n"
            "def compute_measurement(time, species_dict, ureg, constants):\n"
            "    cd8 = species_dict['V_T.CD8']  # CD8 T cells over time\n"
            "    c_cells = species_dict['V_T.C1']  # Cancer cells over time\n"
            "    area_per_cell = constants['area_per_cancer_cell']  # From measurement_constants\n"
            "    tumor_area = c_cells * area_per_cell\n"
            "    density = cd8 / tumor_area\n"
            "    return density.to(ureg('cell/mm**2'))\n\n"
            "Example (simple ratio - no constants needed):\n"
            "def compute_measurement(time, species_dict, ureg, constants):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    treg = species_dict['V_T.Treg']\n"
            "    ratio = cd8 / treg\n"
            "    return ratio.to(ureg.dimensionless)"
        )
    )

    threshold_description: str = Field(
        description=(
            "Text description of WHEN the measurement occurs:\n"
            "- Trigger: What biological/experimental event triggers measurement\n"
            "- Threshold: Specific value if stated\n"
            "- Context: Experimental context\n\n"
            "Clinical/in vivo examples:\n"
            "- 'At tumor resection when tumor burden reaches approximately 1e9 cells (~500 mm³)'\n"
            "- 'At baseline/diagnosis before any treatment (tumor burden ~1e9 cells)'\n"
            "- '7 days after first anti-PD-1 dose when tumor begins responding'\n"
            "- 'At clinical presentation (median tumor volume 450 mm³ in study cohort)'\n\n"
            "In vitro examples:\n"
            "- 'At 24 hours post-stimulation'\n"
            "- 'At 48h after co-culture initiation'\n"
            "- 'At confluence (when cells reach ~90% coverage)'\n"
            "- 'After 3 population doublings (~72h for this cell line)'\n"
            "- 'At steady state (no change in cell counts for 24h)'\n"
            "- 'At peak proliferation (determined by prior kinetic studies)'"
        )
    )

    support: SupportType = Field(
        description=(
            "Expected mathematical support (valid value range) of the measurement output.\n\n"
            "- 'positive': Output must be > 0 (densities, concentrations, cell counts)\n"
            "- 'non_negative': Output must be >= 0 (counts that could be zero)\n"
            "- 'unit_interval': Output must be in [0, 1] (fractions, proportions)\n"
            "- 'positive_unbounded': Output must be > 0, no upper limit (fold-changes, ratios > 1)\n"
            "- 'real': Output can be any real number (log-ratios, change scores)"
        )
    )

    measurement_mapping: Optional[MeasurementMapping] = Field(
        default=None,
        description=(
            "Documents how literature measurements map to model species.\n"
            "Recommended when measuring 'total' quantities or when species mapping is non-obvious."
        ),
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
            raise EmptyScenarioError("Scenario must include at least one measurement")
        return v
