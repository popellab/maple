#!/usr/bin/env python3
"""
Observable models for calibration targets.

Defines how to compute observables from model state (full model or submodel).
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from maple.core.calibration.enums import ExtractionMethod, SourceType

# Import SubmodelInput directly (not under TYPE_CHECKING) so Pydantic can resolve it
from maple.core.calibration.shared_models import SubmodelInput


# Support types for measurement output constraints
SupportType = Literal["positive", "non_negative", "unit_interval", "positive_unbounded", "real"]


class AggregationType(str, Enum):
    """Type of population-level aggregation for clinical endpoints.

    Some calibration targets (e.g., ORR, median OS) cannot be expressed as
    single-patient observables — they require aggregating across a virtual
    patient cohort.
    """

    NONE = "none"
    """No aggregation; observable is per-patient (default)."""

    RESPONSE_RATE = "response_rate"
    """Fraction of patients meeting a threshold (e.g., ORR via RECIST).
    Requires threshold_code defining per-patient binary classification.
    """

    MEDIAN_TIME_TO_EVENT = "median_time_to_event"
    """Median time until an event (e.g., median OS, median PFS).
    The observable code computes per-patient event times.
    """

    SURVIVAL_RATE = "survival_rate"
    """Fraction of patients surviving past a time point (e.g., 1-year OS).
    Requires time_point and time_unit.
    """


class PopulationAggregation(BaseModel):
    """
    Population-level aggregation specification for clinical endpoints.

    Attached to Observable when the calibration target is a population summary
    statistic (ORR, median OS, 1-year OS, MPR rate) rather than a per-patient
    observable.
    """

    type: AggregationType = Field(description="Type of population aggregation to apply.")

    threshold_code: Optional[str] = Field(
        default=None,
        description=(
            "Python code defining per-patient binary classification for response_rate.\n"
            "Function signature: classify_patient(time, species_dict, constants, ureg) -> bool\n"
            "Returns True if patient meets the response criterion.\n\n"
            "Example (RECIST partial response):\n"
            "def classify_patient(time, species_dict, constants, ureg):\n"
            "    tumor = species_dict['V_T.C1']\n"
            "    baseline = tumor[0]\n"
            "    nadir = min(tumor)\n"
            "    return (baseline - nadir) / baseline >= 0.3"
        ),
    )

    time_point: Optional[float] = Field(
        default=None,
        description="Time point for survival_rate aggregation (e.g., 365.25 for 1-year OS).",
    )

    time_unit: Optional[str] = Field(
        default=None,
        description="Pint-parseable unit for time_point (e.g., 'day', 'month').",
    )

    rationale: str = Field(
        description=(
            "Why this observable requires population aggregation rather than "
            "per-patient evaluation. Reference the clinical endpoint definition."
        )
    )

    @model_validator(mode="after")
    def validate_aggregation_fields(self) -> "PopulationAggregation":
        """Ensure required fields are present for each aggregation type."""
        if self.type == AggregationType.RESPONSE_RATE:
            if self.threshold_code is None:
                raise ValueError(
                    "threshold_code is required for response_rate aggregation. "
                    "Provide a classify_patient() function that returns True/False per patient."
                )
        elif self.type == AggregationType.SURVIVAL_RATE:
            missing = []
            if self.time_point is None:
                missing.append("time_point")
            if self.time_unit is None:
                missing.append("time_unit")
            if missing:
                raise ValueError(
                    f"survival_rate aggregation requires {', '.join(missing)}. "
                    f"Specify the time horizon for survival assessment."
                )
        return self


class SubmodelPattern(str, Enum):
    """
    Standard ODE patterns for isolated system submodels.

    These correspond to the patterns documented in the prompt and help with
    validation, documentation, and potentially auto-generation of template code.
    """

    FIRST_ORDER_DECAY = "first_order_decay"
    """Pattern 1: dX/dt = -k*X. For clearance, death, dissociation, degradation."""

    PRODUCTION_DECAY = "production_decay"
    """Pattern 2: dC/dt = k_prod - k_decay*C. For cytokine steady-state, protein turnover."""

    EXPONENTIAL_GROWTH = "exponential_growth"
    """Pattern 3: dN/dt = k*N. For cell proliferation, viral replication, early tumor growth."""

    LOGISTIC_GROWTH = "logistic_growth"
    """Pattern 4: dN/dt = k*N*(1 - N/K). For growth with carrying capacity, spheroid expansion."""

    BIRTH_DEATH = "birth_death"
    """Pattern 5: dN/dt = (k_pro - k_death)*N. For populations with separate birth and death."""

    BINDING_EQUILIBRIUM = "binding_equilibrium"
    """Pattern 6: Receptor-ligand binding dynamics. For Kd, kon, koff estimation."""

    MICHAELIS_MENTEN = "michaelis_menten"
    """Pattern 7: dS/dt = -Vmax*S/(Km + S). For enzyme kinetics, saturable processes."""

    TWO_SPECIES_INTERACTION = "two_species_interaction"
    """Pattern 8: Two coupled ODEs. For effector-target killing, predator-prey."""

    CUSTOM = "custom"
    """Non-standard pattern that doesn't fit the above categories."""


class ConstantSourceType(str, Enum):
    """How an observable constant's value is justified."""

    REFERENCE_DB = "reference_db"
    """Value comes directly from the curated reference values database.
    Requires reference_db_name to match an entry in reference_values.yaml."""

    DERIVED_FROM_REFERENCE_DB = "derived_from_reference_db"
    """Value is computed from one or more reference DB entries (e.g., area from diameter).
    Requires reference_db_names listing the entries used in the derivation.
    The biological_basis must show the derivation calculation."""

    LITERATURE = "literature"
    """Value comes from a specific paper.
    Requires source_tag matching a defined source in the calibration target."""


class ObservableConstant(BaseModel):
    """
    A constant used in observable code with explicit biological justification.

    All numeric constants with units that appear in observable code must be declared
    here. This prevents arbitrary magic numbers and forces explicit documentation
    of biological assumptions.

    Every constant MUST be traceable to either the reference values database or
    a specific literature source. No ungrounded "modeling assumptions" allowed.
    """

    name: str = Field(
        description=(
            "Variable name used in observable code to access this constant.\n"
            "Must be a valid Python identifier (e.g., 'area_per_cancer_cell', 'cell_volume')."
        )
    )

    value: float = Field(description="Numeric value of the constant (without units).")

    units: str = Field(
        description=(
            "Pint-parseable unit string (e.g., 'mm**2/cell', 'micrometer**3', 'dimensionless').\n"
            "The constant will be passed to observable code as a Pint Quantity."
        )
    )

    biological_basis: str = Field(
        min_length=20,
        description=(
            "REQUIRED explanation of where this value comes from biologically.\n"
            "Include the reasoning or calculation, not just the value.\n"
            "Must be substantive (minimum 20 characters).\n\n"
            "Examples:\n"
            "- 'From reference DB pdac_cancer_cell_diameter (17 μm) → area = π×(8.5 μm)² = 2.27e-4 mm²'\n"
            "- 'From reference DB ihc_section_volume_4um: 4e-6 cm³/mm² for standard 4-μm sections'\n"
            "- 'From Table 2 of Smith2020: mean cancer cell diameter = 17.3 ± 2.1 μm'"
        ),
    )

    source_type: ConstantSourceType = Field(
        description=(
            "How this constant's value is justified. Every constant must trace to a verifiable source:\n"
            "- reference_db: Value taken directly from curated reference_values.yaml\n"
            "- derived_from_reference_db: Computed from reference DB entries (show derivation in biological_basis)\n"
            "- literature: From a specific paper (must match a source_tag in this target's sources)"
        )
    )

    reference_db_name: Optional[str] = Field(
        default=None,
        description=(
            "Name of the reference DB entry (for source_type='reference_db').\n"
            "Must exactly match a 'name' field in reference_values.yaml.\n"
            "Example: 'ihc_section_volume_4um', 'pdac_cancer_cell_diameter'"
        ),
    )

    reference_db_names: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of reference DB entries used in derivation (for source_type='derived_from_reference_db').\n"
            "Each must match a 'name' field in reference_values.yaml.\n"
            "Example: ['pdac_cancer_cell_diameter'] for area derived from diameter."
        ),
    )

    source_tag: Optional[str] = Field(
        default=None,
        description=(
            "Source tag for literature-derived constants (for source_type='literature').\n"
            "Must match a source_tag defined in this target's primary or secondary sources.\n"
            "Example: 'Smith2020_CellSize'"
        ),
    )

    @model_validator(mode="after")
    def validate_source_fields(self) -> "ObservableConstant":
        """Ensure required fields are present for each source type."""
        if self.source_type == ConstantSourceType.REFERENCE_DB:
            if not self.reference_db_name:
                raise ValueError(
                    f"Observable constant '{self.name}': source_type='reference_db' requires "
                    f"reference_db_name matching an entry in reference_values.yaml."
                )
        elif self.source_type == ConstantSourceType.DERIVED_FROM_REFERENCE_DB:
            if not self.reference_db_names:
                raise ValueError(
                    f"Observable constant '{self.name}': source_type='derived_from_reference_db' requires "
                    f"reference_db_names listing the reference DB entries used in the derivation."
                )
        elif self.source_type == ConstantSourceType.LITERATURE:
            if not self.source_tag:
                raise ValueError(
                    f"Observable constant '{self.name}': source_type='literature' requires "
                    f"source_tag matching a defined source in this calibration target."
                )
        return self


class Observable(BaseModel):
    """
    Observable specification for CalibrationTarget (full model).

    Defines how to compute the experimental observable from full model species.
    """

    code: str = Field(
        description=(
            "Python function that computes the observable from full model species.\n\n"
            "Function signature: compute_observable(time, species_dict, constants, ureg)\n"
            "- time: numpy array with time values (Pint Quantity with day units)\n"
            "- species_dict: dict mapping species names to numpy arrays (Pint Quantities)\n"
            "- constants: dict mapping constant names to Pint Quantities (from constants field)\n"
            "- ureg: Pint UnitRegistry for unit conversions\n\n"
            "Must return a Pint Quantity array with units matching empirical_data.units.\n\n"
            "Example (cell density with stroma correction):\n"
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    cancer_cells = species_dict['V_T.C1']\n"
            "    area_per_cell = constants['area_per_cancer_cell']\n"
            "    stroma_frac = constants['stromal_fraction']\n"
            "    # Tissue area includes cancer cells AND stroma\n"
            "    tumor_area = cancer_cells * area_per_cell / (1 - stroma_frac)\n"
            "    density = cd8 / tumor_area\n"
            "    return density.to('cell/mm**2')"
        )
    )

    units: str = Field(
        description="Pint-parseable units of the observable output (must match empirical_data.units)"
    )

    species: List[str] = Field(
        description=(
            "List of full model species accessed by the observable code.\n"
            "Format: 'compartment.species' (e.g., ['V_T.CD8', 'V_T.C1'])."
        )
    )

    constants: List[ObservableConstant] = Field(
        default_factory=list,
        description="List of geometric/modeling constants used in the observable code.",
    )

    inputs: List[SubmodelInput] = Field(
        default_factory=list,
        description=(
            "List of literature inputs used in the observable code.\n"
            "Use for values that come from experimental papers and need provenance tracking.\n"
            "For derived geometric/modeling constants, use the 'constants' field instead."
        ),
    )

    support: SupportType = Field(
        default="real",
        description=(
            "Mathematical support of the observable output.\n"
            "- 'positive': strictly > 0 (e.g., concentrations, counts)\n"
            "- 'non_negative': >= 0 (e.g., distances, volumes)\n"
            "- 'unit_interval': [0, 1] (e.g., fractions, probabilities)\n"
            "- 'positive_unbounded': > 0 with no upper bound (e.g., ratios)\n"
            "- 'real': any real number (e.g., log-transformed values)"
        ),
    )

    mapping_rationale: Optional[str] = Field(
        default=None,
        description=(
            "Explanation of how the literature measurement maps to model species.\n"
            "Recommended when the mapping is non-obvious or involves aggregation."
        ),
    )

    experimental_denominator: Optional[str] = Field(
        default=None,
        description=(
            "What the experimental measurement divides by. Required when the "
            "observable is a density or fraction.\n\n"
            "Examples:\n"
            "- 'mm^2 of tumor tissue (whole section including stroma)'\n"
            "- 'all cells in ROI (all nucleated cells)'\n"
            "- 'CD3+ T cells (pan-T-cell marker)'\n"
            "- 'CD45+ leukocytes'"
        ),
    )

    model_denominator_species: Optional[List[str]] = Field(
        default=None,
        description=(
            "Which model species compose the denominator in the observable code. "
            "Required when experimental_denominator is set.\n"
            "Format: 'compartment.species' (e.g., ['V_T.CD8', 'V_T.Th', 'V_T.Treg']).\n"
            "For area-based denominators, list the species used to compute area "
            "(e.g., ['V_T.C1'] when tumor area = C1 * area_per_cell)."
        ),
    )

    unmodeled_denominator_components: Optional[str] = Field(
        default=None,
        description=(
            "Components present in the experimental denominator but absent from the "
            "model. Document expected direction and magnitude of systematic bias.\n\n"
            "Examples:\n"
            "- 'B cells (50-70% of LA cells) not modeled; model prediction will be "
            "~2-3x higher than experimental value.'\n"
            "- 'Stromal cells constitute 60-90% of PDAC tissue area; without stromal "
            "fraction correction the model would overpredict density by 3-10x.'\n"
            "- None (denominator fully captured by model species)."
        ),
    )

    @model_validator(mode="after")
    def validate_denominator_fields(self) -> "Observable":
        """Validate denominator audit fields for density/fraction observables."""
        if self.experimental_denominator and not self.model_denominator_species:
            raise ValueError(
                f"Observable has experimental_denominator='{self.experimental_denominator}' "
                f"but model_denominator_species is not set. Specify which model species "
                f"compose the denominator to complete the denominator audit."
            )

        # Density observables (units like cell/mm**2) must have denominator audit
        is_density = (
            self.units not in ("dimensionless",)
            and "/" in self.units
            and self.support in ("positive", "non_negative")
        )
        if is_density and not self.experimental_denominator:
            raise ValueError(
                f"Observable with units='{self.units}' and support='{self.support}' "
                f"is a density but experimental_denominator is not set. "
                f"Document what the experiment divides by (e.g., 'mm^2 of tumor "
                f"tissue including stroma') to enable denominator audit."
            )

        return self

    aggregation: Optional[PopulationAggregation] = Field(
        default=None,
        description=(
            "Population-level aggregation specification. Use when the calibration target "
            "is a population summary (ORR, median OS, 1-year OS) rather than a per-patient "
            "observable. When None (default), the observable is per-patient."
        ),
    )


class SubmodelStateVariable(BaseModel):
    """
    State variable in an isolated submodel with self-contained initial condition.

    Each state variable includes its initial value and full provenance,
    eliminating the need to reference inputs defined elsewhere.
    """

    name: str = Field(description="Name of the state variable in ODE (e.g., 'spheroid_cells')")
    units: str = Field(description="Pint-parseable units (e.g., 'cell', 'micrometer')")
    initial_value: float = Field(description="Initial condition value for this state variable")
    # Provenance fields for the initial value
    source_ref: str = Field(
        description=(
            "Source reference tag for the initial value. MUST match a source_tag in "
            "primary_data_source or secondary_data_sources."
        )
    )
    value_location: str = Field(
        description="Where the initial value appears in the source (e.g., 'Methods p.3', 'Table 1')"
    )
    value_snippet: str = Field(
        description="Exact text snippet from the source containing the initial value"
    )

    # Figure extraction fields
    source_type: SourceType = Field(
        default=SourceType.TEXT,
        description=(
            "Type of source from which the value was extracted:\n"
            "- text: Body text, results section, or abstract (default)\n"
            "- table: Table\n"
            "- figure: Figure (requires figure_id and extraction_method)"
        ),
    )

    figure_id: Optional[str] = Field(
        None,
        description=(
            "Figure identifier (e.g., 'Figure 2A', 'Fig. 3B'). "
            "Required when source_type='figure'."
        ),
    )

    extraction_method: Optional[ExtractionMethod] = Field(
        None,
        description=(
            "Method used to extract value from figure. Required when source_type='figure'.\n"
            "- manual: Manual reading from figure axes\n"
            "- digitizer: Generic digitizer software\n"
            "- webplotdigitizer: WebPlotDigitizer tool\n"
            "- other: Other method (specify in extraction_notes)"
        ),
    )

    extraction_notes: Optional[str] = Field(
        None,
        description=(
            "Additional context for figure extraction.\n"
            "Example: 'Read from y-axis at day 14 timepoint'\n"
            "Example: 'Digitized all points from survival curve'"
        ),
    )

    @model_validator(mode="after")
    def validate_figure_fields(self) -> "SubmodelStateVariable":
        """Ensure figure sources have required figure_id and extraction_method."""
        if self.source_type == SourceType.FIGURE:
            missing = []
            if not self.figure_id:
                missing.append("figure_id")
            if not self.extraction_method:
                missing.append("extraction_method")
            if missing:
                raise ValueError(
                    f"When source_type='figure', the following fields are required: {', '.join(missing)}"
                )
        return self


class SubmodelObservable(BaseModel):
    """
    Observable specification for IsolatedSystemTarget (submodel).

    Defines how to compute the experimental observable from integrated submodel state.
    If code is omitted, defaults to returning y[0] with the specified units.
    """

    code: Optional[str] = Field(
        default=None,
        description=(
            "Python function that computes the observable from integrated submodel state.\n\n"
            "OPTIONAL: If omitted, defaults to returning y[0] * ureg(units).\n"
            "Only write code if you need transformations (e.g., cell count → diameter).\n\n"
            "Function signature: compute_observable(t, y, constants, ureg)\n"
            "- t: time value (float, in t_unit)\n"
            "- y: state vector (list of floats, same order as state_variables)\n"
            "- constants: dict mapping constant names to Pint Quantities\n"
            "- ureg: Pint UnitRegistry for unit conversions\n\n"
            "Must return a Pint Quantity with units matching empirical_data.units.\n\n"
            "Example (convert cell count to spheroid diameter):\n"
            "def compute_observable(t, y, constants, ureg):\n"
            "    import numpy as np\n"
            "    cells = y[0]\n"
            "    cell_volume = constants['cell_volume']\n"
            "    volume = cells * cell_volume\n"
            "    radius = ((3 * volume) / (4 * np.pi)) ** (1/3)\n"
            "    return (2 * radius).to('micrometer')"
        ),
    )

    units: str = Field(
        description="Pint-parseable units of the observable output (must match empirical_data.units)"
    )

    constants: List[ObservableConstant] = Field(
        default_factory=list, description="List of constants used in the observable code."
    )

    rationale: Optional[str] = Field(
        default=None,
        description=(
            "Explanation of why this transformation from submodel state to observable is appropriate.\n"
            "Only needed when the transformation is non-trivial (beyond simple unit conversion).\n\n"
            "Examples:\n"
            "- 'Spheroid diameter computed from cell count assuming spherical geometry and uniform packing'\n"
            "- 'Fraction bound computed from equilibrium expression R_bound = L/(Kd + L)'"
        ),
    )


class Submodel(BaseModel):
    """
    Isolated submodel for IsolatedSystemTarget.

    Defines an ODE system that approximates the full model dynamics
    for an isolated experimental system (in vitro, preclinical).
    """

    code: str = Field(
        description=(
            "Python code defining submodel(t, y, params, inputs) -> dydt.\n"
            "  t: time (float, in t_unit)\n"
            "  y: state vector (list of floats, order matches state_variables)\n"
            "  params: dict of parameter values (use full model parameter names)\n"
            "  inputs: dict of experimental conditions (from submodel.inputs)\n"
            "  returns: list of derivatives (same length as y)"
        ),
        examples=[
            # Exponential growth
            "def submodel(t, y, params, inputs):\n"
            "    N = y[0]\n"
            "    k = params['k_growth']\n"
            "    return [k * N]",
            # Logistic growth
            "def submodel(t, y, params, inputs):\n"
            "    N = y[0]\n"
            "    k = params['k_growth']\n"
            "    K = params['carrying_capacity']\n"
            "    return [k * N * (1 - N / K)]",
            # Birth-death
            "def submodel(t, y, params, inputs):\n"
            "    N = y[0]\n"
            "    k_pro = params['k_proliferation']\n"
            "    k_death = params['k_death']\n"
            "    return [(k_pro - k_death) * N]",
            # First-order decay
            "def submodel(t, y, params, inputs):\n"
            "    X = y[0]\n"
            "    k = params['k_decay']\n"
            "    return [-k * X]",
        ],
    )

    inputs: List[SubmodelInput] = Field(
        default_factory=list,
        description=(
            "Experimental inputs used in submodel code.\n"
            "These are values from the paper like E:T ratio, drug concentration, etc.\n"
            "Each input has full provenance (source_ref, value_location, value_snippet).\n"
            "Accessed in submodel code via inputs['name']."
        ),
    )

    state_variables: List[SubmodelStateVariable] = Field(
        description=(
            "State variables in order matching the y vector in submodel code.\n\n"
            "Each state variable is SELF-CONTAINED with:\n"
            "- name: variable name in ODE (e.g., 'T_cells')\n"
            "- units: Pint-parseable units (e.g., 'cell')\n"
            "- initial_value: numeric initial condition\n"
            "- source_ref: reference to source (must match a defined source_tag)\n"
            "- value_location: where in the source (e.g., 'Methods p.3')\n"
            "- value_snippet: exact text from source\n\n"
            "For figure-extracted values, also include:\n"
            "- source_type: 'figure'\n"
            "- figure_id: e.g., 'Figure 2A'\n"
            "- extraction_method: 'manual', 'digitizer', 'webplotdigitizer', or 'other'"
        )
    )

    t_span: List[float] = Field(
        description="Integration time span [t_start, t_end] for ODE solver.",
        min_length=2,
        max_length=2,
    )

    t_unit: str = Field(
        default="day", description="Pint-parseable time unit for t_span (e.g., 'day', 'hour')"
    )

    observable: SubmodelObservable = Field(
        description="How to compute the experimental observable from submodel state."
    )

    pattern: SubmodelPattern = Field(
        default=SubmodelPattern.CUSTOM,
        description=(
            "Which standard ODE pattern this submodel follows.\n\n"
            "Pattern selection by data type:\n"
            "- Half-life/decay curve → first_order_decay: dX/dt = -k*X\n"
            "- Steady-state + turnover → production_decay: dC/dt = k_prod - k_decay*C\n"
            "- Doubling time/fold expansion → exponential_growth: dN/dt = k*N\n"
            "- Growth curve with plateau → logistic_growth: dN/dt = k*N*(1-N/K)\n"
            "- Separate birth + death rates → birth_death: dN/dt = (k_pro - k_death)*N\n"
            "- Kd, kon, koff binding data → binding_equilibrium: receptor-ligand ODE\n"
            "- Saturation curve (Vmax, Km) → michaelis_menten: dS/dt = -Vmax*S/(Km+S)\n"
            "- Killing assay / E:T response → two_species_interaction: coupled ODEs\n"
            "- Other (CFSE, delays, etc.) → custom\n\n"
            "Key identifiability:\n"
            "- birth_death: only net rate identifiable unless both measured separately\n"
            "- binding_equilibrium: Kd from equilibrium; kon/koff require kinetic data\n"
            "- two_species_interaction: k_kill has units 1/(cell × time)"
        ),
    )

    identifiability_notes: Optional[str] = Field(
        default=None,
        description=(
            "Notes on parameter identifiability from this experimental data.\n"
            "Document which parameters are independently identifiable vs. only jointly identifiable.\n\n"
            "Examples:\n"
            "- 'Only net growth rate (k_pro - k_death) identifiable from this data; "
            "individual rates require separate proliferation and death assays'\n"
            "- 'Kd identifiable but kon and koff require kinetic (non-equilibrium) data'\n"
            "- 'Vmax and Km jointly identifiable from saturation curve'"
        ),
    )

    rationale: str = Field(
        description=(
            "Explanation of why this ODE structure is appropriate for the experimental system.\n"
            "Address:\n"
            "- Why this pattern (e.g., exponential vs logistic growth) was chosen\n"
            "- How the submodel approximates the relevant full model dynamics\n"
            "- Any simplifications made and their justification\n\n"
            "Examples:\n"
            "- 'Exponential growth valid for early expansion before contact inhibition'\n"
            "- 'First-order decay appropriate for linear elimination regime'\n"
            "- 'Two-species interaction captures E:T dynamics without modeling effector exhaustion'"
        )
    )
