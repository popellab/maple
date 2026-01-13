#!/usr/bin/env python3
"""
Observable models for calibration targets.

Defines how to compute observables from model state (full model or submodel).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# Support types for measurement output constraints
SupportType = Literal["positive", "non_negative", "unit_interval", "positive_unbounded", "real"]


class ObservableConstant(BaseModel):
    """
    A constant used in observable code with explicit biological justification.

    All numeric constants with units that appear in observable code must be declared
    here. This prevents arbitrary magic numbers and forces explicit documentation
    of biological assumptions.
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
        description=(
            "Explanation of where this value comes from biologically.\n"
            "Must include the reasoning or calculation, not just the value.\n\n"
            "Examples:\n"
            "- 'Cancer cell diameter ~17 μm → cross-sectional area = π×(8.5 μm)² = 227 μm² = 2.27e-4 mm²'\n"
            "- 'T cell diameter ~7 μm → volume = 4/3×π×(3.5 μm)³ ≈ 180 μm³'\n"
            "- 'Assumed spherical packing with 74% density'"
        )
    )

    source_ref: str = Field(
        description=(
            "Reference for this constant value.\n"
            "Use 'modeling_assumption' for geometric calculations or well-established values.\n"
            "Use a source_tag (e.g., 'Smith2020_CellSize') for literature-derived values."
        )
    )


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
            "Must return a Pint Quantity array with units matching calibration_target_estimates.units.\n\n"
            "Example (cell density):\n"
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    cd8 = species_dict['V_T.CD8']\n"
            "    cancer_cells = species_dict['V_T.C1']\n"
            "    area_per_cell = constants['area_per_cancer_cell']\n"
            "    tumor_area = cancer_cells * area_per_cell\n"
            "    density = cd8 / tumor_area\n"
            "    return density.to('cell/mm**2')"
        )
    )

    units: str = Field(
        description="Pint-parseable units of the observable output (must match calibration_target_estimates.units)"
    )

    species: List[str] = Field(
        description=(
            "List of full model species accessed by the observable code.\n"
            "Format: 'compartment.species' (e.g., ['V_T.CD8', 'V_T.C1'])."
        )
    )

    constants: List[ObservableConstant] = Field(
        default_factory=list, description="List of constants used in the observable code."
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


class SubmodelStateVariable(BaseModel):
    """State variable in an isolated submodel."""

    name: str = Field(description="Name of the state variable (e.g., 'PDAC_spheroid_cells')")
    units: str = Field(description="Pint-parseable units (e.g., 'cell', 'micrometer')")


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
            "Must return a Pint Quantity with units matching calibration_target_estimates.units.\n\n"
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
        description="Pint-parseable units of the observable output (must match calibration_target_estimates.units)"
    )

    constants: List[ObservableConstant] = Field(
        default_factory=list, description="List of constants used in the observable code."
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
            "  inputs: dict of experimental conditions (from Input objects)\n"
            "  returns: list of derivatives (same length as y)\n\n"
            "Example:\n"
            "def submodel(t, y, params, inputs):\n"
            "    S = y[0]  # spheroid cell count\n"
            "    k_prolif = params['k_C1_growth']\n"
            "    C_max = params['C_max']\n"
            "    dSdt = k_prolif * S * (1 - S / C_max)\n"
            "    return [dSdt]"
        )
    )

    state_variables: List[SubmodelStateVariable] = Field(
        description="State variables in order matching the y vector in submodel code."
    )

    parameters: List[str] = Field(
        description=(
            "Parameter names from the full QSP model used in this submodel.\n"
            "These enable joint inference across calibration targets."
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

    rationale: str = Field(
        description="Why this submodel approximation is appropriate for the experimental data."
    )
