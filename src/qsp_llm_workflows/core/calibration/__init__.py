"""
Calibration target models for QSP LLM Workflows.

This module provides Pydantic models for calibration targets extracted from
scientific literature, used to calibrate QSP model parameters via Bayesian inference.

Main classes:
- CalibrationTarget: For clinical/in vivo data (full model context)
  - Uses `observable` field to define how to compute measurement from full model species
- IsolatedSystemTarget: For in vitro/preclinical data with Python submodels
  - Uses `submodel` field with nested ODE code, state variables, parameters, and observable
- IndexType: Enum for vector-valued data index dimension (time, dose, ratio, etc.)

CalibrationTarget:
Uses the full QSP model. The `observable` field defines Python code to compute the
experimental measurement from model species (e.g., ratio of CD8 cells to tumor cells).

IsolatedSystemTarget:
For isolated systems (in vitro, preclinical), the LLM builds a Python submodel that
approximates the relevant dynamics from the full QSP model. The `submodel` field includes:
- `code`: ODE function using parameter names from full model (for joint inference)
- `state_variables`: State variables with names and units
- `parameters`: List of parameter names from the full model
- `observable`: How to compute the measurement from submodel state

Supporting modules:
- enums: Species, Indication, System enums (Compartment kept for backward compatibility)
- scenario: Intervention, Scenario models
- observable: Observable, Submodel, ObservableConstant models
- experimental_context: Stage, TreatmentContext, ExperimentalContext
- shared_models: Input (scalar/vector), Source, Snippet
- validators: Validation helper functions (resolve_doi, fuzzy_match, etc.)
- exceptions: Custom exception classes for validation errors

Vector-valued data:
Calibration targets support both scalar and vector-valued data through a unified pathway:
- Scalar: median=[42.0], ci95=[[37.0, 47.0]] (length-1 lists)
- Vector: index_values=[0, 24, 48, 72], median=[10, 15, 20, 18] (matching lengths)
- Input.value can be float (broadcast) or List[float] (per-index-point)
"""

# Main calibration target classes
from qsp_llm_workflows.core.calibration.calibration_target_models import (
    CalibrationTarget,
    CalibrationTargetEstimates,
    CalibrationTargetFooters,
    IndexType,
)
from qsp_llm_workflows.core.calibration.isolated_system_target import (
    IsolatedSystemTarget,
)

# Enums
from qsp_llm_workflows.core.calibration.enums import (
    Compartment,
    Indication,
    MouseSubspecifier,
    Species,
    StageBurden,
    StageExtent,
    System,
    TreatmentHistory,
    TreatmentStatus,
    enum_field_description,
)

# Scenario models
from qsp_llm_workflows.core.calibration.scenario import (
    Intervention,
    Scenario,
)

# Observable models
from qsp_llm_workflows.core.calibration.observable import (
    Observable,
    ObservableConstant,
    Submodel,
    SubmodelObservable,
    SubmodelPattern,
    SubmodelStateVariable,
    SupportType,
)

# Experimental context
from qsp_llm_workflows.core.calibration.experimental_context import (
    ExperimentalContext,
    Stage,
    TreatmentContext,
)

# Shared models
from qsp_llm_workflows.core.calibration.shared_models import (
    CellLine,
    CellSpecies,
    ContextMismatch,
    CultureConditions,
    DoseResponseData,
    InputType,
    KeyAssumption,  # Kept for backward compatibility with ParameterMetadata/TestStatistic
    LiteratureInput,
    MismatchDimension,
    ModelingAssumption,
    SecondarySource,
    Snippet,
    Source,
    TrajectoryData,
    UncertaintyType,
    Validation,
)

# Validators
from qsp_llm_workflows.core.calibration.validators import (
    check_value_in_text,
    create_mock_species,
    fuzzy_match,
    get_typical_species_value,
    resolve_doi,
)

# Code validation
from qsp_llm_workflows.core.calibration.code_validator import (
    EXPECTED_SIGNATURES,
    CodeType,
    CodeValidationResult,
    CodeValidator,
    ValidationIssue,
    find_hardcoded_constants,
    validate_code_block,
)

# Exceptions
from qsp_llm_workflows.core.calibration.exceptions import (
    ArrayLengthError,
    CalibrationReferenceError,
    CalibrationTargetValidationError,
    CodeStructureError,
    CodeSyntaxError,
    ComputedValueMismatchError,
    ContentValidationError,
    ControlCharacterError,
    DataConsistencyError,
    DimensionalityMismatchError,
    DOIResolutionError,
    EmptyScenarioError,
    HardcodedConstantError,
    MissingUnitsError,
    PaperTitleMismatchError,
    ReturnStructureError,
    ReturnValueError,
    ScaleMismatchError,
    ScalarReturnError,
    SourceRefError,
    SpeciesNotFoundError,
    UnitConversionError,
    UnitValidationError,
)

__all__ = [
    # Main classes
    "CalibrationTarget",
    "CalibrationTargetEstimates",
    "CalibrationTargetFooters",
    "IndexType",
    "IsolatedSystemTarget",
    # Enums
    "Species",
    "MouseSubspecifier",
    "TreatmentHistory",
    "TreatmentStatus",
    "StageExtent",
    "StageBurden",
    "Indication",
    "Compartment",
    "System",
    "enum_field_description",
    # Scenario
    "Intervention",
    "Scenario",
    # Observable
    "Observable",
    "ObservableConstant",
    "SupportType",
    "Submodel",
    "SubmodelObservable",
    "SubmodelPattern",
    "SubmodelStateVariable",
    # Context
    "Stage",
    "TreatmentContext",
    "ExperimentalContext",
    # Shared models
    "LiteratureInput",
    "ModelingAssumption",
    "InputType",
    "KeyAssumption",
    "Source",
    "SecondarySource",
    "Snippet",
    "Validation",
    "ContextMismatch",
    "MismatchDimension",
    "CellSpecies",
    "CellLine",
    "CultureConditions",
    "UncertaintyType",
    "TrajectoryData",
    "DoseResponseData",
    # Validators
    "resolve_doi",
    "fuzzy_match",
    "check_value_in_text",
    "get_typical_species_value",
    "create_mock_species",
    # Code validation
    "CodeType",
    "CodeValidator",
    "CodeValidationResult",
    "ValidationIssue",
    "validate_code_block",
    "find_hardcoded_constants",
    "EXPECTED_SIGNATURES",
    # Exceptions
    "CalibrationTargetValidationError",
    "CodeSyntaxError",
    "CodeStructureError",
    "UnitValidationError",
    "MissingUnitsError",
    "DimensionalityMismatchError",
    "UnitConversionError",
    "ReturnValueError",
    "ScalarReturnError",
    "ArrayLengthError",
    "ReturnStructureError",
    "DataConsistencyError",
    "ComputedValueMismatchError",
    "ScaleMismatchError",
    "HardcodedConstantError",
    "CalibrationReferenceError",
    "DOIResolutionError",
    "PaperTitleMismatchError",
    "SourceRefError",
    "SpeciesNotFoundError",
    "ContentValidationError",
    "ControlCharacterError",
    "EmptyScenarioError",
]
