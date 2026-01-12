"""
Calibration target models for QSP LLM Workflows.

This module provides Pydantic models for calibration targets extracted from
scientific literature, used to calibrate QSP model parameters via Bayesian inference.

Main classes:
- CalibrationTarget: For clinical/in vivo data (full model context)
- IsolatedSystemTarget: For in vitro data with model "cuts" defining reduced systems

Supporting modules:
- enums: Species, Indication, Compartment, System enums
- scenario: Intervention, Measurement, Scenario models
- experimental_context: Stage, TreatmentContext, ExperimentalContext
- shared_models: Input, Source, Snippet, TrajectoryData, DoseResponseData
- validators: Validation helper functions (resolve_doi, fuzzy_match, etc.)
- exceptions: Custom exception classes for validation errors
"""

# Main calibration target classes
from qsp_llm_workflows.core.calibration.calibration_target_models import (
    CalibrationTarget,
    CalibrationTargetEstimates,
    CalibrationTargetFooters,
)
from qsp_llm_workflows.core.calibration.isolated_system_target import (
    CompartmentCut,
    Cut,
    IsolatedSystemTarget,
    ReactionCut,
    SpeciesCut,
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
    Measurement,
    MeasurementConstant,
    MeasurementMapping,
    Scenario,
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
    CultureConditions,
    DoseResponseData,
    Input,
    KeyAssumption,
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
    "IsolatedSystemTarget",
    "SpeciesCut",
    "CompartmentCut",
    "ReactionCut",
    "Cut",
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
    "MeasurementMapping",
    "MeasurementConstant",
    "Measurement",
    "Scenario",
    # Context
    "Stage",
    "TreatmentContext",
    "ExperimentalContext",
    # Shared models
    "Input",
    "KeyAssumption",
    "Source",
    "SecondarySource",
    "Snippet",
    "Validation",
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
