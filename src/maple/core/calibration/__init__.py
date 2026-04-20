"""
Calibration target models for maple.

Main classes:
- CalibrationTarget: For clinical/in vivo data (full model context)
- SubmodelTarget: For in vitro/preclinical data with typed forward models and Julia translation
"""

# Main calibration target classes
from maple.core.calibration.calibration_target_models import (
    CalibrationTarget,
    CalibrationTargetEstimates,
    CalibrationTargetFooters,
    IndexType,
)
from maple.core.calibration.submodel_target import (
    SubmodelTarget,
)

# Enums
from maple.core.calibration.enums import (
    Compartment,
    ExtractionMethod,
    Indication,
    IndicationMatch,
    MouseSubspecifier,
    PerturbationType,
    SourceQuality,
    SourceType,
    Species,
    StageBurden,
    StageExtent,
    System,
    TMECompatibility,
    TreatmentHistory,
    TreatmentStatus,
    enum_field_description,
)

# Scenario models
from maple.core.calibration.scenario import (
    Intervention,
    Scenario,
)

# Observable models
from maple.core.calibration.observable import (
    AggregationType,
    ConstantSourceType,
    Observable,
    ObservableConstant,
    PopulationAggregation,
    Submodel,
    SubmodelObservable,
    SubmodelPattern,
    SubmodelStateVariable,
    SupportType,
)

# Experimental context
from maple.core.calibration.experimental_context import (
    ExperimentalContext,
    Stage,
    TreatmentContext,
)

# Shared models
from maple.core.calibration.shared_models import (
    CellLine,
    CellSpecies,
    CultureConditions,
    DoseResponseData,
    EstimateInput,
    FigureExcerpt,
    InputType,
    KeyAssumption,
    ModelingAssumption,
    SecondarySource,
    Snippet,
    Source,
    TableExcerpt,
    ClinicalSourceRelevance,
    SourceRelevanceAssessment,
    SubmodelInput,
    TrajectoryData,
    UncertaintyType,
    Validation,
)

# Validators
from maple.core.calibration.validators import (
    check_value_in_text,
    create_mock_species,
    fuzzy_match,
    get_typical_species_value,
    resolve_doi,
)

# Snippet validation
from maple.core.calibration.snippet_validator import (
    validate_snippets_in_file,
    validate_snippets_in_dir,
)

# Code validation
from maple.core.calibration.code_validator import (
    EXPECTED_SIGNATURES,
    CodeType,
    CodeValidationResult,
    CodeValidator,
    ValidationIssue,
    find_hardcoded_constants,
    validate_code_block,
)

# Exceptions
from maple.core.calibration.exceptions import (
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
    "SubmodelTarget",
    # Enums
    "Species",
    "MouseSubspecifier",
    "TreatmentHistory",
    "TreatmentStatus",
    "StageExtent",
    "StageBurden",
    "Indication",
    "IndicationMatch",
    "SourceQuality",
    "PerturbationType",
    "TMECompatibility",
    "Compartment",
    "System",
    "SourceType",
    "ExtractionMethod",
    "enum_field_description",
    # Scenario
    "Intervention",
    "Scenario",
    # Observable
    "AggregationType",
    "ConstantSourceType",
    "PopulationAggregation",
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
    "EstimateInput",
    "FigureExcerpt",
    "SubmodelInput",
    "ModelingAssumption",
    "InputType",
    "KeyAssumption",
    "TableExcerpt",
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
    "ClinicalSourceRelevance",
    "SourceRelevanceAssessment",
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
    # Snippet validation
    "validate_snippets_in_file",
    "validate_snippets_in_dir",
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
