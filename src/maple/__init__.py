"""
Maple - Tools for extracting QSP calibration targets from scientific literature.

Provides structured YAML schemas with Pydantic validation and Julia/Turing.jl
translation for Bayesian inference.
"""

__version__ = "0.1.0"

# Public API
from maple.core.prompt_builder import (
    PromptBuilder,
    CalibrationTargetPromptBuilder,
    SubmodelTargetPromptBuilder,
)
from maple.core.workflow_orchestrator import WorkflowOrchestrator

# Exception classes for validation error handling
from maple.core.calibration.exceptions import (
    CalibrationTargetValidationError,
    CodeSyntaxError,
    CodeStructureError,
    UnitValidationError,
    MissingUnitsError,
    DimensionalityMismatchError,
    UnitConversionError,
    ReturnValueError,
    ScalarReturnError,
    ArrayLengthError,
    ReturnStructureError,
    DataConsistencyError,
    ComputedValueMismatchError,
    ScaleMismatchError,
    HardcodedConstantError,
    CalibrationReferenceError,
    DOIResolutionError,
    PaperTitleMismatchError,
    SourceRefError,
    SpeciesNotFoundError,
    ContentValidationError,
    ControlCharacterError,
    EmptyScenarioError,
)

__all__ = [
    "__version__",
    "PromptBuilder",
    "CalibrationTargetPromptBuilder",
    "SubmodelTargetPromptBuilder",
    "WorkflowOrchestrator",
    # Exception classes
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
