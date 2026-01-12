"""
Custom exception hierarchy for QSP LLM workflows.

Provides specific exception types for different failure modes with
proper error chaining and context preservation.
"""


class WorkflowException(Exception):
    """
    Base exception for all workflow-related errors.

    Provides consistent error handling and context preservation across
    the workflow execution pipeline.
    """

    def __init__(self, message: str, context: dict = None):
        """
        Initialize workflow exception.

        Args:
            message: Human-readable error description
            context: Optional dict with additional error context
        """
        super().__init__(message)
        self.context = context or {}


class ConfigurationError(WorkflowException):
    """Configuration validation or loading failed."""

    pass


class ImmediateProcessingError(WorkflowException):
    """Direct processing via Responses API failed."""

    pass


class ResultsUnpackError(WorkflowException):
    """Unpacking results to output directory failed."""

    pass


class ValidationError(WorkflowException):
    """Validation check failed."""

    def __init__(self, message: str, validation_type: str, failures: list = None):
        """
        Initialize validation exception.

        Args:
            message: Error description
            validation_type: Type of validation that failed
            failures: List of specific validation failures
        """
        super().__init__(
            message,
            context={
                "validation_type": validation_type,
                "failure_count": len(failures) if failures else 0,
            },
        )
        self.validation_type = validation_type
        self.failures = failures or []


# ============================================================================
# Calibration Target Validation Exceptions
# ============================================================================


class CalibrationTargetValidationError(ValueError):
    """Base class for all calibration target validation errors that should propagate."""

    pass


# Code structure errors
class CodeSyntaxError(CalibrationTargetValidationError):
    """Code has syntax errors."""

    pass


class CodeStructureError(CalibrationTargetValidationError):
    """Code structure is invalid (wrong function name, signature, etc)."""

    pass


# Unit/dimensionality errors
class UnitValidationError(CalibrationTargetValidationError):
    """Base class for unit-related validation errors."""

    pass


class MissingUnitsError(UnitValidationError):
    """Return value missing Pint units."""

    pass


class DimensionalityMismatchError(UnitValidationError):
    """Unit dimensionality doesn't match expected."""

    pass


class UnitConversionError(UnitValidationError):
    """Pint unit conversion/operation failed."""

    pass


# Return value errors
class ReturnValueError(CalibrationTargetValidationError):
    """Base class for return value structure errors."""

    pass


class ScalarReturnError(ReturnValueError):
    """Function returned scalar when array was expected."""

    pass


class ArrayLengthError(ReturnValueError):
    """Array has wrong length."""

    pass


class ReturnStructureError(ReturnValueError):
    """Return value has wrong structure (not dict, missing keys, etc)."""

    pass


# Data consistency errors
class DataConsistencyError(CalibrationTargetValidationError):
    """Base class for data consistency errors."""

    pass


class ComputedValueMismatchError(DataConsistencyError):
    """Computed values don't match reported values."""

    pass


class ScaleMismatchError(DataConsistencyError):
    """Scale mismatch between code output and calibration target."""

    pass


class HardcodedConstantError(DataConsistencyError):
    """Hardcoded numeric constant with units found in measurement_code."""

    pass


# Reference/lookup errors
class CalibrationReferenceError(CalibrationTargetValidationError):
    """Base class for reference lookup errors."""

    pass


class DOIResolutionError(CalibrationReferenceError):
    """DOI failed to resolve."""

    pass


class PaperTitleMismatchError(CalibrationReferenceError):
    """Paper title doesn't match CrossRef metadata."""

    pass


class SourceRefError(CalibrationReferenceError):
    """source_ref not defined in sources."""

    pass


class SpeciesNotFoundError(CalibrationReferenceError):
    """Species not found in model."""

    pass


# Content validation errors
class ContentValidationError(CalibrationTargetValidationError):
    """Base class for content validation errors."""

    pass


class ControlCharacterError(ContentValidationError):
    """Control characters found in text."""

    pass


class EmptyScenarioError(ContentValidationError):
    """Scenario has no measurements."""

    pass
