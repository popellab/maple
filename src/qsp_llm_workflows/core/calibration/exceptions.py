"""
Calibration target validation exception hierarchy.

Provides specific exception types for calibration target validation failures,
organized by category (code structure, units, return values, data consistency,
references, content).
"""


class CalibrationTargetValidationError(ValueError):
    """Base class for all calibration target validation errors that should propagate."""

    pass


# ============================================================================
# Code structure errors
# ============================================================================


class CodeSyntaxError(CalibrationTargetValidationError):
    """Code has syntax errors."""

    pass


class CodeStructureError(CalibrationTargetValidationError):
    """Code structure is invalid (wrong function name, signature, etc)."""

    pass


# ============================================================================
# Unit/dimensionality errors
# ============================================================================


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


# ============================================================================
# Return value errors
# ============================================================================


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


# ============================================================================
# Data consistency errors
# ============================================================================


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


# ============================================================================
# Reference/lookup errors
# ============================================================================


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


class SnippetValueMismatchError(CalibrationReferenceError):
    """Value not found in value_snippet text."""

    pass


class SpeciesNotFoundError(CalibrationReferenceError):
    """Species not found in model."""

    pass


# ============================================================================
# Content validation errors
# ============================================================================


class ContentValidationError(CalibrationTargetValidationError):
    """Base class for content validation errors."""

    pass


class ControlCharacterError(ContentValidationError):
    """Control characters found in text."""

    pass


class EmptyScenarioError(ContentValidationError):
    """Scenario has no measurements."""

    pass
