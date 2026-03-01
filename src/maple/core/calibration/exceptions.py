"""
Calibration target validation exception hierarchy.

Provides specific exception types for calibration target validation failures,
organized by category (code structure, units, return values, data consistency,
references, content).

Each exception class has a `category` attribute for grouping in analytics/reports.
The category is captured in Logfire traces via exception.type, enabling filtering
and aggregation of validation errors by type.

Categories:
    - hallucination: LLM fabricated values not in source text
    - fabrication: LLM fabricated citations/DOIs
    - reference: Internal reference inconsistency
    - code: Custom code syntax/execution errors
    - units: Unit parsing or mismatch errors
    - structural: Schema/structural issues
    - prior: Prior specification issues
    - source_quality: Source reliability issues
    - translation: Translation uncertainty issues
    - encoding: Character encoding issues
"""

from typing import List, Optional


class CalibrationTargetValidationError(ValueError):
    """Base class for all calibration target validation errors that should propagate.

    Attributes:
        category: High-level error category for grouping in reports
        details: List of specific error messages
    """

    category: str = "unknown"

    def __init__(self, message: str, details: Optional[List[str]] = None):
        self.details = details or []
        super().__init__(message)

    @classmethod
    def from_errors(cls, errors: List[str], prefix: str = "") -> "CalibrationTargetValidationError":
        """Create exception from list of error strings."""
        if prefix:
            message = f"{prefix}:\n  - " + "\n  - ".join(errors)
        else:
            message = "\n  - ".join(errors)
        return cls(message, details=errors)


# Alias for backward compatibility
SubmodelTargetValidationError = CalibrationTargetValidationError


# ============================================================================
# Code structure errors
# ============================================================================


class CodeSyntaxError(CalibrationTargetValidationError):
    """Code has syntax errors."""

    category = "code"

    def __init__(self, code_type: str, error_msg: str, code_snippet: str = ""):
        self.code_type = code_type
        self.error_msg = error_msg
        message = f"{code_type} syntax error: {error_msg}"
        if code_snippet:
            message += f"\n\nCode:\n{code_snippet[:200]}..."
        super().__init__(message, details=[error_msg])


class CodeStructureError(CalibrationTargetValidationError):
    """Code structure is invalid (wrong function name, signature, etc)."""

    category = "code"


class CodeExecutionError(CalibrationTargetValidationError):
    """Custom code failed during test execution."""

    category = "code"

    def __init__(self, code_type: str, error_msg: str, error_type: str = ""):
        self.code_type = code_type
        self.error_msg = error_msg
        self.error_type = error_type
        message = f"{code_type} execution error"
        if error_type:
            message += f" ({error_type})"
        message += f": {error_msg}"
        super().__init__(message, details=[error_msg])


class CodeSignatureError(CalibrationTargetValidationError):
    """Custom code has incorrect function signature or return type."""

    category = "code"


# ============================================================================
# Unit/dimensionality errors
# ============================================================================


class UnitValidationError(CalibrationTargetValidationError):
    """Base class for unit-related validation errors."""

    category = "units"


class MissingUnitsError(UnitValidationError):
    """Return value missing Pint units."""

    category = "units"


class DimensionalityMismatchError(UnitValidationError):
    """Unit dimensionality doesn't match expected."""

    category = "units"


class UnitConversionError(UnitValidationError):
    """Pint unit conversion/operation failed."""

    category = "units"


class UnitParsingError(UnitValidationError):
    """A unit string could not be parsed by Pint."""

    category = "units"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "UnitParsingError":
        message = "Invalid Pint units:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


# ============================================================================
# Return value errors
# ============================================================================


class ReturnValueError(CalibrationTargetValidationError):
    """Base class for return value structure errors."""

    category = "code"


class ScalarReturnError(ReturnValueError):
    """Function returned scalar when array was expected."""

    category = "code"


class ArrayLengthError(ReturnValueError):
    """Array has wrong length."""

    category = "code"


class ReturnStructureError(ReturnValueError):
    """Return value has wrong structure (not dict, missing keys, etc)."""

    category = "code"


# ============================================================================
# Data consistency errors
# ============================================================================


class DataConsistencyError(CalibrationTargetValidationError):
    """Base class for data consistency errors."""

    category = "structural"


class ComputedValueMismatchError(DataConsistencyError):
    """Computed values don't match reported values."""

    category = "structural"


class ScaleMismatchError(DataConsistencyError):
    """Scale mismatch between code output and calibration target."""

    category = "structural"


class HardcodedConstantError(DataConsistencyError):
    """Hardcoded numeric constant with units found in measurement_code."""

    category = "code"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "HardcodedConstantError":
        message = "Hardcoded values in code (should use inputs):\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


# ============================================================================
# Reference/lookup errors
# ============================================================================


class CalibrationReferenceError(CalibrationTargetValidationError):
    """Base class for reference lookup errors."""

    category = "reference"


class DOIResolutionError(CalibrationReferenceError):
    """DOI failed to resolve via CrossRef API.

    This catches fabricated citations where the LLM invents a DOI
    that doesn't correspond to a real publication.
    """

    category = "fabrication"

    def __init__(self, doi: str, source_type: str = "Primary"):
        self.doi = doi
        self.source_type = source_type
        message = (
            f"{source_type} source DOI '{doi}' failed to resolve. "
            f"Verify at https://doi.org/{doi}"
        )
        super().__init__(message, details=[message])


class PaperTitleMismatchError(CalibrationReferenceError):
    """Paper title doesn't match CrossRef metadata."""

    category = "fabrication"


class DOIMetadataMismatchError(CalibrationReferenceError):
    """DOI resolved but metadata doesn't match claimed title/year/author.

    This catches cases where the LLM uses a real DOI but misattributes
    the content (wrong paper, wrong year, etc.).
    """

    category = "fabrication"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "DOIMetadataMismatchError":
        message = "DOI metadata mismatches:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class SourceRefError(CalibrationReferenceError):
    """source_ref not defined in sources."""

    category = "reference"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "SourceRefError":
        message = "Invalid source references:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class SnippetValueMismatchError(CalibrationReferenceError):
    """Extracted numeric value not found in the quoted source snippet.

    This is the primary anti-hallucination check. When an LLM extracts a value,
    it must appear verbatim in the quoted text from the paper.
    """

    category = "hallucination"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "SnippetValueMismatchError":
        message = (
            "Value-snippet mismatches (possible hallucination):\n  - "
            + "\n  - ".join(errors)
            + "\n\nCheck that extracted values match the source text exactly."
        )
        return cls(message, details=errors)


class InputReferenceError(CalibrationReferenceError):
    """A uses_inputs or input_ref references an input name that doesn't exist."""

    category = "reference"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "InputReferenceError":
        message = "Invalid input references:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class ParameterReferenceError(CalibrationReferenceError):
    """A model field references a parameter that doesn't exist."""

    category = "reference"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "ParameterReferenceError":
        message = "Invalid parameter references:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class ReferenceRefError(CalibrationReferenceError):
    """A ReferenceRef references a name not found in the reference database."""

    category = "reference"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "ReferenceRefError":
        message = "Invalid reference_ref values:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class StateVariableReferenceError(CalibrationReferenceError):
    """An observable references a state variable that doesn't exist."""

    category = "reference"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "StateVariableReferenceError":
        message = "Invalid state variable references:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class SpeciesNotFoundError(CalibrationReferenceError):
    """Species not found in model."""

    category = "reference"


# ============================================================================
# Content validation errors
# ============================================================================


class ContentValidationError(CalibrationTargetValidationError):
    """Base class for content validation errors."""

    category = "structural"


class ControlCharacterError(ContentValidationError):
    """Control characters found in text."""

    category = "encoding"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "ControlCharacterError":
        message = "Invisible/non-ASCII characters found:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class EmptyScenarioError(ContentValidationError):
    """Scenario has no measurements."""

    category = "structural"


# ============================================================================
# Structural validation errors
# ============================================================================


class SpanOrderingError(CalibrationTargetValidationError):
    """Time span has start >= end."""

    category = "structural"

    def __init__(self, span: list):
        self.span = span
        message = f"Independent variable span must have start < end, got {span}"
        super().__init__(message, details=[message])


class MissingFieldError(CalibrationTargetValidationError):
    """Required field missing for the specified model type."""

    category = "structural"

    @classmethod
    def from_errors(
        cls, errors: List[str], prefix: str = "Missing required fields"
    ) -> "MissingFieldError":
        message = f"{prefix}:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class ObservableConfigError(CalibrationTargetValidationError):
    """Observable configuration is invalid or inconsistent."""

    category = "structural"

    @classmethod
    def from_errors(cls, errors: List[str]) -> "ObservableConfigError":
        message = "Observable configuration errors:\n  - " + "\n  - ".join(errors)
        return cls(message, details=errors)


class EvaluationPointsError(CalibrationTargetValidationError):
    """Evaluation points are outside the time span or invalid."""

    category = "structural"


class SampleSizeError(CalibrationTargetValidationError):
    """Sample size specification is inconsistent with evaluation points."""

    category = "structural"


# ============================================================================
# Prior validation errors
# ============================================================================


class PriorParameterError(CalibrationTargetValidationError):
    """Prior distribution parameters are invalid or missing."""

    category = "prior"

    def __init__(self, distribution: str, error_msg: str):
        self.distribution = distribution
        message = f"Invalid {distribution} prior: {error_msg}"
        super().__init__(message, details=[message])


class PriorScaleError(CalibrationTargetValidationError):
    """Prior scale doesn't match documented translation uncertainty."""

    category = "prior"


# ============================================================================
# Source quality validation errors
# ============================================================================


class TranslationUncertaintyError(CalibrationTargetValidationError):
    """Translation uncertainty not properly documented."""

    category = "translation"


# ============================================================================
# Exception category mapping for external tools
# ============================================================================

EXCEPTION_CATEGORIES = {
    # Hallucination detection
    "SnippetValueMismatchError": "hallucination",
    # Fabrication detection
    "DOIResolutionError": "fabrication",
    "DOIMetadataMismatchError": "fabrication",
    "PaperTitleMismatchError": "fabrication",
    # Reference consistency
    "InputReferenceError": "reference",
    "SourceRefError": "reference",
    "ParameterReferenceError": "reference",
    "ReferenceRefError": "reference",
    "StateVariableReferenceError": "reference",
    "SpeciesNotFoundError": "reference",
    "CalibrationReferenceError": "reference",
    # Code validation
    "CodeSyntaxError": "code",
    "CodeStructureError": "code",
    "CodeExecutionError": "code",
    "CodeSignatureError": "code",
    "HardcodedConstantError": "code",
    "ReturnValueError": "code",
    "ScalarReturnError": "code",
    "ArrayLengthError": "code",
    "ReturnStructureError": "code",
    # Unit validation
    "UnitValidationError": "units",
    "UnitParsingError": "units",
    "MissingUnitsError": "units",
    "DimensionalityMismatchError": "units",
    "UnitConversionError": "units",
    # Structural validation
    "SpanOrderingError": "structural",
    "MissingFieldError": "structural",
    "ObservableConfigError": "structural",
    "EvaluationPointsError": "structural",
    "SampleSizeError": "structural",
    "DataConsistencyError": "structural",
    "ComputedValueMismatchError": "structural",
    "ScaleMismatchError": "structural",
    "ContentValidationError": "structural",
    "EmptyScenarioError": "structural",
    # Prior validation
    "PriorParameterError": "prior",
    "PriorScaleError": "prior",
    # Source quality
    "TranslationUncertaintyError": "translation",
    # Encoding
    "ControlCharacterError": "encoding",
}


def categorize_exception(exception_type: str) -> str:
    """
    Get category from exception type name.

    Args:
        exception_type: Exception class name (may include module path)

    Returns:
        Category string, or "other" if not found
    """
    for exc_name, category in EXCEPTION_CATEGORIES.items():
        if exc_name in exception_type:
            return category
    return "other"
