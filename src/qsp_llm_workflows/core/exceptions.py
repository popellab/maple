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


class BatchCreationError(WorkflowException):
    """Batch request creation from CSV failed."""

    pass


class BatchUploadError(WorkflowException):
    """Batch upload to OpenAI API failed."""

    pass


class BatchMonitoringError(WorkflowException):
    """Batch monitoring or status retrieval failed."""

    pass


class BatchTimeoutError(WorkflowException):
    """Batch did not complete within timeout period."""

    def __init__(self, message: str, batch_id: str, timeout: int):
        """
        Initialize timeout exception.

        Args:
            message: Error description
            batch_id: ID of the batch that timed out
            timeout: Timeout value in seconds
        """
        super().__init__(message, context={"batch_id": batch_id, "timeout_seconds": timeout})
        self.batch_id = batch_id
        self.timeout = timeout


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
