"""
Unit tests for custom exception hierarchy.

Tests exception creation, context preservation, and inheritance.
"""

from qsp_llm_workflows.core.exceptions import (
    WorkflowException,
    ConfigurationError,
    BatchCreationError,
    BatchUploadError,
    BatchMonitoringError,
    BatchTimeoutError,
    ImmediateProcessingError,
    ResultsUnpackError,
    ValidationError,
)


class TestWorkflowException:
    """Test base WorkflowException."""

    def test_create_basic_exception(self):
        """Test creating exception with message only."""
        exc = WorkflowException("Something went wrong")

        assert str(exc) == "Something went wrong"
        assert exc.context == {}

    def test_create_exception_with_context(self):
        """Test creating exception with context dict."""
        context = {"batch_id": "batch_123", "step": "upload"}
        exc = WorkflowException("Upload failed", context=context)

        assert str(exc) == "Upload failed"
        assert exc.context == context
        assert exc.context["batch_id"] == "batch_123"

    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from WorkflowException."""
        exceptions = [
            ConfigurationError("test"),
            BatchCreationError("test"),
            BatchUploadError("test"),
            BatchMonitoringError("test"),
            BatchTimeoutError("test", "batch_123", 3600),
            ImmediateProcessingError("test"),
            ResultsUnpackError("test"),
            ValidationError("test", "schema"),
        ]

        for exc in exceptions:
            assert isinstance(exc, WorkflowException)
            assert isinstance(exc, Exception)


class TestBatchTimeoutError:
    """Test BatchTimeoutError with specific timeout context."""

    def test_create_timeout_error(self):
        """Test creating timeout error with batch_id and timeout."""
        exc = BatchTimeoutError(
            "Batch timed out after 3600s", batch_id="batch_abc123", timeout=3600
        )

        assert str(exc) == "Batch timed out after 3600s"
        assert exc.batch_id == "batch_abc123"
        assert exc.timeout == 3600
        assert exc.context["batch_id"] == "batch_abc123"
        assert exc.context["timeout_seconds"] == 3600

    def test_timeout_error_inheritance(self):
        """Test that timeout error is a WorkflowException."""
        exc = BatchTimeoutError("Timeout", "batch_123", 7200)

        assert isinstance(exc, BatchTimeoutError)
        assert isinstance(exc, WorkflowException)
        assert isinstance(exc, Exception)


class TestValidationError:
    """Test ValidationError with validation-specific context."""

    def test_create_validation_error_basic(self):
        """Test creating validation error with type only."""
        exc = ValidationError("Schema validation failed", "schema")

        assert str(exc) == "Schema validation failed"
        assert exc.validation_type == "schema"
        assert exc.failures == []
        assert exc.context["validation_type"] == "schema"
        assert exc.context["failure_count"] == 0

    def test_create_validation_error_with_failures(self):
        """Test creating validation error with failure list."""
        failures = [
            "file1.yaml: Missing required field 'parameter_name'",
            "file2.yaml: Invalid type for 'value' field",
        ]
        exc = ValidationError("Schema validation failed", "schema", failures=failures)

        assert exc.validation_type == "schema"
        assert exc.failures == failures
        assert len(exc.failures) == 2
        assert exc.context["failure_count"] == 2

    def test_validation_error_inheritance(self):
        """Test that validation error is a WorkflowException."""
        exc = ValidationError("Validation failed", "doi")

        assert isinstance(exc, ValidationError)
        assert isinstance(exc, WorkflowException)
        assert isinstance(exc, Exception)


class TestExceptionChaining:
    """Test exception chaining for preserving original errors."""

    def test_chain_exceptions(self):
        """Test raising custom exception from original exception."""
        original = ValueError("Invalid input value")

        try:
            try:
                raise original
            except ValueError as e:
                raise BatchCreationError("Failed to create batch") from e
        except BatchCreationError as exc:
            assert str(exc) == "Failed to create batch"
            assert exc.__cause__ is original
            assert isinstance(exc.__cause__, ValueError)

    def test_chain_preserves_original_traceback(self):
        """Test that chaining preserves original exception info."""
        try:
            try:
                # Simulate some nested call stack
                def inner():
                    raise RuntimeError("Original error")

                def outer():
                    inner()

                outer()
            except RuntimeError as e:
                raise BatchUploadError("Upload failed", context={"batch_id": "123"}) from e
        except BatchUploadError as exc:
            assert exc.context["batch_id"] == "123"
            assert exc.__cause__ is not None
            assert str(exc.__cause__) == "Original error"


class TestSpecificExceptions:
    """Test specific exception types have correct behavior."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Missing API key")
        assert str(exc) == "Missing API key"
        assert isinstance(exc, WorkflowException)

    def test_batch_creation_error(self):
        """Test BatchCreationError."""
        exc = BatchCreationError("Invalid CSV format")
        assert str(exc) == "Invalid CSV format"
        assert isinstance(exc, WorkflowException)

    def test_batch_upload_error(self):
        """Test BatchUploadError."""
        exc = BatchUploadError("API connection failed")
        assert str(exc) == "API connection failed"
        assert isinstance(exc, WorkflowException)

    def test_batch_monitoring_error(self):
        """Test BatchMonitoringError."""
        exc = BatchMonitoringError("Failed to retrieve batch status")
        assert str(exc) == "Failed to retrieve batch status"
        assert isinstance(exc, WorkflowException)

    def test_immediate_processing_error(self):
        """Test ImmediateProcessingError."""
        exc = ImmediateProcessingError("Request failed")
        assert str(exc) == "Request failed"
        assert isinstance(exc, WorkflowException)

    def test_results_unpack_error(self):
        """Test ResultsUnpackError."""
        exc = ResultsUnpackError("Failed to write output file")
        assert str(exc) == "Failed to write output file"
        assert isinstance(exc, WorkflowException)
