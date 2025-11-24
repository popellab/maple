"""
Unit tests for WorkflowContext.

Tests the context object that holds workflow state between steps.
"""
from pathlib import Path
from unittest.mock import Mock

from qsp_llm_workflows.core.workflow.context import WorkflowContext


class TestWorkflowContext:
    """Test WorkflowContext creation and methods."""

    def test_create_context(self, tmp_path):
        """Test creating workflow context."""
        input_csv = tmp_path / "input.csv"
        config = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )

        assert context.input_csv == input_csv
        assert context.workflow_type == "parameter"
        assert context.immediate is True
        assert context.config == config
        assert context.progress_callback is None
        assert context.batch_file is None
        assert context.batch_id is None
        assert context.results_file is None
        assert context.output_directory is None
        assert context.file_count == 0
        assert context.metadata == {}

    def test_create_context_with_callback(self, tmp_path):
        """Test creating context with progress callback."""
        input_csv = tmp_path / "input.csv"
        config = Mock()
        callback = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=False,
            config=config,
            progress_callback=callback,
        )

        assert context.progress_callback == callback

    def test_report_progress_with_callback(self, tmp_path):
        """Test reporting progress when callback is set."""
        input_csv = tmp_path / "input.csv"
        config = Mock()
        callback = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
            config=config,
            progress_callback=callback,
        )

        context.report_progress("Test message")

        callback.assert_called_once_with("Test message")

    def test_report_progress_without_callback(self, tmp_path):
        """Test reporting progress when callback is None."""
        input_csv = tmp_path / "input.csv"
        config = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )

        # Should not raise an error
        context.report_progress("Test message")

    def test_set_and_get_metadata(self, tmp_path):
        """Test setting and getting metadata."""
        input_csv = tmp_path / "input.csv"
        config = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )

        context.set_metadata("key1", "value1")
        context.set_metadata("key2", 42)

        assert context.get_metadata("key1") == "value1"
        assert context.get_metadata("key2") == 42

    def test_get_metadata_with_default(self, tmp_path):
        """Test getting metadata with default value."""
        input_csv = tmp_path / "input.csv"
        config = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )

        assert context.get_metadata("nonexistent") is None
        assert context.get_metadata("nonexistent", "default") == "default"

    def test_modify_context_state(self, tmp_path):
        """Test modifying context state during workflow."""
        input_csv = tmp_path / "input.csv"
        config = Mock()

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=False,
            config=config,
        )

        # Simulate workflow steps modifying context
        context.batch_file = tmp_path / "batch.jsonl"
        context.batch_id = "batch_123"
        context.results_file = tmp_path / "results.jsonl"
        context.output_directory = tmp_path / "output"
        context.file_count = 15

        assert context.batch_file == tmp_path / "batch.jsonl"
        assert context.batch_id == "batch_123"
        assert context.results_file == tmp_path / "results.jsonl"
        assert context.output_directory == tmp_path / "output"
        assert context.file_count == 15
