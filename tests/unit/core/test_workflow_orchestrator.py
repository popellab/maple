"""
Unit tests for WorkflowOrchestrator.

Tests the orchestrator's step selection and execution logic.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator, WorkflowResult
from qsp_llm_workflows.core.config import WorkflowConfig
from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.workflow.steps import (
    CreateBatchStep,
    UploadBatchStep,
    MonitorBatchStep,
    ProcessImmediateStep,
    UnpackResultsStep,
)


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator initialization and configuration."""

    def test_init(self, tmp_path):
        """Test orchestrator initialization."""
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )

        orchestrator = WorkflowOrchestrator(config)

        assert orchestrator.config == config
        # Directories should be created
        assert config.batch_jobs_dir.exists()
        assert config.to_review_dir.exists()

    def test_get_workflow_steps_immediate(self, tmp_path):
        """Test getting workflow steps for immediate mode."""
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )
        orchestrator = WorkflowOrchestrator(config)

        steps = orchestrator._get_workflow_steps(immediate=True)

        assert len(steps) == 2
        assert isinstance(steps[0], ProcessImmediateStep)
        assert isinstance(steps[1], UnpackResultsStep)

    def test_get_workflow_steps_batch(self, tmp_path):
        """Test getting workflow steps for batch mode."""
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )
        orchestrator = WorkflowOrchestrator(config)

        steps = orchestrator._get_workflow_steps(immediate=False)

        assert len(steps) == 4
        assert isinstance(steps[0], CreateBatchStep)
        assert isinstance(steps[1], UploadBatchStep)
        assert isinstance(steps[2], MonitorBatchStep)
        assert isinstance(steps[3], UnpackResultsStep)


class TestWorkflowExecution:
    """Test workflow execution with mocked steps."""

    @patch("qsp_llm_workflows.core.workflow_orchestrator.ProcessImmediateStep")
    @patch("qsp_llm_workflows.core.workflow_orchestrator.UnpackResultsStep")
    def test_run_immediate_workflow_success(self, mock_unpack_class, mock_process_class, tmp_path):
        """Test running immediate workflow successfully."""
        # Setup config
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )

        # Mock steps
        mock_process = Mock()
        mock_unpack = Mock()

        def process_execute(context):
            context.results_file = tmp_path / "results.jsonl"
            return context

        def unpack_execute(context):
            context.output_directory = tmp_path / "output"
            context.file_count = 5
            return context

        mock_process.execute = process_execute
        mock_unpack.execute = unpack_execute

        mock_process_class.return_value = mock_process
        mock_unpack_class.return_value = mock_unpack

        # Execute
        orchestrator = WorkflowOrchestrator(config)
        input_csv = tmp_path / "input.csv"
        input_csv.touch()

        result = orchestrator.run_complete_workflow(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
        )

        # Verify
        assert result.status == "success"
        assert result.workflow_type == "parameter"
        assert result.immediate_mode is True
        assert result.file_count == 5
        assert result.error is None
        assert result.duration_seconds is not None

    @patch("qsp_llm_workflows.core.workflow_orchestrator.CreateBatchStep")
    @patch("qsp_llm_workflows.core.workflow_orchestrator.UploadBatchStep")
    @patch("qsp_llm_workflows.core.workflow_orchestrator.MonitorBatchStep")
    @patch("qsp_llm_workflows.core.workflow_orchestrator.UnpackResultsStep")
    def test_run_batch_workflow_success(
        self, mock_unpack_class, mock_monitor_class, mock_upload_class, mock_create_class, tmp_path
    ):
        """Test running batch workflow successfully."""
        # Setup config
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )

        # Mock steps
        mock_create = Mock()
        mock_upload = Mock()
        mock_monitor = Mock()
        mock_unpack = Mock()

        def create_execute(context):
            context.batch_file = tmp_path / "batch.jsonl"
            return context

        def upload_execute(context):
            context.batch_id = "batch_123"
            return context

        def monitor_execute(context):
            context.results_file = tmp_path / "results.jsonl"
            return context

        def unpack_execute(context):
            context.output_directory = tmp_path / "output"
            context.file_count = 10
            return context

        mock_create.execute = create_execute
        mock_upload.execute = upload_execute
        mock_monitor.execute = monitor_execute
        mock_unpack.execute = unpack_execute

        mock_create_class.return_value = mock_create
        mock_upload_class.return_value = mock_upload
        mock_monitor_class.return_value = mock_monitor
        mock_unpack_class.return_value = mock_unpack

        # Execute
        orchestrator = WorkflowOrchestrator(config)
        input_csv = tmp_path / "input.csv"
        input_csv.touch()

        result = orchestrator.run_complete_workflow(
            input_csv=input_csv,
            workflow_type="test_statistic",
            immediate=False,
        )

        # Verify
        assert result.status == "success"
        assert result.workflow_type == "test_statistic"
        assert result.immediate_mode is False
        assert result.batch_id == "batch_123"
        assert result.file_count == 10
        assert result.error is None

    @patch("qsp_llm_workflows.core.workflow_orchestrator.ProcessImmediateStep")
    def test_run_workflow_failure(self, mock_process_class, tmp_path):
        """Test workflow failure handling."""
        # Setup config
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )

        # Mock step that raises exception
        mock_process = Mock()
        mock_process.execute.side_effect = RuntimeError("API error")
        mock_process_class.return_value = mock_process

        # Execute
        orchestrator = WorkflowOrchestrator(config)
        input_csv = tmp_path / "input.csv"
        input_csv.touch()

        result = orchestrator.run_complete_workflow(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=True,
        )

        # Verify failure is captured
        assert result.status == "failed"
        assert result.error == "API error"
        assert result.duration_seconds is not None

    def test_progress_callback(self, tmp_path):
        """Test that progress callback is passed to steps."""
        config = WorkflowConfig(
            base_dir=tmp_path / "workflows",
            storage_dir=tmp_path / "storage",
            openai_api_key="test-key",
        )

        callback = Mock()

        with patch("qsp_llm_workflows.core.workflow_orchestrator.ProcessImmediateStep") as mock_process_class:
            with patch("qsp_llm_workflows.core.workflow_orchestrator.UnpackResultsStep") as mock_unpack_class:
                # Mock successful execution
                def execute_with_callback(context):
                    # Verify callback is in context
                    assert context.progress_callback == callback
                    context.report_progress("Test message")
                    context.results_file = tmp_path / "results.jsonl"
                    context.output_directory = tmp_path / "output"
                    context.file_count = 1
                    return context

                mock_process = Mock()
                mock_process.execute = execute_with_callback
                mock_process_class.return_value = mock_process

                mock_unpack = Mock()
                mock_unpack.execute = execute_with_callback
                mock_unpack_class.return_value = mock_unpack

                # Execute
                orchestrator = WorkflowOrchestrator(config)
                input_csv = tmp_path / "input.csv"
                input_csv.touch()

                orchestrator.run_complete_workflow(
                    input_csv=input_csv,
                    workflow_type="parameter",
                    immediate=True,
                    progress_callback=callback,
                )

                # Verify callback was called
                callback.assert_called()


class TestWorkflowResult:
    """Test WorkflowResult creation."""

    def test_create_success_result(self, tmp_path):
        """Test creating success result from context."""
        config = Mock()
        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.output_directory = tmp_path / "output"
        context.file_count = 5
        context.batch_id = "batch_123"
        context.set_metadata("started_at", "2025-11-23T14:30:00")

        result = WorkflowResult(context, duration=10.5)

        assert result.status == "success"
        assert result.workflow_type == "parameter"
        assert result.immediate_mode is True
        assert result.file_count == 5
        assert result.batch_id == "batch_123"
        assert result.error is None
        assert result.duration_seconds == 10.5
        assert result.started_at == "2025-11-23T14:30:00"

    def test_create_error_result(self, tmp_path):
        """Test creating error result from exception."""
        config = Mock()
        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.set_metadata("started_at", "2025-11-23T14:30:00")

        error = RuntimeError("Test error")
        result = WorkflowResult.from_error(context, error, duration=5.2)

        assert result.status == "failed"
        assert result.error == "Test error"
        assert result.duration_seconds == 5.2
