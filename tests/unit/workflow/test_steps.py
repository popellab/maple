"""
Unit tests for concrete workflow steps.

Tests each workflow step in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, mock_open

from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.workflow.steps import (
    CreateBatchStep,
    UploadBatchStep,
    UnpackResultsStep,
)
from qsp_llm_workflows.core.exceptions import (
    BatchCreationError,
    BatchUploadError,
    ResultsUnpackError,
)


class TestCreateBatchStep:
    """Test CreateBatchStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.ParameterBatchCreator")
    def test_create_parameter_batch(self, mock_creator_class, tmp_path):
        """Test creating parameter batch."""
        # Setup
        config = Mock()
        config.base_dir = tmp_path

        input_csv = tmp_path / "input.csv"
        batch_file = tmp_path / "batch.jsonl"
        batch_file.touch()

        mock_creator = Mock()
        mock_creator.run.return_value = batch_file
        mock_creator_class.return_value = mock_creator

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=False,
            config=config,
        )

        # Execute
        step = CreateBatchStep()
        result = step.execute(context)

        # Verify
        assert result.batch_file == batch_file
        mock_creator_class.assert_called_once_with(tmp_path)
        mock_creator.run.assert_called_once_with(None, input_csv)

    @patch("qsp_llm_workflows.core.workflow.steps.TestStatisticBatchCreator")
    def test_create_test_statistic_batch(self, mock_creator_class, tmp_path):
        """Test creating test statistic batch."""
        # Setup
        config = Mock()
        config.base_dir = tmp_path

        input_csv = tmp_path / "input.csv"
        batch_file = tmp_path / "batch.jsonl"
        batch_file.touch()

        mock_creator = Mock()
        mock_creator.run.return_value = batch_file
        mock_creator_class.return_value = mock_creator

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="test_statistic",
            immediate=False,
            config=config,
        )

        # Execute
        step = CreateBatchStep()
        result = step.execute(context)

        # Verify
        assert result.batch_file == batch_file
        mock_creator_class.assert_called_once_with(tmp_path)

    def test_unknown_workflow_type(self, tmp_path):
        """Test error handling for unknown workflow type."""
        config = Mock()
        config.base_dir = tmp_path

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="invalid",
            immediate=False,
            config=config,
        )

        step = CreateBatchStep()

        with pytest.raises(BatchCreationError, match="Unknown workflow type"):
            step.execute(context)

    def test_step_name(self):
        """Test step name property."""
        step = CreateBatchStep()
        assert step.name == "Create Batch"


class TestUploadBatchStep:
    """Test UploadBatchStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.OpenAI")
    @patch("builtins.open", new_callable=mock_open, read_data=b"batch data")
    def test_upload_batch(self, mock_file, mock_openai_class, tmp_path):
        """Test uploading batch to OpenAI."""
        # Setup
        config = Mock()
        config.openai_api_key = "test-key"
        config.batch_completion_window = "24h"

        batch_file = tmp_path / "batch.jsonl"
        batch_file.touch()

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_batch_input = Mock()
        mock_batch_input.id = "file-123"
        mock_client.files.create.return_value = mock_batch_input

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=False,
            config=config,
        )
        context.batch_file = batch_file

        # Execute
        step = UploadBatchStep()
        result = step.execute(context)

        # Verify
        assert result.batch_id == "batch-456"
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once_with(
            input_file_id="file-123",
            endpoint="/v1/responses",
            completion_window="24h",
        )

    def test_missing_batch_file(self, tmp_path):
        """Test error when batch file is not in context."""
        config = Mock()

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=False,
            config=config,
        )
        # No batch_file set

        step = UploadBatchStep()

        with pytest.raises(BatchUploadError, match="Batch file not found"):
            step.execute(context)


class TestUnpackResultsStep:
    """Test UnpackResultsStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.process_results")
    @patch("qsp_llm_workflows.core.workflow.steps.create_unique_output_directory")
    def test_unpack_results(self, mock_create_dir, mock_process, tmp_path):
        """Test unpacking results to unique directory."""
        # Setup
        config = Mock()
        config.to_review_dir = tmp_path / "to-review"

        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        output_dir = tmp_path / "to-review" / "20251123_143022_parameter_immediate"
        output_dir.mkdir(parents=True)

        # Create some YAML files
        (output_dir / "file1.yaml").touch()
        (output_dir / "file2.yaml").touch()
        (output_dir / "file3.yaml").touch()

        mock_create_dir.return_value = output_dir

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.results_file = results_file

        # Execute
        step = UnpackResultsStep()
        result = step.execute(context)

        # Verify
        assert result.output_directory == output_dir
        assert result.file_count == 3
        mock_create_dir.assert_called_once_with(
            base_dir=config.to_review_dir,
            workflow_type="parameter",
            immediate=True,
            batch_id=None,
        )
        mock_process.assert_called_once_with(results_file, output_dir, tmp_path / "input.csv")

    def test_missing_results_file(self, tmp_path):
        """Test error when results file is not in context."""
        config = Mock()

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        # No results_file set

        step = UnpackResultsStep()

        with pytest.raises(ResultsUnpackError, match="Results file not found"):
            step.execute(context)
