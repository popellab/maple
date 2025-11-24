"""
Unit tests for concrete workflow steps.

Tests each workflow step in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, mock_open, AsyncMock

from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.workflow.steps import (
    CreateBatchStep,
    UploadBatchStep,
    UnpackResultsStep,
    CreateValidationFixBatchStep,
    ProcessImmediateValidationFixStep,
    UnpackValidationFixResultsStep,
)
from qsp_llm_workflows.core.exceptions import (
    BatchCreationError,
    BatchUploadError,
    ResultsUnpackError,
    ImmediateProcessingError,
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

    @patch("qsp_llm_workflows.core.workflow.steps.ParameterBatchCreator")
    def test_preview_mode_creates_preview_file(self, mock_creator_class, tmp_path):
        """Test that preview mode creates a human-readable preview file."""
        # Setup
        config = Mock()
        config.base_dir = tmp_path

        input_csv = tmp_path / "input.csv"
        batch_file = tmp_path / "batch.jsonl"

        # Create a sample batch file with a request
        batch_file.write_text(
            '{"custom_id":"test_001","method":"POST","url":"/v1/responses",'
            '"body":{"model":"gpt-5","input":"Test prompt content"}}\n'
        )

        mock_creator = Mock()
        mock_creator.run.return_value = batch_file
        mock_creator_class.return_value = mock_creator

        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type="parameter",
            immediate=False,
            config=config,
        )
        context.set_metadata("preview_prompts", True)

        # Execute
        step = CreateBatchStep()
        result = step.execute(context)

        # Verify
        assert result.batch_file == batch_file

        # Check that preview file was created
        preview_file = batch_file.with_suffix(".preview.txt")
        assert preview_file.exists()

        # Check preview file content
        preview_content = preview_file.read_text()
        assert "PROMPT PREVIEW" in preview_content
        assert "REQUEST 1/1" in preview_content
        assert "Custom ID: test_001" in preview_content
        assert "Model: gpt-5" in preview_content
        assert "Test prompt content" in preview_content


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


class TestCreateValidationFixBatchStep:
    """Test CreateValidationFixBatchStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.ValidationFixBatchCreator")
    def test_create_validation_fix_batch(self, mock_creator_class, tmp_path):
        """Test creating validation fix batch for parameter workflow."""
        # Setup
        config = Mock()
        config.batch_jobs_dir = tmp_path / "batch_jobs"
        config.batch_jobs_dir.mkdir()

        data_dir = tmp_path / "data"
        validation_dir = tmp_path / "validation_results"

        batch_file = (
            config.batch_jobs_dir / "validation_fix_parameter_20251124_120000_requests.jsonl"
        )
        batch_file.touch()

        mock_creator = Mock()
        mock_creator.run.return_value = batch_file
        mock_creator_class.return_value = mock_creator

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=False,
            config=config,
        )
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_dir)

        # Execute
        with patch("qsp_llm_workflows.core.workflow.steps.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20251124_120000"
            step = CreateValidationFixBatchStep()
            result = step.execute(context)

        # Verify
        assert result.batch_file == batch_file
        mock_creator_class.assert_called_once()

    def test_missing_metadata(self, tmp_path):
        """Test error when required metadata is missing."""
        config = Mock()
        config.batch_jobs_dir = tmp_path

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=False,
            config=config,
        )
        # No metadata set

        step = CreateValidationFixBatchStep()

        with pytest.raises(BatchCreationError, match="Missing required metadata"):
            step.execute(context)

    def test_unknown_workflow_type(self, tmp_path):
        """Test error for unknown workflow type."""
        config = Mock()
        config.batch_jobs_dir = tmp_path

        context = WorkflowContext(
            input_csv=None,
            workflow_type="invalid",
            immediate=False,
            config=config,
        )
        context.set_metadata("data_dir", tmp_path)
        context.set_metadata("validation_results_dir", tmp_path)

        step = CreateValidationFixBatchStep()

        with pytest.raises(BatchCreationError, match="Unknown workflow type"):
            step.execute(context)

    def test_step_name(self):
        """Test step name property."""
        step = CreateValidationFixBatchStep()
        assert step.name == "Create Validation Fix Batch"

    @patch("qsp_llm_workflows.core.workflow.steps.ValidationFixBatchCreator")
    def test_preview_mode_creates_preview_file(self, mock_creator_class, tmp_path):
        """Test that preview mode creates a human-readable preview file."""
        # Setup
        config = Mock()
        config.batch_jobs_dir = tmp_path / "batch_jobs"
        config.batch_jobs_dir.mkdir()

        data_dir = tmp_path / "data"
        validation_dir = tmp_path / "validation_results"

        batch_file = (
            config.batch_jobs_dir / "validation_fix_parameter_20251124_120000_requests.jsonl"
        )

        # Create a sample batch file with a validation fix request
        batch_file.write_text(
            '{"custom_id":"file1","method":"POST","url":"/v1/responses",'
            '"body":{"model":"gpt-5","input":"Fix this validation error"}}\n'
        )

        mock_creator = Mock()
        mock_creator.run.return_value = batch_file
        mock_creator_class.return_value = mock_creator

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=False,
            config=config,
        )
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_dir)
        context.set_metadata("preview_prompts", True)

        # Execute
        with patch("qsp_llm_workflows.core.workflow.steps.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20251124_120000"
            step = CreateValidationFixBatchStep()
            result = step.execute(context)

        # Verify
        assert result.batch_file == batch_file

        # Check that preview file was created
        preview_file = batch_file.with_suffix(".preview.txt")
        assert preview_file.exists()

        # Check preview file content
        preview_content = preview_file.read_text()
        assert "VALIDATION FIX PROMPT PREVIEW" in preview_content
        assert "REQUEST 1/1" in preview_content
        assert "Custom ID: file1" in preview_content
        assert "Model: gpt-5" in preview_content
        assert "Fix this validation error" in preview_content


class TestProcessImmediateValidationFixStep:
    """Test ProcessImmediateValidationFixStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.AsyncOpenAI")
    @patch("qsp_llm_workflows.core.workflow.steps.ValidationFixBatchCreator")
    def test_process_validation_fixes(self, mock_creator_class, mock_openai_class, tmp_path):
        """Test processing validation fixes via Responses API."""
        # Setup
        config = Mock()
        config.batch_jobs_dir = tmp_path / "batch_jobs"
        config.batch_jobs_dir.mkdir()
        config.openai_api_key = "test-key"

        data_dir = tmp_path / "data"
        validation_dir = tmp_path / "validation_results"

        # Mock creator with correct Responses API structure
        mock_creator = Mock()
        mock_requests = [
            {
                "custom_id": "file1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5",
                    "input": "fix this",
                    "reasoning": {"effort": "high"},
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "parametermetadata",
                            "strict": True,
                            "schema": {},
                        }
                    },
                },
            }
        ]
        mock_creator.create_batch_requests.return_value = mock_requests
        mock_creator_class.return_value = mock_creator

        # Mock AsyncOpenAI response with Pydantic model (same as extraction workflow)
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_parsed = Mock()
        mock_parsed.model_dump.return_value = {"fixed": True}
        mock_response.output_parsed = mock_parsed
        # Make parse return an awaitable
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_dir)
        context.set_metadata("reasoning_effort", "high")

        # Execute
        step = ProcessImmediateValidationFixStep()
        result = step.execute(context)

        # Verify
        assert result.results_file is not None
        assert result.results_file.exists()

        # Verify correct Responses API call with Pydantic model (same as extraction workflow)
        from qsp_llm_workflows.core.pydantic_models import ParameterMetadata

        mock_client.responses.parse.assert_called_once_with(
            model="gpt-5",
            input="fix this",
            reasoning={"effort": "high"},
            text_format=ParameterMetadata,  # Direct Pydantic model
        )

    @patch("qsp_llm_workflows.core.workflow.steps.ValidationFixBatchCreator")
    def test_no_validation_errors(self, mock_creator_class, tmp_path):
        """Test handling when no validation errors exist."""
        # Setup
        config = Mock()
        config.batch_jobs_dir = tmp_path / "batch_jobs"
        config.batch_jobs_dir.mkdir()

        data_dir = tmp_path / "data"
        validation_dir = tmp_path / "validation_results"

        mock_creator = Mock()
        mock_creator.create_batch_requests.return_value = []  # No errors
        mock_creator_class.return_value = mock_creator

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_dir)

        # Execute
        step = ProcessImmediateValidationFixStep()
        result = step.execute(context)

        # Verify
        assert result.results_file is not None
        assert result.results_file.exists()

    def test_missing_metadata(self, tmp_path):
        """Test error when required metadata is missing."""
        config = Mock()

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        # No metadata set

        step = ProcessImmediateValidationFixStep()

        with pytest.raises(ImmediateProcessingError, match="Missing required metadata"):
            step.execute(context)

    def test_step_name(self):
        """Test step name property."""
        step = ProcessImmediateValidationFixStep()
        assert step.name == "Process Immediate Validation Fix"


class TestUnpackValidationFixResultsStep:
    """Test UnpackValidationFixResultsStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.process_results")
    def test_unpack_to_original_directory(self, mock_process, tmp_path):
        """Test unpacking results back to original data directory."""
        # Setup
        config = Mock()

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.yaml").touch()
        (data_dir / "file2.yaml").touch()

        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.results_file = results_file
        context.set_metadata("data_dir", data_dir)

        # Execute
        step = UnpackValidationFixResultsStep()
        result = step.execute(context)

        # Verify
        assert result.output_directory == data_dir
        assert result.file_count == 2
        mock_process.assert_called_once_with(results_file, data_dir, input_csv=None)

    def test_missing_results_file(self, tmp_path):
        """Test error when results file is not in context."""
        config = Mock()

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        # No results_file set

        step = UnpackValidationFixResultsStep()

        with pytest.raises(ResultsUnpackError, match="Results file not found"):
            step.execute(context)

    def test_missing_data_dir(self, tmp_path):
        """Test error when data_dir metadata is missing."""
        config = Mock()

        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        context = WorkflowContext(
            input_csv=None,
            workflow_type="parameter",
            immediate=True,
            config=config,
        )
        context.results_file = results_file
        # No data_dir metadata

        step = UnpackValidationFixResultsStep()

        with pytest.raises(ResultsUnpackError, match="Data directory not found"):
            step.execute(context)

    def test_step_name(self):
        """Test step name property."""
        step = UnpackValidationFixResultsStep()
        assert step.name == "Unpack Validation Fix Results"
