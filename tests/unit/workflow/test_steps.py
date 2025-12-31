"""
Unit tests for concrete workflow steps.

Tests each workflow step in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch, mock_open, AsyncMock

from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.workflow.steps import (
    UnpackResultsStep,
    ProcessValidationFixStep,
    UnpackValidationFixResultsStep,
)
from qsp_llm_workflows.core.exceptions import (
    ResultsUnpackError,
    ImmediateProcessingError,
)


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
        )
        mock_process.assert_called_once_with(results_file, output_dir, tmp_path / "input.csv")

    def test_missing_results_file(self, tmp_path):
        """Test error when results file is not in context."""
        config = Mock()

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            config=config,
        )
        # No results_file set

        step = UnpackResultsStep()

        with pytest.raises(ResultsUnpackError, match="Results file not found"):
            step.execute(context)


class TestProcessValidationFixStep:
    """Test ProcessValidationFixStep."""

    @patch("qsp_llm_workflows.core.workflow.steps.AsyncOpenAI")
    @patch("qsp_llm_workflows.core.workflow.steps.ValidationFixPromptBuilder")
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
            config=config,
        )
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_dir)
        context.set_metadata("reasoning_effort", "high")

        # Execute
        step = ProcessValidationFixStep()
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

    @patch("qsp_llm_workflows.core.workflow.steps.ValidationFixPromptBuilder")
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
            config=config,
        )
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_dir)

        # Execute
        step = ProcessValidationFixStep()
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
            config=config,
        )
        # No metadata set

        step = ProcessValidationFixStep()

        with pytest.raises(ImmediateProcessingError, match="Missing required metadata"):
            step.execute(context)

    def test_step_name(self):
        """Test step name property."""
        step = ProcessValidationFixStep()
        assert step.name == "Process Validation Fix"


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
