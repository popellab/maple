"""
Unit tests for concrete workflow steps.

Tests each workflow step in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch

from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.workflow.steps import (
    UnpackResultsStep,
)
from qsp_llm_workflows.core.exceptions import (
    ResultsUnpackError,
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
