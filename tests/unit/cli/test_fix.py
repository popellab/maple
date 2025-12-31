"""
Unit tests for qsp-fix CLI.

Tests the validation fix CLI in isolation with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, patch
from io import StringIO


class TestFixCLI:
    """Test qsp-fix CLI."""

    @patch("qsp_llm_workflows.core.workflow_orchestrator.WorkflowOrchestrator")
    @patch("qsp_llm_workflows.core.config.WorkflowConfig")
    @patch("qsp_llm_workflows.cli.fix.load_api_key")
    def test_fix_with_custom_dir(
        self, mock_load_api_key, mock_config_class, mock_orchestrator_class, tmp_path
    ):
        """Test qsp-fix with custom directory."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()

        mock_load_api_key.return_value = "test-api-key"

        # Mock config
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # Mock orchestrator and result
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.status = "success"
        mock_result.file_count = 5
        mock_result.output_directory = str(data_dir)
        mock_result.duration_seconds = 10.5
        mock_orchestrator.run_validation_fix_workflow.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create test args
        from argparse import Namespace

        args = Namespace(
            workflow_type="parameter_estimates",
            dir=str(data_dir),
            preview_prompts=False,
            validation_results_dir=str(validation_dir),
        )

        # Import main here to avoid running it at import time
        from qsp_llm_workflows.cli.fix import main

        # Patch sys.argv and run
        with patch(
            "sys.argv", ["qsp-fix", "parameter_estimates", "--immediate", "--dir", str(data_dir)]
        ):
            with patch("argparse.ArgumentParser.parse_args", return_value=args):
                with patch("sys.stdout", new_callable=StringIO):
                    main()

        # Verify config was created correctly
        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args.kwargs
        assert call_kwargs["openai_api_key"] == "test-api-key"
        # base_dir and storage_dir should be set
        assert "base_dir" in call_kwargs
        assert "storage_dir" in call_kwargs

        # Verify orchestrator was called with correct parameters
        mock_orchestrator.run_validation_fix_workflow.assert_called_once()
        call_kwargs = mock_orchestrator.run_validation_fix_workflow.call_args.kwargs
        assert call_kwargs["data_dir"] == data_dir
        assert call_kwargs["workflow_type"] == "parameter"  # Converted from parameter_estimates
        assert "validation_results_dir" in call_kwargs
        assert "progress_callback" in call_kwargs

    @patch("qsp_llm_workflows.cli.fix.load_api_key")
    def test_fix_missing_validation_results(self, mock_load_api_key, tmp_path):
        """Test error when validation results directory doesn't exist."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Intentionally don't create validation_dir to test error handling
        nonexistent_validation = tmp_path / "nonexistent_validation"

        mock_load_api_key.return_value = "test-api-key"

        # Create test args
        from argparse import Namespace

        args = Namespace(
            workflow_type="parameter_estimates",
            dir=str(data_dir),
            preview_prompts=False,
            validation_results_dir=str(nonexistent_validation),
        )

        # Import main
        from qsp_llm_workflows.cli.fix import main

        # Patch sys.argv and run - should exit with error
        with patch(
            "sys.argv", ["qsp-fix", "parameter_estimates", "--immediate", "--dir", str(data_dir)]
        ):
            with patch("argparse.ArgumentParser.parse_args", return_value=args):
                with patch("sys.stdout", new_callable=StringIO):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    # Should exit with code 1 (error)
                    assert exc_info.value.code == 1

    @patch("qsp_llm_workflows.cli.fix.load_api_key")
    def test_fix_missing_data_dir(self, mock_load_api_key, tmp_path):
        """Test error when data directory doesn't exist."""
        # Setup
        nonexistent_dir = tmp_path / "nonexistent"
        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()  # Create validation dir but not data dir

        mock_load_api_key.return_value = "test-api-key"

        # Create test args
        from argparse import Namespace

        args = Namespace(
            workflow_type="parameter_estimates",
            dir=str(nonexistent_dir),
            preview_prompts=False,
            validation_results_dir=str(validation_dir),
        )

        # Import main
        from qsp_llm_workflows.cli.fix import main

        # Patch sys.argv and run - should exit with error
        with patch(
            "sys.argv",
            ["qsp-fix", "parameter_estimates", "--immediate", "--dir", str(nonexistent_dir)],
        ):
            with patch("argparse.ArgumentParser.parse_args", return_value=args):
                with patch("sys.stdout", new_callable=StringIO):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    # Should exit with code 1 (error)
                    assert exc_info.value.code == 1

    @patch("qsp_llm_workflows.core.workflow_orchestrator.WorkflowOrchestrator")
    @patch("qsp_llm_workflows.core.config.WorkflowConfig")
    @patch("qsp_llm_workflows.cli.fix.load_api_key")
    def test_fix_workflow_failure(
        self, mock_load_api_key, mock_config_class, mock_orchestrator_class, tmp_path
    ):
        """Test handling of workflow failure."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()

        mock_load_api_key.return_value = "test-api-key"

        # Mock config
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # Mock orchestrator with failure result
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.status = "failed"
        mock_result.error = "Test error message"
        mock_orchestrator.run_validation_fix_workflow.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create test args
        from argparse import Namespace

        args = Namespace(
            workflow_type="parameter_estimates",
            dir=str(data_dir),
            preview_prompts=False,
            validation_results_dir=str(validation_dir),
        )

        # Import main
        from qsp_llm_workflows.cli.fix import main

        # Patch sys.argv and run - should exit with error
        with patch(
            "sys.argv", ["qsp-fix", "parameter_estimates", "--immediate", "--dir", str(data_dir)]
        ):
            with patch("argparse.ArgumentParser.parse_args", return_value=args):
                with patch("sys.stdout", new_callable=StringIO):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

                    # Should exit with code 1 (error)
                    assert exc_info.value.code == 1

    @patch("qsp_llm_workflows.core.workflow_orchestrator.WorkflowOrchestrator")
    @patch("qsp_llm_workflows.core.config.WorkflowConfig")
    @patch("qsp_llm_workflows.cli.fix.load_api_key")
    def test_fix_test_statistics_workflow(
        self, mock_load_api_key, mock_config_class, mock_orchestrator_class, tmp_path
    ):
        """Test qsp-fix with test_statistics workflow type."""
        # Setup
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        validation_dir = tmp_path / "validation_results"
        validation_dir.mkdir()

        mock_load_api_key.return_value = "test-api-key"

        # Mock config
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_result = Mock()
        mock_result.status = "success"
        mock_result.file_count = 3
        mock_result.output_directory = str(data_dir)
        mock_result.duration_seconds = 5.2
        mock_orchestrator.run_validation_fix_workflow.return_value = mock_result
        mock_orchestrator_class.return_value = mock_orchestrator

        # Create test args
        from argparse import Namespace

        args = Namespace(
            workflow_type="test_statistics",
            dir=str(data_dir),
            preview_prompts=False,
            validation_results_dir=str(validation_dir),
        )

        # Import main
        from qsp_llm_workflows.cli.fix import main

        # Patch sys.argv and run
        with patch(
            "sys.argv", ["qsp-fix", "test_statistics", "--immediate", "--dir", str(data_dir)]
        ):
            with patch("argparse.ArgumentParser.parse_args", return_value=args):
                with patch("sys.stdout", new_callable=StringIO):
                    main()

        # Verify orchestrator was called with test_statistic workflow type
        call_kwargs = mock_orchestrator.run_validation_fix_workflow.call_args.kwargs
        assert call_kwargs["workflow_type"] == "test_statistic"  # Converted from test_statistics
