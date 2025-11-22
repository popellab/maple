"""
Integration tests for batch mode workflow.

Tests the complete end-to-end batch workflow:
1. CSV input → Batch JSONL creation
2. Mock batch upload → Mock monitoring
3. Mock results download → Unpack
4. Git operations → Review branch

All OpenAI API calls and git operations are mocked.
"""

from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator


def test_batch_mode_parameter_extraction(
    tmp_workspace, mock_openai_batch, mock_git_operations, api_key
):
    """
    Test complete batch mode workflow for parameter extraction.

    Verifies:
    - Batch JSONL is created
    - Batch upload and monitoring work
    - Results are unpacked to YAML
    - Git operations complete successfully
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"

    # Patch OpenAI client
    with patch(
        "qsp_llm_workflows.core.workflow_orchestrator.OpenAI",
        return_value=mock_openai_batch
    ):
        # Patch git operations
        with mock_git_operations:
            # Mock user input for prompt verification (auto-confirm)
            with patch("builtins.input", return_value="y"):
                # Create orchestrator
                orchestrator = WorkflowOrchestrator(
                    base_dir=workflows_dir,
                    storage_dir=storage_dir,
                    api_key=api_key
                )

                # Run complete workflow in batch mode
                result = orchestrator.run_complete_workflow(
                    input_csv=input_csv,
                    workflow_type="parameter",
                    immediate=False,  # Batch mode
                    push=False,
                    progress_callback=lambda msg: print(f"[TEST] {msg}")
                )

    # Verify results
    assert result["status"] == "success"
    assert result.get("immediate_mode", False) is False

    # Check batch file was created
    batch_file = Path(result["batch_file"])
    assert batch_file.exists()
    assert batch_file.suffix == ".jsonl"

    # Check batch ID was recorded
    assert "batch_id" in result
    assert result["batch_id"].startswith("batch_test_")

    # Check results file was downloaded
    results_file = Path(result["results_file"])
    assert results_file.exists()

    # Check YAML files were created
    output_dir = storage_dir / "to-review" / "parameter_estimates"
    yaml_files = list(output_dir.glob("*.yaml"))
    assert len(yaml_files) == 2  # Two rows in CSV

    # Verify file naming and content
    filenames = [f.name for f in yaml_files]
    assert any("k_C1_death" in fn for fn in filenames)
    assert any("k_C1_growth" in fn for fn in filenames)

    # Read one YAML and verify structure
    yaml_content = yaml_files[0].read_text()
    assert "schema_version:" in yaml_content
    assert "parameter_name:" in yaml_content
    assert "cancer_type: PDAC" in yaml_content
    assert "mathematical_role:" in yaml_content


def test_batch_mode_test_statistics(
    tmp_workspace, mock_git_operations, api_key
):
    """
    Test complete batch mode workflow for test statistics.

    Verifies test statistic batch processing works correctly.
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "test_stat_input.csv"

    # Create custom mock for test statistics
    mock_client = Mock()

    # Mock file upload
    mock_file = Mock()
    mock_file.id = "file_test_stats"
    mock_client.files.create = Mock(return_value=mock_file)

    # Mock batch
    mock_batch = Mock()
    mock_batch.id = "batch_test_stats"
    mock_batch.status = "completed"
    mock_batch.request_counts = Mock(total=2, completed=2, failed=0)
    mock_batch.output_file_id = "file_test_stats_output"
    mock_client.batches.create = Mock(return_value=mock_batch)
    mock_client.batches.retrieve = Mock(return_value=mock_batch)

    # Mock results for test statistics
    import json
    sample_results = [
        {
            "custom_id": "PDAC_tumor_volume_day14_0",
            "response": {
                "status_code": 200,
                "request_id": "req_001",
                "body": {
                    "test_statistic_overview": "Tumor volume at day 14",
                    "analysis_method": "bootstrap",
                    "bootstrap_code": "# R bootstrap code\nmean(c(150, 165, 142))",
                    "primary_data_sources": [{
                        "title": "Test study",
                        "first_author": "Smith",
                        "year": 2020,
                        "doi": "10.1234/test",
                        "relevant_sections": "Methods"
                    }]
                }
            },
            "error": None
        },
        {
            "custom_id": "PDAC_cd8_treg_ratio_peak_1",
            "response": {
                "status_code": 200,
                "request_id": "req_002",
                "body": {
                    "test_statistic_overview": "CD8/Treg ratio at peak",
                    "analysis_method": "bootstrap",
                    "bootstrap_code": "# R bootstrap code\nratio <- cd8 / treg",
                    "primary_data_sources": [{
                        "title": "Immune study",
                        "first_author": "Jones",
                        "year": 2021,
                        "doi": "10.5678/test",
                        "relevant_sections": "Results"
                    }]
                }
            },
            "error": None
        }
    ]

    results_jsonl = "\n".join([json.dumps(r) for r in sample_results]).encode()
    mock_content = Mock()
    mock_content.content = results_jsonl
    mock_client.files.content = Mock(return_value=mock_content)

    # Patch OpenAI client
    with patch(
        "qsp_llm_workflows.core.workflow_orchestrator.OpenAI",
        return_value=mock_client
    ):
        # Patch git operations
        with mock_git_operations:
            # Mock prompt verification
            with patch("builtins.input", return_value="y"):
                # Create orchestrator
                orchestrator = WorkflowOrchestrator(
                    base_dir=workflows_dir,
                    storage_dir=storage_dir,
                    api_key=api_key
                )

                # Run batch workflow
                result = orchestrator.run_complete_workflow(
                    input_csv=input_csv,
                    workflow_type="test_statistic",
                    immediate=False,
                    push=False,
                    progress_callback=lambda msg: print(f"[TEST] {msg}")
                )

    # Verify results
    assert result["status"] == "success"

    # Check YAML files
    output_dir = storage_dir / "to-review" / "test_statistics"
    yaml_files = list(output_dir.glob("*.yaml"))
    assert len(yaml_files) == 2

    # Verify test statistic content
    yaml_content = yaml_files[0].read_text()
    assert "test_statistic_id:" in yaml_content
    assert "bootstrap_code:" in yaml_content or "bootstrap_code: |" in yaml_content


def test_batch_mode_monitoring(tmp_workspace, mock_git_operations, api_key):
    """
    Test batch monitoring with different status progressions.

    Verifies the workflow correctly handles:
    - Status progression: validating → in_progress → completed
    - Request count updates
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"

    # Create mock with status progression
    mock_client = Mock()

    # Mock file upload
    mock_file = Mock()
    mock_file.id = "file_test"
    mock_client.files.create = Mock(return_value=mock_file)

    # Mock batch with status progression
    call_count = {"count": 0}

    def mock_retrieve(batch_id):
        """Simulate status progression over multiple calls."""
        statuses = ["validating", "in_progress", "in_progress", "completed"]
        counts = [
            {"total": 2, "completed": 0, "failed": 0},
            {"total": 2, "completed": 1, "failed": 0},
            {"total": 2, "completed": 2, "failed": 0},
            {"total": 2, "completed": 2, "failed": 0},
        ]

        idx = min(call_count["count"], len(statuses) - 1)
        call_count["count"] += 1

        mock_batch = Mock()
        mock_batch.id = batch_id
        mock_batch.status = statuses[idx]
        mock_batch.request_counts = Mock(**counts[idx])
        mock_batch.output_file_id = "file_output" if statuses[idx] == "completed" else None
        return mock_batch

    mock_client.batches.create = Mock(return_value=mock_retrieve("batch_test"))
    mock_client.batches.retrieve = Mock(side_effect=mock_retrieve)

    # Mock results download
    import json
    results = [
        {
            "custom_id": "test_0",
            "response": {
                "status_code": 200,
                "body": {
                    "mathematical_role": "Test",
                    "parameter_range": "positive_reals",
                    "study_overview": "Test",
                    "parameter_estimates": [{
                        "value_type": "point_estimate",
                        "value": 0.1,
                        "units": "1/day",
                        "source_ref": "source_1",
                        "value_snippet": "Test",
                        "derivation_method": "direct_measurement"
                    }],
                    "primary_data_sources": [{
                        "title": "Test",
                        "first_author": "Smith",
                        "year": 2020,
                        "doi": "10.1234/test",
                        "relevant_sections": "Methods"
                    }]
                }
            },
            "error": None
        }
    ] * 2

    results_jsonl = "\n".join([json.dumps(r) for r in results]).encode()
    mock_content = Mock()
    mock_content.content = results_jsonl
    mock_client.files.content = Mock(return_value=mock_content)

    # Patch OpenAI and git
    with patch(
        "qsp_llm_workflows.core.workflow_orchestrator.OpenAI",
        return_value=mock_client
    ):
        with mock_git_operations:
            with patch("builtins.input", return_value="y"):
                # Patch time.sleep to speed up test
                with patch("time.sleep"):
                    orchestrator = WorkflowOrchestrator(
                        base_dir=workflows_dir,
                        storage_dir=storage_dir,
                        api_key=api_key
                    )

                    result = orchestrator.run_complete_workflow(
                        input_csv=input_csv,
                        workflow_type="parameter",
                        immediate=False,
                        push=False,
                        progress_callback=lambda msg: print(f"[TEST] {msg}")
                    )

    # Verify monitoring worked
    assert result["status"] == "success"
    assert call_count["count"] >= 3  # At least a few monitoring calls


def test_batch_file_structure(tmp_workspace, api_key):
    """
    Test that batch JSONL file has correct structure.

    Verifies:
    - Each line is valid JSON
    - Required fields present (custom_id, method, url, body)
    - Body contains model, input, reasoning, text format
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"

    # Don't need API calls for batch file creation
    orchestrator = WorkflowOrchestrator(
        base_dir=workflows_dir,
        storage_dir=storage_dir,
        api_key=api_key
    )

    # Create batch file only (don't run full workflow)
    batch_file = orchestrator.create_batch(
        input_csv=input_csv,
        workflow_type="parameter",
        progress_callback=lambda msg: print(f"[TEST] {msg}")
    )

    # Verify file exists
    assert batch_file.exists()

    # Read and validate structure
    import json
    with open(batch_file) as f:
        lines = f.readlines()

    assert len(lines) == 2  # Two rows in CSV

    for line in lines:
        request = json.loads(line)

        # Verify required fields
        assert "custom_id" in request
        assert "method" in request
        assert request["method"] == "POST"
        assert "url" in request
        assert request["url"] == "/v1/responses"
        assert "body" in request

        # Verify body structure
        body = request["body"]
        assert "model" in body
        assert body["model"] == "gpt-5"
        assert "input" in body
        assert isinstance(body["input"], str)  # Prompt text
        assert "reasoning" in body
        assert body["reasoning"]["effort"] == "high"  # Default
        assert "text" in body
        assert "format" in body["text"]
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
