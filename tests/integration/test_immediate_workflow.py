"""
Integration tests for immediate mode workflow.

Tests the complete end-to-end workflow:
1. CSV input → ImmediateRequestProcessor
2. Mock API calls → Results JSONL
3. Unpack results → YAML files
4. Git operations → Review branch

All OpenAI API calls and git operations are mocked.
"""

from pathlib import Path
from unittest.mock import patch
import pytest

from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator


def test_immediate_mode_parameter_extraction(
    tmp_workspace, mock_openai_immediate, mock_git_operations, api_key
):
    """
    Test complete immediate mode workflow for parameter extraction.

    Verifies:
    - Results JSONL is created
    - YAML files are written to to-review/parameter_estimates/
    - Header fields are added correctly
    - Git branch is created and files committed
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"

    # Patch AsyncOpenAI to use our mock
    with patch(
        "qsp_llm_workflows.core.immediate_processor.AsyncOpenAI",
        return_value=mock_openai_immediate
    ):
        # Patch git operations
        with mock_git_operations:
            # Create orchestrator
            orchestrator = WorkflowOrchestrator(
                base_dir=workflows_dir,
                storage_dir=storage_dir,
                api_key=api_key
            )

            # Run complete workflow in immediate mode
            result = orchestrator.run_complete_workflow(
                input_csv=input_csv,
                workflow_type="parameter",
                immediate=True,
                push=False,  # Don't push in tests
                progress_callback=lambda msg: print(f"[TEST] {msg}")
            )

    # Verify results
    assert result["status"] == "success"
    assert result["immediate_mode"] is True

    # Check JSONL results file was created
    results_file = Path(result["results_file"])
    assert results_file.exists()
    assert "immediate_parameter" in results_file.name

    # Check YAML files were created in to-review
    output_dir = storage_dir / "to-review" / "parameter_estimates"
    yaml_files = list(output_dir.glob("*.yaml"))
    assert len(yaml_files) == 2  # Two rows in CSV

    # Verify file naming pattern: {param}_{cancer}_{hash}_deriv{num}.yaml
    filenames = [f.name for f in yaml_files]
    assert any("k_C1_death" in fn for fn in filenames)
    assert any("k_C1_growth" in fn for fn in filenames)
    assert all("PDAC" in fn for fn in filenames)
    assert all("deriv" in fn for fn in filenames)

    # Read one YAML and verify headers were added
    yaml_content = yaml_files[0].read_text()
    assert "schema_version:" in yaml_content
    assert "parameter_name:" in yaml_content
    assert "cancer_type: PDAC" in yaml_content
    assert "derivation_id:" in yaml_content
    assert "derivation_timestamp:" in yaml_content
    assert "context_hash:" in yaml_content
    assert "tags:" in yaml_content
    assert "ai-generated" in yaml_content

    # Verify mathematical_role (from Pydantic model) is present
    assert "mathematical_role:" in yaml_content


def test_immediate_mode_test_statistics(
    tmp_workspace, mock_openai_immediate, mock_git_operations, api_key
):
    """
    Test complete immediate mode workflow for test statistics.

    Verifies:
    - Test statistic YAML files created
    - Bootstrap code is present
    - Header fields added correctly
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "test_stat_input.csv"

    # Patch AsyncOpenAI to use our mock
    with patch(
        "qsp_llm_workflows.core.immediate_processor.AsyncOpenAI",
        return_value=mock_openai_immediate
    ):
        # Patch git operations
        with mock_git_operations:
            # Create orchestrator
            orchestrator = WorkflowOrchestrator(
                base_dir=workflows_dir,
                storage_dir=storage_dir,
                api_key=api_key
            )

            # Run complete workflow in immediate mode
            result = orchestrator.run_complete_workflow(
                input_csv=input_csv,
                workflow_type="test_statistic",
                immediate=True,
                push=False,
                progress_callback=lambda msg: print(f"[TEST] {msg}")
            )

    # Verify results
    assert result["status"] == "success"
    assert result["immediate_mode"] is True

    # Check YAML files were created
    output_dir = storage_dir / "to-review" / "test_statistics"
    yaml_files = list(output_dir.glob("*.yaml"))
    assert len(yaml_files) == 2  # Two rows in CSV

    # Verify file naming pattern
    filenames = [f.name for f in yaml_files]
    assert any("tumor_volume_day14" in fn for fn in filenames)
    assert any("cd8_treg_ratio_peak" in fn for fn in filenames)

    # Read one YAML and verify test statistic structure
    yaml_content = yaml_files[0].read_text()
    assert "schema_version:" in yaml_content
    assert "test_statistic_id:" in yaml_content
    assert "cancer_type: PDAC" in yaml_content
    assert "derivation_id:" in yaml_content
    assert "context_hash:" in yaml_content

    # Verify test statistic-specific fields
    assert "test_statistic_overview:" in yaml_content
    assert "analysis_method:" in yaml_content
    assert "bootstrap_code:" in yaml_content or "bootstrap_code: |" in yaml_content


def test_immediate_mode_handles_api_errors(
    tmp_workspace, mock_git_operations, api_key
):
    """
    Test that workflow handles API errors gracefully.

    Verifies error results are written to JSONL with error field populated.
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"

    # Create mock that raises an exception
    from unittest.mock import Mock, AsyncMock

    mock_client = Mock()
    mock_responses = Mock()

    async def mock_parse_error(**kwargs):
        raise Exception("API quota exceeded")

    mock_responses.parse = AsyncMock(side_effect=mock_parse_error)
    mock_client.responses = mock_responses

    # Patch AsyncOpenAI to use our error mock
    with patch(
        "qsp_llm_workflows.core.immediate_processor.AsyncOpenAI",
        return_value=mock_client
    ):
        # Patch git operations
        with mock_git_operations:
            # Create orchestrator
            orchestrator = WorkflowOrchestrator(
                base_dir=workflows_dir,
                storage_dir=storage_dir,
                api_key=api_key
            )

            # Run workflow - should not crash even with API errors
            result = orchestrator.run_complete_workflow(
                input_csv=input_csv,
                workflow_type="parameter",
                immediate=True,
                push=False,
                progress_callback=lambda msg: print(f"[TEST] {msg}")
            )

    # Verify workflow still completes (with errors recorded)
    assert result["status"] == "success"

    # Check results file exists and contains error information
    results_file = Path(result["results_file"])
    assert results_file.exists()

    # Read results and verify errors are recorded
    import json
    with open(results_file) as f:
        results = [json.loads(line) for line in f]

    assert len(results) == 2  # Two rows
    for res in results:
        assert res["error"] is not None
        assert "API quota exceeded" in res["error"]["message"]


def test_immediate_mode_api_parameters(
    tmp_workspace, mock_git_operations, api_key
):
    """
    Test that API parameters are passed correctly to OpenAI.

    Verifies the API call receives correct model, tools, and reasoning.
    """
    workflows_dir = tmp_workspace["workflows_dir"]
    storage_dir = tmp_workspace["storage_dir"]
    input_csv = workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"

    # Track API calls to verify parameters
    api_calls = []

    from unittest.mock import Mock, AsyncMock
    from qsp_llm_workflows.core.pydantic_models import (
        ParameterMetadata, ParameterEstimates, Input, KeyAssumption,
        BiologicalRelevance, WeightScore, Source
    )

    mock_client = Mock()
    mock_responses = Mock()

    async def mock_parse(**kwargs):
        api_calls.append(kwargs)
        mock_response = Mock()
        mock_response.id = "resp_test"
        mock_response.output_parsed = ParameterMetadata(
            mathematical_role="Test",
            parameter_range="positive_reals",
            study_overview="Test",
            study_design="Test design",
            parameter_estimates=ParameterEstimates(
                inputs=[
                    Input(
                        name="test", value=0.1, units="1/day", description="Test",
                        source_ref="source_1", value_table_or_section="Table 1",
                        value_snippet="Test", units_table_or_section="Methods",
                        units_snippet="Test"
                    )
                ],
                derivation_code="```python\ntest\n```",
                median=0.1, iqr=0.01, ci95=[0.09, 0.11], units="1/day"
            ),
            key_assumptions=[KeyAssumption(number=1, text="Test")],
            derivation_explanation="Test",
            key_study_limitations="Test",
            primary_data_sources=[
                Source(source_tag="source_1", title="Test", first_author="Smith", year=2020, doi="10.1234/test")
            ],
            secondary_data_sources=[],
            methodological_sources=[],
            biological_relevance=BiologicalRelevance(
                species_match=WeightScore(value=0.8, justification="Test"),
                system_match=WeightScore(value=0.8, justification="Test"),
                overall_confidence=WeightScore(value=0.8, justification="Test"),
                indication_match=WeightScore(value=0.8, justification="Test"),
                regimen_match=WeightScore(value=0.8, justification="Test"),
                biomarker_population_match=WeightScore(value=0.8, justification="Test"),
                stage_burden_match=WeightScore(value=0.8, justification="Test")
            )
        )
        return mock_response

    mock_responses.parse = AsyncMock(side_effect=mock_parse)
    mock_client.responses = mock_responses

    # Patch AsyncOpenAI
    with patch(
        "qsp_llm_workflows.core.immediate_processor.AsyncOpenAI",
        return_value=mock_client
    ):
        # Patch git operations
        with mock_git_operations:
            # Create orchestrator
            orchestrator = WorkflowOrchestrator(
                base_dir=workflows_dir,
                storage_dir=storage_dir,
                api_key=api_key
            )

            # Run workflow
            result = orchestrator.run_complete_workflow(
                input_csv=input_csv,
                workflow_type="parameter",
                immediate=True,
                push=False,
                progress_callback=lambda msg: print(f"[TEST] {msg}")
            )

    # Verify API parameters
    assert len(api_calls) == 2  # Two rows
    for call in api_calls:
        assert call["reasoning"] == {"effort": "high"}  # Default
        assert call["model"] == "gpt-5"
        assert call["tools"] == [{"type": "web_search"}]
        assert isinstance(call["input"], str)  # Prompt text
