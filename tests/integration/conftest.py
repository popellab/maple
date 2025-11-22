"""
Shared fixtures for integration tests.

Provides mocks for OpenAI API calls and git operations,
plus temporary workspace setup for isolated testing.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import pytest

from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic


@pytest.fixture
def tmp_workspace(tmp_path):
    """
    Create temporary workspace with qsp-llm-workflows and qsp-metadata-storage directories.

    Returns:
        dict with 'workflows_dir' and 'storage_dir' paths
    """
    workflows_dir = tmp_path / "qsp-llm-workflows"
    storage_dir = tmp_path / "qsp-metadata-storage"

    # Create directory structure
    workflows_dir.mkdir()
    (workflows_dir / "batch_jobs").mkdir()
    (workflows_dir / "batch_jobs" / "input_data").mkdir()

    storage_dir.mkdir()
    (storage_dir / "to-review").mkdir()
    (storage_dir / "to-review" / "parameter_estimates").mkdir()
    (storage_dir / "to-review" / "test_statistics").mkdir()
    (storage_dir / ".git").mkdir()  # Mock git repo

    # Copy fixture CSVs to input_data
    fixtures_dir = Path(__file__).parent / "fixtures"
    shutil.copy(
        fixtures_dir / "parameter_input.csv",
        workflows_dir / "batch_jobs" / "input_data" / "parameter_input.csv"
    )
    shutil.copy(
        fixtures_dir / "test_stat_input.csv",
        workflows_dir / "batch_jobs" / "input_data" / "test_stat_input.csv"
    )

    return {
        "workflows_dir": workflows_dir,
        "storage_dir": storage_dir,
    }


@pytest.fixture
def sample_parameter_metadata():
    """Sample ParameterMetadata for mock API responses."""
    from qsp_llm_workflows.core.pydantic_models import (
        ParameterEstimates, Input, KeyAssumption, BiologicalRelevance,
        WeightScore, Source, SecondarySource, MethodologicalSource
    )

    return ParameterMetadata(
        mathematical_role="Represents the death rate of cancer subpopulation 1 cells in the PDAC model.",
        parameter_range="positive_reals",
        study_overview="Parameter estimated from in vitro cell death assays of PDAC cell lines.",
        study_design="Cell death rates measured in triplicate using flow cytometry over 72 hours.",
        parameter_estimates=ParameterEstimates(
            inputs=[
                Input(
                    name="cell_death_rate",
                    value=0.05,
                    units="1/day",
                    description="Measured cell death rate",
                    source_ref="source_1",
                    value_table_or_section="Table 1",
                    value_snippet="The estimated death rate was 0.05 per day",
                    units_table_or_section="Methods",
                    units_snippet="Rates reported in per day units"
                )
            ],
            derivation_code="```python\nimport numpy as np\nmedian = 0.05\niqr = 0.01\nci95 = [0.04, 0.06]\n```",
            median=0.05,
            iqr=0.01,
            ci95=[0.04, 0.06],
            units="1/day"
        ),
        key_assumptions=[
            KeyAssumption(number=1, text="Cell death rate is constant over time"),
            KeyAssumption(number=2, text="In vitro rates translate to in vivo conditions")
        ],
        derivation_explanation="The median death rate was calculated from experimental measurements.",
        key_study_limitations="Limited to single cell line, may not generalize to patient tumors.",
        primary_data_sources=[
            Source(
                source_tag="source_1",
                title="Quantification of PDAC cell death kinetics in vitro and in vivo",
                first_author="Smith",
                year=2020,
                doi="10.1234/test.2020.001"
            )
        ],
        secondary_data_sources=[],
        methodological_sources=[],
        biological_relevance=BiologicalRelevance(
            species_match=WeightScore(value=0.8, justification="Mouse model"),
            system_match=WeightScore(value=0.9, justification="PDAC specific"),
            overall_confidence=WeightScore(value=0.85, justification="High quality data"),
            indication_match=WeightScore(value=1.0, justification="PDAC"),
            regimen_match=WeightScore(value=0.7, justification="No treatment"),
            biomarker_population_match=WeightScore(value=0.8, justification="General PDAC"),
            stage_burden_match=WeightScore(value=0.75, justification="Early stage")
        )
    )


@pytest.fixture
def sample_test_statistic():
    """Sample TestStatistic for mock API responses."""
    from qsp_llm_workflows.core.pydantic_models import (
        ModelOutput, TestStatisticEstimates, Input, KeyAssumption,
        ValidationWeights, WeightScore, Source, SecondarySource, MethodologicalSource
    )

    return TestStatistic(
        model_output=ModelOutput(
            code="```python\ntumor_volume = simulation['V_T.C'][14]\n```"
        ),
        test_statistic_definition="Tumor volume (mm³) at day 14 post-inoculation",
        study_overview="Measures baseline tumor growth in untreated PDAC xenograft models.",
        study_design="NOD/SCID mice with subcutaneous PDAC tumors measured at day 14 (n=6).",
        test_statistic_estimates=TestStatisticEstimates(
            inputs=[
                Input(
                    name="tumor_volume_day14",
                    value=155.0,
                    units="mm³",
                    description="Tumor volume at day 14",
                    source_ref="source_1",
                    value_table_or_section="Figure 1",
                    value_snippet="Tumor volumes at day 14: 150, 165, 142, 158, 171, 148 mm³",
                    units_table_or_section="Methods",
                    units_snippet="Volumes measured in mm³"
                )
            ],
            derivation_code="```python\nimport numpy as np\nvalues = [150, 165, 142, 158, 171, 148]\nmedian = np.median(values)\n```",
            median=155.0,
            iqr=20.0,
            ci95=[145.0, 168.0],
            units="mm³",
            key_assumptions=[
                KeyAssumption(number=1, text="Tumor growth is consistent across animals"),
                KeyAssumption(number=2, text="Day 14 is representative of baseline growth")
            ]
        ),
        derivation_explanation="Bootstrap analysis of tumor volumes from experimental data.",
        key_study_limitations="Limited sample size (n=6), single cell line tested.",
        primary_data_sources=[
            Source(
                source_tag="source_1",
                title="Baseline PDAC tumor growth kinetics in NOD/SCID mice",
                first_author="Johnson",
                year=2021,
                doi="10.5678/test.2021.002"
            )
        ],
        secondary_data_sources=[],
        methodological_sources=[],
        validation_weights=ValidationWeights(
            species_match=WeightScore(value=0.8, justification="Mouse model"),
            system_match=WeightScore(value=0.9, justification="PDAC specific"),
            overall_confidence=WeightScore(value=0.85, justification="Good quality"),
            indication_match=WeightScore(value=1.0, justification="PDAC"),
            regimen_match=WeightScore(value=0.7, justification="Baseline"),
            biomarker_population_match=WeightScore(value=0.8, justification="General PDAC"),
            stage_burden_match=WeightScore(value=0.75, justification="Early stage")
        )
    )


@pytest.fixture
def mock_openai_immediate(sample_parameter_metadata, sample_test_statistic):
    """
    Mock AsyncOpenAI client for immediate mode.

    Returns mock client with responses.parse() configured.
    """
    mock_client = Mock()
    mock_responses = Mock()

    # Track which model type to return based on call order
    # (Alternates between parameter and test stat for multi-type tests)
    call_count = {"count": 0}

    async def mock_parse(**kwargs):
        # Create mock response
        mock_response = Mock()
        mock_response.id = f"resp_test_{call_count['count']:03d}"

        # Use text_format to determine which model to return
        model_class = kwargs.get("text_format")
        if model_class.__name__ == "ParameterMetadata":
            mock_response.output_parsed = sample_parameter_metadata
        elif model_class.__name__ == "TestStatistic":
            mock_response.output_parsed = sample_test_statistic
        else:
            # Default to parameter metadata
            mock_response.output_parsed = sample_parameter_metadata

        call_count["count"] += 1
        return mock_response

    mock_responses.parse = AsyncMock(side_effect=mock_parse)
    mock_client.responses = mock_responses

    return mock_client


@pytest.fixture
def mock_openai_batch(sample_parameter_metadata):
    """
    Mock OpenAI client for batch mode.

    Returns mock client with files and batches configured.
    """
    mock_client = Mock()

    # Mock file upload
    mock_file = Mock()
    mock_file.id = "file_test_abc123"
    mock_client.files.create = Mock(return_value=mock_file)

    # Mock batch creation
    mock_batch = Mock()
    mock_batch.id = "batch_test_xyz789"
    mock_batch.status = "completed"
    mock_batch.request_counts = Mock(total=2, completed=2, failed=0)
    mock_batch.output_file_id = "file_test_output_def456"
    mock_client.batches.create = Mock(return_value=mock_batch)

    # Mock batch retrieval (monitoring)
    mock_client.batches.retrieve = Mock(return_value=mock_batch)

    # Mock results download - must match batch API format with output/message/content structure
    import json as json_module

    def wrap_in_batch_format(data_dict):
        """Wrap JSON data in batch API response format."""
        json_str = json_module.dumps(data_dict, indent=2)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"text": f"```json\n{json_str}\n```"}
                    ]
                }
            ]
        }

    # Use the actual fixture value
    param_data = sample_parameter_metadata.model_dump()

    sample_results = [
        {
            "custom_id": "PDAC_k_C1_death_0",
            "response": {
                "status_code": 200,
                "request_id": "req_test_001",
                "body": wrap_in_batch_format(param_data)
            },
            "error": None,
        },
        {
            "custom_id": "PDAC_k_C1_growth_1",
            "response": {
                "status_code": 200,
                "request_id": "req_test_002",
                "body": wrap_in_batch_format(param_data)
            },
            "error": None,
        }
    ]

    # Convert to JSONL bytes
    results_jsonl = "\n".join([json.dumps(r) for r in sample_results]).encode()
    mock_content = Mock()
    mock_content.content = results_jsonl
    mock_client.files.content = Mock(return_value=mock_content)

    return mock_client


@pytest.fixture
def mock_git_operations():
    """
    Mock all git operations via subprocess.

    Returns a context manager that patches subprocess.run.
    """
    def _mock_run(cmd, *args, **kwargs):
        """Mock git command responses."""
        # Convert cmd list to string for easier matching
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd

        # Mock different git commands
        if "git status --porcelain" in cmd_str:
            return Mock(returncode=0, stdout="", stderr="")
        elif "git rev-parse --abbrev-ref HEAD" in cmd_str:
            return Mock(returncode=0, stdout="main\n", stderr="")
        elif "git checkout -b" in cmd_str:
            return Mock(returncode=0, stdout="", stderr="")
        elif "git add" in cmd_str:
            return Mock(returncode=0, stdout="", stderr="")
        elif "git commit" in cmd_str:
            return Mock(returncode=0, stdout="[test-branch abc123] Test commit\n", stderr="")
        elif "git push" in cmd_str:
            return Mock(returncode=0, stdout="", stderr="")
        else:
            # Default: success with no output
            return Mock(returncode=0, stdout="", stderr="")

    return patch("subprocess.run", side_effect=_mock_run)


@pytest.fixture
def api_key():
    """Test API key."""
    return "sk-test-abc123"
