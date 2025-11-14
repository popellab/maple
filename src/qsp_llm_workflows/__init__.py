"""
QSP LLM Workflows - Automated extraction of QSP metadata from scientific literature.

This package provides tools for:
- Parameter extraction from literature
- Test statistic extraction and validation
- Automated batch processing via OpenAI API
- Validation and quality control
"""

__version__ = "0.1.0"

# Public API
from qsp_llm_workflows.core.batch_creator import (
    BatchCreator,
    ParameterBatchCreator,
    TestStatisticBatchCreator,
    ValidationFixBatchCreator,
)
from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator
from qsp_llm_workflows.core.prompt_assembly import PromptAssembler

__all__ = [
    "__version__",
    "BatchCreator",
    "ParameterBatchCreator",
    "TestStatisticBatchCreator",
    "ValidationFixBatchCreator",
    "WorkflowOrchestrator",
    "PromptAssembler",
]
