"""
QSP LLM Workflows - Automated extraction of QSP metadata from scientific literature.

This package provides tools for:
- Parameter extraction from literature
- Test statistic extraction and validation
- Automated extraction processing via Pydantic AI
- Validation and quality control
"""

__version__ = "0.1.0"

# Public API
from qsp_llm_workflows.core.prompt_builder import (
    PromptBuilder,
    ParameterPromptBuilder,
    TestStatisticPromptBuilder,
)
from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator

__all__ = [
    "__version__",
    "PromptBuilder",
    "ParameterPromptBuilder",
    "TestStatisticPromptBuilder",
    "WorkflowOrchestrator",
]
