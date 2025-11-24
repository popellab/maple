"""
Abstract base class for workflow steps.

Implements the Chain of Responsibility pattern for workflow execution.
"""
from abc import ABC, abstractmethod
from qsp_llm_workflows.core.workflow.context import WorkflowContext


class WorkflowStep(ABC):
    """
    Abstract base class for workflow steps.

    Each step receives a WorkflowContext, performs its work,
    and returns the updated context for the next step.
    """

    @abstractmethod
    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """
        Execute this workflow step.

        Args:
            context: Current workflow context

        Returns:
            Updated workflow context

        Raises:
            WorkflowException: If step execution fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this step."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
