"""
Unit tests for WorkflowStep base class.

Tests the abstract base class for workflow steps.
"""

import pytest
from unittest.mock import Mock

from qsp_llm_workflows.core.workflow.step import WorkflowStep
from qsp_llm_workflows.core.workflow.context import WorkflowContext


class ConcreteStep(WorkflowStep):
    """Concrete implementation for testing."""

    def __init__(self, should_modify=True):
        self.should_modify = should_modify
        self.executed = False

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        self.executed = True

        if self.should_modify:
            context.set_metadata("executed_by", self.name)

        return context

    @property
    def name(self) -> str:
        return "ConcreteStep"


class TestWorkflowStep:
    """Test WorkflowStep base class."""

    def test_cannot_instantiate_abstract_step(self):
        """Test that WorkflowStep cannot be instantiated directly."""
        with pytest.raises(TypeError):
            WorkflowStep()

    def test_concrete_step_can_be_instantiated(self):
        """Test that concrete step can be created."""
        step = ConcreteStep()
        assert step is not None
        assert isinstance(step, WorkflowStep)

    def test_step_execute(self, tmp_path):
        """Test executing a workflow step."""
        step = ConcreteStep()
        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=Mock(),
        )

        result = step.execute(context)

        assert step.executed is True
        assert result is context  # Same object
        assert result.get_metadata("executed_by") == "ConcreteStep"

    def test_step_name(self):
        """Test step name property."""
        step = ConcreteStep()
        assert step.name == "ConcreteStep"

    def test_step_repr(self):
        """Test step string representation."""
        step = ConcreteStep()
        assert repr(step) == "ConcreteStep()"

    def test_step_modifies_context(self, tmp_path):
        """Test that step can modify context."""
        step = ConcreteStep(should_modify=True)
        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=Mock(),
        )

        # Context starts empty
        assert context.get_metadata("executed_by") is None

        # Execute step
        result = step.execute(context)

        # Context is modified
        assert result.get_metadata("executed_by") == "ConcreteStep"

    def test_step_chain(self, tmp_path):
        """Test chaining multiple steps."""
        step1 = ConcreteStep()
        step2 = ConcreteStep()
        step3 = ConcreteStep()

        context = WorkflowContext(
            input_csv=tmp_path / "input.csv",
            workflow_type="parameter",
            immediate=True,
            config=Mock(),
        )

        # Chain steps
        result = step1.execute(context)
        result = step2.execute(result)
        result = step3.execute(result)

        # All steps executed
        assert step1.executed is True
        assert step2.executed is True
        assert step3.executed is True

        # Context was passed through
        assert result is context
