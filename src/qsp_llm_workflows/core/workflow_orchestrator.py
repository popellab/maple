#!/usr/bin/env python3
"""
Refactored workflow orchestrator using Chain of Responsibility pattern.

Uses workflow steps for clean, testable architecture with clear separation
of concerns. Each step performs a specific part of the workflow and passes
state through a context object.
"""

import time
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime

from qsp_llm_workflows.core.config import WorkflowConfig
from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.workflow.step import WorkflowStep
from qsp_llm_workflows.core.workflow.steps import (
    CreateBatchStep,
    UploadBatchStep,
    MonitorBatchStep,
    ProcessImmediateStep,
    UnpackResultsStep,
)


class WorkflowResult:
    """Result of workflow execution with metadata."""

    def __init__(self, context: WorkflowContext, duration: float):
        """
        Create workflow result from context.

        Args:
            context: Final workflow context after all steps
            duration: Workflow execution duration in seconds
        """
        self.workflow_type = context.workflow_type
        self.input_csv = str(context.input_csv)
        self.immediate_mode = context.immediate
        self.batch_id = context.batch_id
        self.batch_file = str(context.batch_file) if context.batch_file else None
        self.results_file = str(context.results_file) if context.results_file else None
        self.output_directory = str(context.output_directory) if context.output_directory else None
        self.file_count = context.file_count
        self.status = "success"
        self.error = None
        self.started_at = context.get_metadata("started_at")
        self.completed_at = datetime.now().isoformat()
        self.duration_seconds = duration

    @classmethod
    def from_error(cls, context: WorkflowContext, error: Exception, duration: float):
        """
        Create failure result from error.

        Args:
            context: Workflow context at time of failure
            error: Exception that caused failure
            duration: Duration before failure

        Returns:
            WorkflowResult with error status
        """
        result = cls(context, duration)
        result.status = "failed"
        result.error = str(error)
        return result


class WorkflowOrchestrator:
    """
    Orchestrates extraction workflow using workflow steps.

    Uses Chain of Responsibility pattern to execute workflow steps in sequence,
    passing state through a WorkflowContext object.
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initialize workflow orchestrator.

        Args:
            config: WorkflowConfig instance with all settings
        """
        self.config = config

        # Ensure directories exist
        self.config.batch_jobs_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_review_dir.mkdir(parents=True, exist_ok=True)

    def _get_workflow_steps(self, immediate: bool) -> List[WorkflowStep]:
        """
        Get workflow steps for execution mode.

        Args:
            immediate: True for immediate mode, False for batch mode

        Returns:
            List of workflow steps to execute
        """
        if immediate:
            # Immediate mode: process directly and unpack
            return [
                ProcessImmediateStep(),
                UnpackResultsStep(),
            ]
        else:
            # Batch mode: create, upload, monitor, unpack
            return [
                CreateBatchStep(),
                UploadBatchStep(),
                MonitorBatchStep(),
                UnpackResultsStep(),
            ]

    def run_complete_workflow(
        self,
        input_csv: Path,
        workflow_type: str,
        immediate: bool = False,
        timeout: Optional[int] = None,
        reasoning_effort: str = "high",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> WorkflowResult:
        """
        Run complete extraction workflow from start to finish.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            immediate: Use Responses API for immediate processing (faster, good for testing)
            timeout: Maximum seconds to wait for batch completion (uses config default if None)
            reasoning_effort: Reasoning effort level (low/medium/high, default: high)
            progress_callback: Optional callback for progress updates

        Returns:
            WorkflowResult with execution metadata

        Raises:
            Exception: If any workflow step fails
        """
        start_time = time.time()

        # Create workflow context
        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type=workflow_type,
            immediate=immediate,
            config=self.config,
            progress_callback=progress_callback,
        )

        # Store reasoning effort in metadata
        context.set_metadata("reasoning_effort", reasoning_effort)

        # Store start time in metadata
        context.set_metadata("started_at", datetime.now().isoformat())

        try:
            # Get workflow steps for this mode
            steps = self._get_workflow_steps(immediate)

            # Execute steps in sequence
            for step in steps:
                context = step.execute(context)

            # Success!
            duration = time.time() - start_time
            return WorkflowResult(context, duration)

        except Exception as e:
            # Failure
            duration = time.time() - start_time
            return WorkflowResult.from_error(context, e, duration)
