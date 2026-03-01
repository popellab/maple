#!/usr/bin/env python3
"""
Refactored workflow orchestrator using Chain of Responsibility pattern.

Uses workflow steps for clean, testable architecture with clear separation
of concerns. Each step performs a specific part of the workflow and passes
state through a context object.
"""

import time
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from maple.core.config import WorkflowConfig
from maple.core.workflow.context import WorkflowContext
from maple.core.workflow.steps import (
    CreatePreviewStep,
    ProcessPromptsStep,
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
        self.input_csv = str(context.input_csv) if context.input_csv else None
        self.preview_file = str(context.preview_file) if context.preview_file else None
        self.results_file = str(context.results_file) if context.results_file else None

        # In preview mode, output_directory is the preview file path
        # In normal mode, output_directory is the unpacked results directory
        if context.get_metadata("preview_prompts", False):
            self.output_directory = str(context.preview_file) if context.preview_file else None
            self.file_count = context.file_count
        else:
            self.output_directory = (
                str(context.output_directory) if context.output_directory else None
            )
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
        self.config.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_review_dir.mkdir(parents=True, exist_ok=True)

    def run_complete_workflow(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        preview_prompts: bool = False,
    ) -> WorkflowResult:
        """
        Run complete extraction workflow from start to finish.

        Uses Pydantic AI for direct processing.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic/calibration_target)
            progress_callback: Optional callback for progress updates
            preview_prompts: If True, only build and save prompts without sending to API

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
            config=self.config,
            progress_callback=progress_callback,
        )

        # Store start time in metadata
        context.set_metadata("started_at", datetime.now().isoformat())

        # Store preview mode in metadata
        context.set_metadata("preview_prompts", preview_prompts)

        try:
            # Select workflow steps
            if preview_prompts:
                # Preview mode: only create preview file
                steps = [CreatePreviewStep()]
            else:
                # Normal mode: process with Pydantic AI (with streaming unpacking)
                # UnpackResultsStep no longer needed - unpacking happens during processing
                steps = [ProcessPromptsStep()]

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
