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
    CreateValidationFixBatchStep,
    ProcessImmediateValidationFixStep,
    UnpackValidationFixResultsStep,
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
        self.immediate_mode = context.immediate
        self.batch_id = context.batch_id
        self.batch_file = str(context.batch_file) if context.batch_file else None
        self.results_file = str(context.results_file) if context.results_file else None

        # In preview mode, output_directory is the batch file path
        # In normal mode, output_directory is the unpacked results directory
        if context.get_metadata("preview_prompts", False):
            self.output_directory = str(context.batch_file) if context.batch_file else None
            # Count requests in batch file
            if context.batch_file and context.batch_file.exists():
                with open(context.batch_file) as f:
                    self.file_count = sum(1 for _ in f)
            else:
                self.file_count = 0
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
        preview_prompts: bool = False,
        use_pydantic_ai: bool = False,
    ) -> WorkflowResult:
        """
        Run complete extraction workflow from start to finish.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic/calibration_target)
            immediate: Use Responses API for immediate processing (faster, good for testing)
            timeout: Maximum seconds to wait for batch completion (uses config default if None)
            reasoning_effort: Reasoning effort level (low/medium/high, default: high)
            progress_callback: Optional callback for progress updates
            preview_prompts: If True, only build and save prompts without sending to API
            use_pydantic_ai: Use Pydantic AI instead of direct OpenAI API (immediate mode only)

        Returns:
            WorkflowResult with execution metadata

        Raises:
            Exception: If any workflow step fails
        """
        start_time = time.time()

        # Override config with runtime reasoning_effort if provided
        config = self.config
        if reasoning_effort != config.reasoning_effort:
            # Create new config with updated reasoning_effort
            config = WorkflowConfig(
                base_dir=config.base_dir,
                storage_dir=config.storage_dir,
                openai_api_key=config.openai_api_key,
                openai_model=config.openai_model,
                reasoning_effort=reasoning_effort,
                batch_completion_window=config.batch_completion_window,
                batch_timeout=config.batch_timeout,
                poll_interval=config.poll_interval,
            )

        # Create workflow context
        context = WorkflowContext(
            input_csv=input_csv,
            workflow_type=workflow_type,
            immediate=immediate,
            config=config,
            progress_callback=progress_callback,
        )

        # Store start time in metadata
        context.set_metadata("started_at", datetime.now().isoformat())

        # Store preview mode in metadata
        context.set_metadata("preview_prompts", preview_prompts)

        # Store Pydantic AI mode in metadata
        context.set_metadata("use_pydantic_ai", use_pydantic_ai)

        try:
            # Get workflow steps for this mode
            if preview_prompts:
                # Preview mode: only create batch file
                steps = [CreateBatchStep()]
            else:
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

    def _get_validation_fix_steps(self, immediate: bool) -> List[WorkflowStep]:
        """
        Get workflow steps for validation fix mode.

        Args:
            immediate: True for immediate mode, False for batch mode

        Returns:
            List of workflow steps to execute
        """
        if immediate:
            # Immediate mode: process directly and unpack to original dir
            return [
                ProcessImmediateValidationFixStep(),
                UnpackValidationFixResultsStep(),
            ]
        else:
            # Batch mode: create, upload, monitor, unpack to original dir
            return [
                CreateValidationFixBatchStep(),
                UploadBatchStep(),
                MonitorBatchStep(),
                UnpackValidationFixResultsStep(),
            ]

    def run_validation_fix_workflow(
        self,
        data_dir: Path,
        validation_results_dir: Path,
        workflow_type: str,
        immediate: bool = False,
        timeout: Optional[int] = None,
        reasoning_effort: str = "high",
        progress_callback: Optional[Callable[[str], None]] = None,
        preview_prompts: bool = False,
    ) -> WorkflowResult:
        """
        Run validation fix workflow to correct validation errors.

        Args:
            data_dir: Directory containing YAML files with validation errors
            validation_results_dir: Directory containing validation JSON reports
            workflow_type: Type of workflow (parameter/test_statistic)
            immediate: Use Responses API for immediate processing (faster, good for testing)
            timeout: Maximum seconds to wait for batch completion (uses config default if None)
            reasoning_effort: Reasoning effort level (low/medium/high, default: high)
            progress_callback: Optional callback for progress updates
            preview_prompts: If True, only build and save prompts without sending to API

        Returns:
            WorkflowResult with execution metadata

        Raises:
            Exception: If any workflow step fails
        """
        start_time = time.time()

        # Create workflow context (no input CSV needed for validation fix)
        context = WorkflowContext(
            input_csv=None,  # No CSV input for validation fix
            workflow_type=workflow_type,
            immediate=immediate,
            config=self.config,
            progress_callback=progress_callback,
        )

        # Store validation fix metadata
        context.set_metadata("data_dir", data_dir)
        context.set_metadata("validation_results_dir", validation_results_dir)
        context.set_metadata("reasoning_effort", reasoning_effort)
        context.set_metadata("started_at", datetime.now().isoformat())
        context.set_metadata("preview_prompts", preview_prompts)

        try:
            # Get workflow steps for validation fix mode
            if preview_prompts:
                # Preview mode: only create batch file
                steps = [CreateValidationFixBatchStep()]
            else:
                steps = self._get_validation_fix_steps(immediate)

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
