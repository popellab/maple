#!/usr/bin/env python3
"""
Refactored workflow orchestrator for automated extraction pipeline.

Simplified workflow that focuses on extraction and unpacking, delegating
version control to the user.

Workflow steps:
1. Create batch requests (or process immediately)
2. Upload and monitor batch (batch mode only)
3. Unpack results to unique timestamped directory
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from openai import OpenAI

from qsp_llm_workflows.core.config import WorkflowConfig
from qsp_llm_workflows.core.batch_creator import (
    ParameterBatchCreator,
    TestStatisticBatchCreator,
)
from qsp_llm_workflows.core.immediate_processor import ImmediateRequestProcessor
from qsp_llm_workflows.core.output_directory import create_unique_output_directory
from qsp_llm_workflows.process.unpack_results import process_results


class WorkflowResult:
    """Result of workflow execution with metadata."""

    def __init__(self, workflow_type: str, input_csv: Path):
        self.workflow_type = workflow_type
        self.input_csv = str(input_csv)
        self.started_at = datetime.now().isoformat()
        self.status = "running"
        self.immediate_mode = False
        self.batch_id = None
        self.batch_file = None
        self.results_file = None
        self.output_directory = None
        self.file_count = 0
        self.error = None
        self.completed_at = None
        self.duration_seconds = None

    def mark_success(self, output_directory: Path, file_count: int, duration: float):
        """Mark workflow as successful."""
        self.status = "success"
        self.output_directory = str(output_directory)
        self.file_count = file_count
        self.completed_at = datetime.now().isoformat()
        self.duration_seconds = duration

    def mark_failure(self, error: Exception, duration: float):
        """Mark workflow as failed."""
        self.status = "failed"
        self.error = str(error)
        self.completed_at = datetime.now().isoformat()
        self.duration_seconds = duration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_type": self.workflow_type,
            "input_csv": self.input_csv,
            "started_at": self.started_at,
            "status": self.status,
            "immediate_mode": self.immediate_mode,
            "batch_id": self.batch_id,
            "batch_file": self.batch_file,
            "results_file": self.results_file,
            "output_directory": self.output_directory,
            "file_count": self.file_count,
            "error": self.error,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


class WorkflowOrchestrator:
    """Orchestrates complete extraction workflow."""

    def __init__(self, config: WorkflowConfig):
        """
        Initialize workflow orchestrator.

        Args:
            config: WorkflowConfig instance with all settings
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

        # Ensure directories exist
        self.config.batch_jobs_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_review_dir.mkdir(parents=True, exist_ok=True)

    def create_batch(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Create batch requests using appropriate batch creator.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to created batch requests JSONL file
        """
        if progress_callback:
            progress_callback(f"Creating {workflow_type} batch requests...")

        # Select appropriate batch creator
        if workflow_type == "parameter":
            creator = ParameterBatchCreator(self.config.base_dir)
        elif workflow_type == "test_statistic":
            creator = TestStatisticBatchCreator(self.config.base_dir)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Create batch
        output_file = creator.run(None, input_csv)

        if not output_file.exists():
            raise RuntimeError(f"Expected batch file not created: {output_file}")

        if progress_callback:
            progress_callback(f"✓ Batch requests created: {output_file.name}")

        return output_file

    def upload_batch(
        self, batch_file: Path, progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Upload batch to OpenAI API.

        Args:
            batch_file: Path to batch requests JSONL file
            progress_callback: Optional callback for progress updates

        Returns:
            Batch ID
        """
        if progress_callback:
            progress_callback(f"Uploading batch: {batch_file.name}...")

        # Upload file
        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        # Create batch
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window=self.config.batch_completion_window,
        )

        # Save batch metadata
        batch_id_file = batch_file.with_suffix(".batch_id")
        with open(batch_id_file, "w") as f:
            json.dump(
                {
                    "batch_id": batch.id,
                    "batch_type": batch_file.stem.replace("_requests", ""),
                    "source_csv": None,
                    "created_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        if progress_callback:
            progress_callback(f"✓ Batch uploaded: {batch.id}")

        return batch.id

    def monitor_batch(
        self,
        batch_id: str,
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Monitor batch until completion and download results.

        Args:
            batch_id: Batch ID to monitor
            timeout: Maximum seconds to wait (uses config default if None)
            poll_interval: Seconds between checks (uses config default if None)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to downloaded results file

        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If batch fails
        """
        timeout = timeout or self.config.batch_timeout
        poll_interval = poll_interval or self.config.poll_interval

        if progress_callback:
            progress_callback(f"Monitoring batch {batch_id}...")

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

            batch = self.client.batches.retrieve(batch_id)

            # Progress update
            if batch.request_counts and progress_callback:
                completed = batch.request_counts.completed
                total = batch.request_counts.total
                progress_callback(f"  Status: {batch.status} ({completed}/{total} completed)")

            if batch.status == "completed":
                if not batch.output_file_id:
                    raise RuntimeError(f"Batch completed but no output file: {batch_id}")

                # Download results
                content = self.client.files.content(batch.output_file_id)
                output_file = self.config.batch_jobs_dir / f"{batch_id}_results.jsonl"

                with open(output_file, "wb") as f:
                    f.write(content.content)

                if progress_callback:
                    progress_callback(f"✓ Results downloaded: {output_file.name}")

                return output_file

            elif batch.status == "failed":
                raise RuntimeError(f"Batch {batch_id} failed")

            elif batch.status in ["expired", "cancelled"]:
                raise RuntimeError(f"Batch {batch_id} was {batch.status}")

            # Wait before next check
            time.sleep(poll_interval)

    def process_immediate(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Process requests directly via Responses API (no batch file creation).

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to results file (batch-compatible format for unpacker)
        """
        # Create immediate processor
        processor = ImmediateRequestProcessor(self.config.base_dir, self.config.openai_api_key)

        # Process requests directly from CSV
        results = processor.run(input_csv, workflow_type, progress_callback)

        # Write results to file (for unpacker compatibility)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.batch_jobs_dir / f"immediate_{workflow_type}_{timestamp}_results.jsonl"

        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        if progress_callback:
            progress_callback(f"✓ Results saved: {results_file.name}")

        return results_file

    def unpack_results(
        self,
        results_file: Path,
        input_csv: Path,
        workflow_type: str,
        immediate: bool,
        batch_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Path:
        """
        Unpack validated results to unique timestamped directory.

        Args:
            results_file: Path to results JSONL file
            input_csv: Path to original input CSV
            workflow_type: Type of workflow (parameter/test_statistic)
            immediate: True if immediate mode, False if batch mode
            batch_id: Optional batch ID for directory naming
            progress_callback: Optional callback for progress updates

        Returns:
            Path to output directory containing unpacked YAML files
        """
        # Create unique output directory
        output_dir = create_unique_output_directory(
            base_dir=self.config.to_review_dir,
            workflow_type=workflow_type,
            immediate=immediate,
            batch_id=batch_id,
        )

        if progress_callback:
            progress_callback(f"Unpacking results to {output_dir.name}/...")

        # Call unpacker directly (no subprocess)
        process_results(results_file, output_dir, input_csv)

        # Count unpacked files
        unpacked_files = list(output_dir.glob("*.yaml"))
        file_count = len(unpacked_files)

        if progress_callback:
            progress_callback(f"✓ Unpacked {file_count} files to {output_dir.name}/")

        return output_dir

    def run_complete_workflow(
        self,
        input_csv: Path,
        workflow_type: str,
        immediate: bool = False,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> WorkflowResult:
        """
        Run complete extraction workflow from start to finish.

        Args:
            input_csv: Path to input CSV file
            workflow_type: Type of workflow (parameter/test_statistic)
            immediate: Use Responses API for immediate processing (faster, good for testing)
            timeout: Maximum seconds to wait for batch completion (uses config default if None)
            progress_callback: Optional callback for progress updates

        Returns:
            WorkflowResult with execution metadata
        """
        start_time = time.time()
        result = WorkflowResult(workflow_type, input_csv)

        try:
            # Branch: immediate mode or batch mode
            if immediate:
                result.immediate_mode = True

                # Immediate mode: Direct processing via Responses API
                results_file = self.process_immediate(
                    input_csv, workflow_type, progress_callback
                )
                result.results_file = str(results_file)
                batch_id = None

            else:
                # Batch mode: Create batch file, upload, monitor
                batch_file = self.create_batch(input_csv, workflow_type, progress_callback)
                result.batch_file = str(batch_file)

                batch_id = self.upload_batch(batch_file, progress_callback)
                result.batch_id = batch_id

                results_file = self.monitor_batch(
                    batch_id, timeout, progress_callback=progress_callback
                )
                result.results_file = str(results_file)

            # Unpack results to unique directory
            output_dir = self.unpack_results(
                results_file,
                input_csv,
                workflow_type,
                immediate=immediate,
                batch_id=batch_id,
                progress_callback=progress_callback,
            )

            # Count files
            file_count = len(list(output_dir.glob("*.yaml")))

            # Success!
            duration = time.time() - start_time
            result.mark_success(output_dir, file_count, duration)

            return result

        except Exception as e:
            duration = time.time() - start_time
            result.mark_failure(e, duration)
            raise
