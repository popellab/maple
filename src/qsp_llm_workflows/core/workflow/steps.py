"""
Concrete workflow step implementations.

Each step performs a specific part of the extraction workflow.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from qsp_llm_workflows.core.workflow.step import WorkflowStep
from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.batch_creator import (
    ParameterBatchCreator,
    TestStatisticBatchCreator,
)
from qsp_llm_workflows.core.immediate_processor import ImmediateRequestProcessor
from qsp_llm_workflows.core.output_directory import create_unique_output_directory
from qsp_llm_workflows.process.unpack_results import process_results


class CreateBatchStep(WorkflowStep):
    """Create batch requests from input CSV."""

    @property
    def name(self) -> str:
        return "Create Batch"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Create batch requests using appropriate batch creator."""
        context.report_progress(f"Creating {context.workflow_type} batch requests...")

        # Select appropriate batch creator
        if context.workflow_type == "parameter":
            creator = ParameterBatchCreator(context.config.base_dir)
        elif context.workflow_type == "test_statistic":
            creator = TestStatisticBatchCreator(context.config.base_dir)
        else:
            raise ValueError(f"Unknown workflow type: {context.workflow_type}")

        # Create batch
        output_file = creator.run(None, context.input_csv)

        if not output_file.exists():
            raise RuntimeError(f"Expected batch file not created: {output_file}")

        context.batch_file = output_file
        context.report_progress(f"✓ Batch requests created: {output_file.name}")

        return context


class UploadBatchStep(WorkflowStep):
    """Upload batch file to OpenAI API."""

    @property
    def name(self) -> str:
        return "Upload Batch"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Upload batch to OpenAI API."""
        if not context.batch_file:
            raise RuntimeError("Batch file not found in context")

        context.report_progress(f"Uploading batch: {context.batch_file.name}...")

        # Create OpenAI client
        client = OpenAI(api_key=context.config.openai_api_key)

        # Upload file
        with open(context.batch_file, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")

        # Create batch
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window=context.config.batch_completion_window,
        )

        # Save batch metadata
        batch_id_file = context.batch_file.with_suffix(".batch_id")
        with open(batch_id_file, "w") as f:
            json.dump(
                {
                    "batch_id": batch.id,
                    "batch_type": context.batch_file.stem.replace("_requests", ""),
                    "source_csv": str(context.input_csv),
                    "created_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        context.batch_id = batch.id
        context.report_progress(f"✓ Batch uploaded: {batch.id}")

        return context


class MonitorBatchStep(WorkflowStep):
    """Monitor batch until completion and download results."""

    @property
    def name(self) -> str:
        return "Monitor Batch"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Monitor batch until completion and download results."""
        if not context.batch_id:
            raise RuntimeError("Batch ID not found in context")

        context.report_progress(f"Monitoring batch {context.batch_id}...")

        # Create OpenAI client
        client = OpenAI(api_key=context.config.openai_api_key)

        start_time = time.time()
        timeout = context.config.batch_timeout
        poll_interval = context.config.poll_interval

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Batch {context.batch_id} did not complete within {timeout}s"
                )

            batch = client.batches.retrieve(context.batch_id)

            # Progress update
            if batch.request_counts:
                completed = batch.request_counts.completed
                total = batch.request_counts.total
                context.report_progress(
                    f"  Status: {batch.status} ({completed}/{total} completed)"
                )

            if batch.status == "completed":
                if not batch.output_file_id:
                    raise RuntimeError(
                        f"Batch completed but no output file: {context.batch_id}"
                    )

                # Download results
                content = client.files.content(batch.output_file_id)
                output_file = context.config.batch_jobs_dir / f"{context.batch_id}_results.jsonl"

                with open(output_file, "wb") as f:
                    f.write(content.content)

                context.results_file = output_file
                context.report_progress(f"✓ Results downloaded: {output_file.name}")

                return context

            elif batch.status == "failed":
                raise RuntimeError(f"Batch {context.batch_id} failed")

            elif batch.status in ["expired", "cancelled"]:
                raise RuntimeError(f"Batch {context.batch_id} was {batch.status}")

            # Wait before next check
            time.sleep(poll_interval)


class ProcessImmediateStep(WorkflowStep):
    """Process requests directly via Responses API (no batch)."""

    @property
    def name(self) -> str:
        return "Process Immediate"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Process requests directly via Responses API."""
        # Create immediate processor
        processor = ImmediateRequestProcessor(
            context.config.base_dir, context.config.openai_api_key
        )

        # Process requests directly from CSV
        results = processor.run(
            context.input_csv, context.workflow_type, context.progress_callback
        )

        # Write results to file (for unpacker compatibility)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = (
            context.config.batch_jobs_dir
            / f"immediate_{context.workflow_type}_{timestamp}_results.jsonl"
        )

        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        context.results_file = results_file
        context.report_progress(f"✓ Results saved: {results_file.name}")

        return context


class UnpackResultsStep(WorkflowStep):
    """Unpack results to unique timestamped directory."""

    @property
    def name(self) -> str:
        return "Unpack Results"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Unpack validated results to unique timestamped directory."""
        if not context.results_file:
            raise RuntimeError("Results file not found in context")

        # Create unique output directory
        output_dir = create_unique_output_directory(
            base_dir=context.config.to_review_dir,
            workflow_type=context.workflow_type,
            immediate=context.immediate,
            batch_id=context.batch_id,
        )

        context.report_progress(f"Unpacking results to {output_dir.name}/...")

        # Call unpacker directly
        process_results(context.results_file, output_dir, context.input_csv)

        # Count unpacked files
        unpacked_files = list(output_dir.glob("*.yaml"))
        file_count = len(unpacked_files)

        context.output_directory = output_dir
        context.file_count = file_count
        context.report_progress(f"✓ Unpacked {file_count} files to {output_dir.name}/")

        return context
