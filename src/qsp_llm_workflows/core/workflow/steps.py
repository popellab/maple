"""
Concrete workflow step implementations.

Each step performs a specific part of the extraction workflow.
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from openai import OpenAI, AsyncOpenAI

from qsp_llm_workflows.core.workflow.step import WorkflowStep
from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.batch_creator import (
    ParameterBatchCreator,
    TestStatisticBatchCreator,
    CalibrationTargetBatchCreator,
)
from qsp_llm_workflows.core.immediate_processor import ImmediateRequestProcessor
from qsp_llm_workflows.core.output_directory import create_unique_output_directory
from qsp_llm_workflows.core.exceptions import (
    BatchCreationError,
    BatchUploadError,
    BatchMonitoringError,
    BatchTimeoutError,
    ImmediateProcessingError,
    ResultsUnpackError,
)
from qsp_llm_workflows.process.unpack_results import process_results
from qsp_llm_workflows.prepare.create_validation_fix_batch import ValidationFixBatchCreator
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic

logger = logging.getLogger(__name__)


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
        elif context.workflow_type == "calibration_target":
            creator = CalibrationTargetBatchCreator(context.config.base_dir)
        else:
            logger.error("Unknown workflow type: %s", context.workflow_type)
            raise BatchCreationError(
                f"Unknown workflow type: {context.workflow_type}",
                context={"workflow_type": context.workflow_type},
            )

        # Create batch
        try:
            # For parameter workflow, pass storage_dir for existing studies lookup
            if context.workflow_type == "parameter":
                parameter_storage_dir = context.config.storage_dir / "parameter_estimates"
                output_file = creator.run(None, context.input_csv, parameter_storage_dir)
            elif context.workflow_type == "calibration_target":
                # For calibration targets, pass species_units_file if it exists
                species_units_file = (
                    context.config.batch_jobs_dir / "input_data" / "species_units.json"
                )
                output_file = creator.run(None, context.input_csv, species_units_file)
            else:
                output_file = creator.run(None, context.input_csv)
            logger.debug("Batch creator returned file: %s", output_file)
        except Exception as e:
            logger.error("Batch creation failed: %s", e, exc_info=True)
            raise BatchCreationError(
                f"Failed to create batch requests: {e}",
                context={
                    "workflow_type": context.workflow_type,
                    "input_csv": str(context.input_csv),
                },
            ) from e

        if not output_file.exists():
            logger.error("Batch file not found: %s", output_file)
            raise BatchCreationError(
                f"Expected batch file not created: {output_file}",
                context={"expected_file": str(output_file)},
            )

        context.batch_file = output_file

        # If in preview mode, create human-readable preview file
        if context.get_metadata("preview_prompts", False):
            preview_file = output_file.with_suffix(".preview.txt")
            self._create_preview_file(output_file, preview_file)
            context.report_progress(f"✓ Prompt preview created: {preview_file.name}")
        else:
            context.report_progress(f"✓ Batch requests created: {output_file.name}")

        return context

    def _create_preview_file(self, batch_file: Path, preview_file: Path):
        """Create a human-readable preview file from batch requests."""
        with open(batch_file) as f, open(preview_file, "w") as out:
            requests = [json.loads(line) for line in f]

            out.write("=" * 80 + "\n")
            out.write(f"PROMPT PREVIEW - {len(requests)} requests\n")
            out.write("=" * 80 + "\n\n")

            for i, request in enumerate(requests, 1):
                out.write(f"\n{'=' * 80}\n")
                out.write(f"REQUEST {i}/{len(requests)}\n")
                out.write(f"{'=' * 80}\n")
                out.write(f"Custom ID: {request['custom_id']}\n")
                out.write(f"Model: {request['body'].get('model', 'N/A')}\n\n")

                # Extract the prompt from the request
                if "input" in request["body"]:
                    # Responses API format
                    prompt = request["body"]["input"]
                elif "messages" in request["body"]:
                    # Chat completions format (shouldn't happen, but handle it)
                    messages = request["body"]["messages"]
                    prompt = "\n\n".join(
                        f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages
                    )
                else:
                    prompt = "(No prompt found)"

                out.write("PROMPT:\n")
                out.write("-" * 80 + "\n")
                out.write(prompt)
                out.write("\n" + "-" * 80 + "\n")

            out.write(f"\n{'=' * 80}\n")
            out.write("END OF PREVIEW\n")
            out.write(f"Total requests: {len(requests)}\n")
            out.write("=" * 80 + "\n")


class CreateValidationFixBatchStep(WorkflowStep):
    """Create batch requests to fix validation errors."""

    @property
    def name(self) -> str:
        return "Create Validation Fix Batch"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Create batch fix requests from validation reports."""
        context.report_progress("Creating validation fix batch requests...")

        # Get data directory and validation results directory from context
        data_dir = context.get_metadata("data_dir")
        validation_results_dir = context.get_metadata("validation_results_dir")

        if not data_dir or not validation_results_dir:
            raise BatchCreationError(
                "Missing required metadata: data_dir or validation_results_dir",
                context={
                    "data_dir": data_dir,
                    "validation_results_dir": validation_results_dir,
                },
            )

        # Determine model class
        if context.workflow_type == "parameter":
            model_class = ParameterMetadata
        elif context.workflow_type == "test_statistic":
            model_class = TestStatistic
        else:
            raise BatchCreationError(
                f"Unknown workflow type: {context.workflow_type}",
                context={"workflow_type": context.workflow_type},
            )

        # Create validation fix batch
        try:
            # Generate unique output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                context.config.batch_jobs_dir
                / f"validation_fix_{context.workflow_type}_{timestamp}_requests.jsonl"
            )

            creator = ValidationFixBatchCreator(
                data_dir=str(data_dir),
                validation_results_dir=str(validation_results_dir),
                output_file=str(output_file),
                model_class=model_class,
            )

            # Create and save batch using run() method
            creator.run(output_file)

        except Exception as e:
            logger.error("Validation fix batch creation failed: %s", e, exc_info=True)
            raise BatchCreationError(
                f"Failed to create validation fix batch: {e}",
                context={
                    "workflow_type": context.workflow_type,
                    "data_dir": data_dir,
                    "validation_results_dir": validation_results_dir,
                },
            ) from e

        if not output_file.exists():
            logger.error("Batch file not found: %s", output_file)
            raise BatchCreationError(
                f"Expected batch file not created: {output_file}",
                context={"expected_file": str(output_file)},
            )

        context.batch_file = output_file

        # If in preview mode, create human-readable preview file
        if context.get_metadata("preview_prompts", False):
            preview_file = output_file.with_suffix(".preview.txt")
            self._create_preview_file(output_file, preview_file)
            context.report_progress(f"✓ Validation fix prompt preview created: {preview_file.name}")
        else:
            context.report_progress(f"✓ Validation fix batch created: {output_file.name}")

        return context

    def _create_preview_file(self, batch_file: Path, preview_file: Path):
        """Create a human-readable preview file from batch requests."""
        with open(batch_file) as f, open(preview_file, "w") as out:
            requests = [json.loads(line) for line in f]

            out.write("=" * 80 + "\n")
            out.write(f"VALIDATION FIX PROMPT PREVIEW - {len(requests)} requests\n")
            out.write("=" * 80 + "\n\n")

            for i, request in enumerate(requests, 1):
                out.write(f"\n{'=' * 80}\n")
                out.write(f"REQUEST {i}/{len(requests)}\n")
                out.write(f"{'=' * 80}\n")
                out.write(f"Custom ID: {request['custom_id']}\n")
                out.write(f"Model: {request['body'].get('model', 'N/A')}\n\n")

                # Extract the prompt from the request
                if "input" in request["body"]:
                    # Responses API format
                    prompt = request["body"]["input"]
                elif "messages" in request["body"]:
                    # Chat completions format (shouldn't happen, but handle it)
                    messages = request["body"]["messages"]
                    prompt = "\n\n".join(
                        f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages
                    )
                else:
                    prompt = "(No prompt found)"

                out.write("PROMPT:\n")
                out.write("-" * 80 + "\n")
                out.write(prompt)
                out.write("\n" + "-" * 80 + "\n")

            out.write(f"\n{'=' * 80}\n")
            out.write("END OF PREVIEW\n")
            out.write(f"Total requests: {len(requests)}\n")
            out.write("=" * 80 + "\n")


class UploadBatchStep(WorkflowStep):
    """Upload batch file to OpenAI API."""

    @property
    def name(self) -> str:
        return "Upload Batch"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Upload batch to OpenAI API."""
        if not context.batch_file:
            raise BatchUploadError(
                "Batch file not found in context", context={"workflow_type": context.workflow_type}
            )

        context.report_progress(f"Uploading batch: {context.batch_file.name}...")

        try:
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
        except Exception as e:
            raise BatchUploadError(
                f"Failed to upload batch to OpenAI: {e}",
                context={
                    "batch_file": str(context.batch_file),
                    "workflow_type": context.workflow_type,
                },
            ) from e

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
            raise BatchMonitoringError(
                "Batch ID not found in context", context={"workflow_type": context.workflow_type}
            )

        context.report_progress(f"Monitoring batch {context.batch_id}...")

        try:
            # Create OpenAI client
            client = OpenAI(api_key=context.config.openai_api_key)

            start_time = time.time()
            timeout = context.config.batch_timeout
            poll_interval = context.config.poll_interval

            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise BatchTimeoutError(
                        f"Batch did not complete within {timeout}s",
                        batch_id=context.batch_id,
                        timeout=timeout,
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
                        raise BatchMonitoringError(
                            "Batch completed but no output file",
                            context={"batch_id": context.batch_id},
                        )

                    # Download results
                    content = client.files.content(batch.output_file_id)
                    output_file = (
                        context.config.batch_jobs_dir / f"{context.batch_id}_results.jsonl"
                    )

                    with open(output_file, "wb") as f:
                        f.write(content.content)

                    context.results_file = output_file
                    context.report_progress(f"✓ Results downloaded: {output_file.name}")

                    return context

                elif batch.status == "failed":
                    raise BatchMonitoringError(
                        "Batch failed",
                        context={"batch_id": context.batch_id, "status": batch.status},
                    )

                elif batch.status in ["expired", "cancelled"]:
                    raise BatchMonitoringError(
                        f"Batch was {batch.status}",
                        context={"batch_id": context.batch_id, "status": batch.status},
                    )

                # Wait before next check
                time.sleep(poll_interval)

        except (BatchTimeoutError, BatchMonitoringError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            raise BatchMonitoringError(
                f"Failed to monitor batch: {e}", context={"batch_id": context.batch_id}
            ) from e


class ProcessImmediateStep(WorkflowStep):
    """Process requests directly via Responses API (no batch)."""

    @property
    def name(self) -> str:
        return "Process Immediate"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Process requests directly via Responses API."""
        try:
            # Create immediate processor
            processor = ImmediateRequestProcessor(
                context.config.base_dir, context.config.openai_api_key
            )

            # Get reasoning effort from context (default to high)
            reasoning_effort = context.get_metadata("reasoning_effort", "high")

            # Process requests directly from CSV
            results = processor.run(
                context.input_csv,
                context.workflow_type,
                context.progress_callback,
                reasoning_effort=reasoning_effort,
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

        except Exception as e:
            raise ImmediateProcessingError(
                f"Failed to process requests via Responses API: {e}",
                context={
                    "input_csv": str(context.input_csv),
                    "workflow_type": context.workflow_type,
                },
            ) from e


class UnpackResultsStep(WorkflowStep):
    """Unpack results to unique timestamped directory."""

    @property
    def name(self) -> str:
        return "Unpack Results"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Unpack validated results to unique timestamped directory."""
        if not context.results_file:
            raise ResultsUnpackError(
                "Results file not found in context",
                context={"workflow_type": context.workflow_type},
            )

        try:
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

        except ResultsUnpackError:
            # Re-raise our custom exception as-is
            raise
        except Exception as e:
            raise ResultsUnpackError(
                f"Failed to unpack results: {e}",
                context={
                    "results_file": str(context.results_file),
                    "workflow_type": context.workflow_type,
                },
            ) from e


class ProcessImmediateValidationFixStep(WorkflowStep):
    """Process validation fix requests directly via Responses API (no batch)."""

    @property
    def name(self) -> str:
        return "Process Immediate Validation Fix"

    async def _process_single_fix(
        self,
        client: AsyncOpenAI,
        request: dict,
        idx: int,
        pydantic_model,
        reasoning_effort: str,
        progress_callback,
    ) -> dict:
        """Process a single validation fix request asynchronously."""
        custom_id = request["custom_id"]
        body = request["body"]

        if progress_callback:
            progress_callback(f"  [{idx}] Processing {custom_id}...")

        try:
            # Call Responses API with structured outputs (same as extraction workflow)
            response = await client.responses.parse(
                model=body["model"],
                input=body["input"],
                reasoning=body.get("reasoning", {"effort": reasoning_effort}),
                text_format=pydantic_model,  # Direct Pydantic model
            )

            # Use output_parsed and convert to dict (same as extraction workflow)
            parsed_data = response.output_parsed.model_dump()

            if progress_callback:
                progress_callback(f"  [{idx}] ✓ Completed {custom_id}")

            # Format as batch-style result
            return {
                "id": f"immediate_req_{idx}",
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {"output_parsed": parsed_data},
                },
                "error": None,
            }

        except Exception as e:
            logger.error(f"Failed to process {custom_id}: {e}")
            if progress_callback:
                progress_callback(f"  [{idx}] ✗ Failed {custom_id}: {e}")

            # Add error result
            return {
                "id": f"immediate_req_{idx}",
                "custom_id": custom_id,
                "response": None,
                "error": {"message": str(e)},
            }

    async def _process_all_fixes(
        self,
        requests: list,
        api_key: str,
        pydantic_model,
        reasoning_effort: str,
        progress_callback,
    ) -> list:
        """Process all validation fix requests concurrently."""
        # Create async OpenAI client
        client = AsyncOpenAI(api_key=api_key)

        # Create tasks for all requests (process concurrently)
        tasks = [
            self._process_single_fix(
                client, request, idx, pydantic_model, reasoning_effort, progress_callback
            )
            for idx, request in enumerate(requests, 1)
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        return results

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Process validation fix requests directly via Responses API."""
        context.report_progress("Processing validation fixes via Responses API...")

        # Get data directory and validation results directory from context
        data_dir = context.get_metadata("data_dir")
        validation_results_dir = context.get_metadata("validation_results_dir")

        if not data_dir or not validation_results_dir:
            raise ImmediateProcessingError(
                "Missing required metadata: data_dir or validation_results_dir",
                context={
                    "data_dir": data_dir,
                    "validation_results_dir": validation_results_dir,
                },
            )

        # Determine model class
        if context.workflow_type == "parameter":
            model_class = ParameterMetadata
        elif context.workflow_type == "test_statistic":
            model_class = TestStatistic
        else:
            raise ImmediateProcessingError(
                f"Unknown workflow type: {context.workflow_type}",
                context={"workflow_type": context.workflow_type},
            )

        try:
            # Create validation fix batch creator to get requests
            creator = ValidationFixBatchCreator(
                data_dir=str(data_dir),
                validation_results_dir=str(validation_results_dir),
                output_file="temp.jsonl",  # Not used, but required by constructor
                model_class=model_class,
            )

            # Generate requests (but don't save to file)
            requests = creator.create_batch_requests()

            if not requests:
                context.report_progress("No validation errors found - nothing to fix")
                # Return empty results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = (
                    context.config.batch_jobs_dir
                    / f"immediate_validation_fix_{context.workflow_type}_{timestamp}_results.jsonl"
                )
                # Create empty file
                results_file.touch()
                context.results_file = results_file
                context.report_progress("✓ No fixes needed")
                return context

            # Get reasoning effort from context (default to "medium" for validation fixes)
            reasoning_effort = context.get_metadata("reasoning_effort", "medium")

            # Determine pydantic model
            if context.workflow_type == "parameter":
                pydantic_model = ParameterMetadata
            elif context.workflow_type == "test_statistic":
                pydantic_model = TestStatistic
            else:
                raise ImmediateProcessingError(
                    f"Unknown workflow type: {context.workflow_type}",
                    context={"workflow_type": context.workflow_type},
                )

            # Process all requests concurrently using async
            context.report_progress(f"Processing {len(requests)} fixes concurrently...")
            results = asyncio.run(
                self._process_all_fixes(
                    requests,
                    context.config.openai_api_key,
                    pydantic_model,
                    reasoning_effort,
                    context.progress_callback,
                )
            )

            # Write results to file (for unpacker compatibility)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = (
                context.config.batch_jobs_dir
                / f"immediate_validation_fix_{context.workflow_type}_{timestamp}_results.jsonl"
            )

            with open(results_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

            context.results_file = results_file
            context.report_progress(f"✓ Processed {len(results)} validation fixes")

            return context

        except Exception as e:
            raise ImmediateProcessingError(
                f"Failed to process validation fixes via Responses API: {e}",
                context={
                    "data_dir": str(data_dir),
                    "validation_results_dir": str(validation_results_dir),
                    "workflow_type": context.workflow_type,
                },
            ) from e


class UnpackValidationFixResultsStep(WorkflowStep):
    """Unpack validation fix results back to original directory (overwrites)."""

    @property
    def name(self) -> str:
        return "Unpack Validation Fix Results"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Unpack fixed results back to original data directory."""
        if not context.results_file:
            raise ResultsUnpackError(
                "Results file not found in context",
                context={"workflow_type": context.workflow_type},
            )

        # Get original data directory from context
        data_dir = context.get_metadata("data_dir")
        if not data_dir:
            raise ResultsUnpackError(
                "Data directory not found in context",
                context={"workflow_type": context.workflow_type},
            )

        try:
            output_dir = Path(data_dir)

            context.report_progress(f"Unpacking fixed results to {output_dir.name}/...")

            # Call unpacker directly (overwrites original files)
            process_results(context.results_file, output_dir, input_csv=None)

            # Count unpacked files
            unpacked_files = list(output_dir.glob("*.yaml"))
            file_count = len(unpacked_files)

            context.output_directory = output_dir
            context.file_count = file_count
            context.report_progress(f"✓ Fixed {file_count} files in {output_dir.name}/")

            return context

        except ResultsUnpackError:
            # Re-raise our custom exception as-is
            raise
        except Exception as e:
            raise ResultsUnpackError(
                f"Failed to unpack validation fix results: {e}",
                context={
                    "results_file": str(context.results_file),
                    "data_dir": str(data_dir),
                    "workflow_type": context.workflow_type,
                },
            ) from e
