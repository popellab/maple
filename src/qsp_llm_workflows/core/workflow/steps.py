"""
Concrete workflow step implementations.

Each step performs a specific part of the extraction workflow.
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

from qsp_llm_workflows.core.workflow.step import WorkflowStep
from qsp_llm_workflows.core.workflow.context import WorkflowContext
from qsp_llm_workflows.core.prompt_builder import (
    ParameterPromptBuilder,
    TestStatisticPromptBuilder,
    CalibrationTargetPromptBuilder,
)
from qsp_llm_workflows.core.immediate_processor import ImmediateRequestProcessor
from qsp_llm_workflows.core.output_directory import create_unique_output_directory
from qsp_llm_workflows.core.exceptions import (
    ImmediateProcessingError,
    ResultsUnpackError,
)
from qsp_llm_workflows.process.unpack_results import process_results
from qsp_llm_workflows.prepare.create_validation_fix_batch import ValidationFixPromptBuilder
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic

logger = logging.getLogger(__name__)


class CreatePreviewStep(WorkflowStep):
    """Generate prompts and create human-readable preview file."""

    @property
    def name(self) -> str:
        return "Create Preview"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Generate prompts and write preview file."""
        context.report_progress(f"Creating {context.workflow_type} prompt preview...")

        # Select appropriate prompt builder
        if context.workflow_type == "parameter":
            builder = ParameterPromptBuilder(context.config.base_dir)
        elif context.workflow_type == "test_statistic":
            builder = TestStatisticPromptBuilder(context.config.base_dir)
        elif context.workflow_type == "calibration_target":
            builder = CalibrationTargetPromptBuilder(context.config.base_dir)
        else:
            raise ValueError(f"Unknown workflow type: {context.workflow_type}")

        # Generate prompts
        try:
            if context.workflow_type == "parameter":
                parameter_storage_dir = context.config.storage_dir / "parameter_estimates"
                prompts = builder.process(
                    context.input_csv, parameter_storage_dir, context.config.reasoning_effort
                )
            elif context.workflow_type == "calibration_target":
                species_units_file = (
                    context.config.batch_jobs_dir / "input_data" / "species_units.json"
                )
                prompts = builder.process(
                    context.input_csv, species_units_file, context.config.reasoning_effort
                )
            else:
                prompts = builder.process(context.input_csv, None, context.config.reasoning_effort)
        except Exception as e:
            raise ImmediateProcessingError(
                f"Failed to generate prompts: {e}",
                context={
                    "workflow_type": context.workflow_type,
                    "input_csv": str(context.input_csv),
                },
            ) from e

        # Generate output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_file = (
            context.config.batch_jobs_dir / f"{context.workflow_type}_{timestamp}_preview.txt"
        )

        # Write preview file
        with open(preview_file, "w") as out:
            out.write("=" * 80 + "\n")
            out.write(f"PROMPT PREVIEW - {len(prompts)} prompts\n")
            out.write(f"Workflow: {context.workflow_type}\n")
            out.write(f"Input: {context.input_csv.name}\n")
            out.write(f"Reasoning effort: {context.config.reasoning_effort}\n")
            out.write("=" * 80 + "\n\n")

            for i, prompt_dict in enumerate(prompts, 1):
                out.write(f"\n{'=' * 80}\n")
                out.write(f"PROMPT {i}/{len(prompts)}\n")
                out.write(f"{'=' * 80}\n")
                out.write(f"Custom ID: {prompt_dict['custom_id']}\n")
                out.write(f"Model: {prompt_dict['pydantic_model'].__name__}\n\n")

                out.write("PROMPT:\n")
                out.write("-" * 80 + "\n")
                out.write(prompt_dict["prompt"])
                out.write("\n" + "-" * 80 + "\n")

            out.write(f"\n{'=' * 80}\n")
            out.write("END OF PREVIEW\n")
            out.write(f"Total prompts: {len(prompts)}\n")
            out.write("=" * 80 + "\n")

        # Set context fields (for compatibility with unpacker)
        context.batch_file = preview_file
        context.file_count = len(prompts)

        context.report_progress(f"✓ Prompt preview created: {preview_file.name}")
        return context


class ProcessPromptsStep(WorkflowStep):
    """Process requests directly via Pydantic AI."""

    @property
    def name(self) -> str:
        return "Process Prompts"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Process requests directly via Pydantic AI."""
        try:
            # Create immediate processor
            processor = ImmediateRequestProcessor(
                context.config.base_dir,
                context.config.openai_api_key,
            )

            # Process requests directly from CSV
            results = processor.run(
                context.input_csv,
                context.workflow_type,
                context.progress_callback,
                reasoning_effort=context.config.reasoning_effort,
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


class ProcessValidationFixStep(WorkflowStep):
    """Process validation fix requests directly via Pydantic AI."""

    @property
    def name(self) -> str:
        return "Process Validation Fix"

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
            creator = ValidationFixPromptBuilder(
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
