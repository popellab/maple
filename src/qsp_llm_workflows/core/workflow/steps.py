"""
Concrete workflow step implementations.

Each step performs a specific part of the extraction workflow.
"""

import json
import logging
from datetime import datetime

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
                species_units_file = context.config.jobs_dir / "input_data" / "species_units.json"
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
        preview_file = context.config.jobs_dir / f"{context.workflow_type}_{timestamp}_preview.txt"

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
        context.preview_file = preview_file
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
                context.config.jobs_dir
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
