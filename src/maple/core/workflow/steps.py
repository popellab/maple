"""
Concrete workflow step implementations.

Each step performs a specific part of the extraction workflow.
"""

import json
import logging
from datetime import datetime

from maple.core.workflow.step import WorkflowStep
from maple.core.workflow.context import WorkflowContext
from maple.core.prompt_builder import (
    ParameterPromptBuilder,
    TestStatisticPromptBuilder,
    CalibrationTargetPromptBuilder,
    IsolatedSystemTargetPromptBuilder,
    SubmodelTargetPromptBuilder,
)
from maple.core.immediate_processor import ImmediateRequestProcessor
from maple.core.output_directory import create_unique_output_directory
from maple.core.exceptions import (
    ImmediateProcessingError,
    ResultsUnpackError,
)
from maple.process.unpack_results import (
    process_results,
    unpack_single_result,
    load_metadata,
)

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
        elif context.workflow_type == "isolated_system_target":
            builder = IsolatedSystemTargetPromptBuilder(context.config.base_dir)
        elif context.workflow_type == "submodel_target":
            builder = SubmodelTargetPromptBuilder(context.config.base_dir)
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
                    context.input_csv,
                    species_units_file,
                    context.config.reasoning_effort,
                    reference_values_file=context.config.reference_values_file,
                )
            elif context.workflow_type == "isolated_system_target":
                # Requires model_structure_file and model_context_file
                model_structure_file = context.config.model_structure_file
                if not model_structure_file:
                    raise ValueError(
                        "model_structure_file is required for isolated_system_target workflow. "
                        "Use --model-structure option."
                    )
                model_context_file = context.config.model_context_file
                if not model_context_file:
                    raise ValueError(
                        "model_context_file is required for isolated_system_target workflow. "
                        "Use --model-context option."
                    )
                species_units_file = context.config.jobs_dir / "input_data" / "species_units.json"
                prompts = builder.process(
                    context.input_csv,
                    model_structure_file,
                    model_context_file,
                    species_units_file if species_units_file.exists() else None,
                    context.config.reasoning_effort,
                )
            elif context.workflow_type == "submodel_target":
                # Requires model_structure_file and model_context_file
                model_structure_file = context.config.model_structure_file
                if not model_structure_file:
                    raise ValueError(
                        "model_structure_file is required for submodel_target workflow. "
                        "Use --model-structure option."
                    )
                model_context_file = context.config.model_context_file
                if not model_context_file:
                    raise ValueError(
                        "model_context_file is required for submodel_target workflow. "
                        "Use --model-context option."
                    )
                species_units_file = context.config.jobs_dir / "input_data" / "species_units.json"
                prompts = builder.process(
                    context.input_csv,
                    model_structure_file,
                    model_context_file,
                    species_units_file if species_units_file.exists() else None,
                    context.config.reasoning_effort,
                    previous_extractions_dir=context.config.previous_extractions_dir,
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
    """Process requests directly via Pydantic AI with streaming unpacking."""

    @property
    def name(self) -> str:
        return "Process Prompts"

    def execute(self, context: WorkflowContext) -> WorkflowContext:
        """Process requests directly via Pydantic AI with streaming unpacking."""
        try:
            # Create unique output directory for streaming unpacking
            output_dir = create_unique_output_directory(
                base_dir=context.config.to_review_dir,
                workflow_type=context.workflow_type,
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            context.report_progress(f"Unpacking results to {output_dir.name}/...")

            # Load metadata once before processing
            metadata = load_metadata(context.input_csv, context.workflow_type)

            # Track unpacked files
            unpacked_count = [0]  # Use list for closure mutation

            # Define streaming unpacker callback
            def unpack_result(result):
                """Unpack each result immediately as it completes."""
                try:
                    output_path = unpack_single_result(
                        result,
                        output_dir,
                        context.workflow_type,
                        metadata,
                        progress_callback=context.progress_callback,
                        previous_extractions_dir=context.config.previous_extractions_dir,
                    )
                    if output_path:
                        unpacked_count[0] += 1
                        context.report_progress(f"  ✓ Saved: {output_path.name}")
                except Exception as e:
                    context.report_progress(f"  ✗ Failed to unpack: {e}")

            # Create immediate processor
            processor = ImmediateRequestProcessor(
                context.config.base_dir,
                context.config.openai_api_key,
                model_structure_file=context.config.model_structure_file,
                model_context_file=context.config.model_context_file,
                reference_values_file=context.config.reference_values_file,
                previous_extractions_dir=context.config.previous_extractions_dir,
            )

            # Process requests with streaming unpacker
            results = processor.run(
                context.input_csv,
                context.workflow_type,
                progress_callback=context.progress_callback,
                result_callback=unpack_result,  # Unpack as each completes
                reasoning_effort=context.config.reasoning_effort,
                model=context.config.openai_model,
                max_retries=context.config.max_retries,
            )

            # Write results to file (for audit trail and debugging)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = (
                context.config.jobs_dir
                / f"immediate_{context.workflow_type}_{timestamp}_results.jsonl"
            )

            with open(results_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

            # Set context fields
            context.results_file = results_file
            context.output_directory = output_dir
            context.file_count = unpacked_count[0]

            context.report_progress(f"✓ Unpacked {unpacked_count[0]} files to {output_dir.name}/")
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
