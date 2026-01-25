#!/usr/bin/env python3
"""
Direct immediate mode processing via Pydantic AI.

Processes CSV rows directly using Pydantic AI.
Uses Pydantic AI with tool calling for structured outputs (supports discriminated unions).
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic_ai import Agent, WebSearchTool, CodeExecutionTool

from qsp_llm_workflows.core.prompt_builder import (
    ParameterPromptBuilder,
    TestStatisticPromptBuilder,
    CalibrationTargetPromptBuilder,
    IsolatedSystemTargetPromptBuilder,
    SubmodelTargetPromptBuilder,
)

# Optional logfire instrumentation (comes with pydantic-ai)
try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


class ImmediateRequestProcessor:
    """Process extraction requests directly via Pydantic AI."""

    def __init__(
        self,
        base_dir: Path,
        api_key: str,
        model_structure_file: Optional[Path] = None,
        model_context_file: Optional[Path] = None,
    ):
        """
        Initialize immediate request processor.

        Args:
            base_dir: Base directory for prompt assembly
            api_key: OpenAI API key
            model_structure_file: Optional path to model_structure.json for isolated system targets
            model_context_file: Optional path to model_context.txt for isolated system targets
        """
        self.base_dir = Path(base_dir)
        self.api_key = api_key
        self.model_structure_file = model_structure_file
        self.model_context_file = model_context_file

        # Setup logfire once (optional instrumentation for debugging)
        if LOGFIRE_AVAILABLE:
            logfire.configure()
            logfire.instrument_pydantic_ai()

        # Initialize prompt builders for prompt building (DRY principle)
        self.parameter_creator = ParameterPromptBuilder(base_dir)
        self.test_statistic_creator = TestStatisticPromptBuilder(base_dir)
        self.calibration_target_creator = CalibrationTargetPromptBuilder(base_dir)
        self.isolated_system_target_creator = IsolatedSystemTargetPromptBuilder(base_dir)
        self.submodel_target_creator = SubmodelTargetPromptBuilder(base_dir)

    def get_prompts(
        self,
        input_csv: Path,
        workflow_type: str,
        species_units_file: Optional[Path] = None,
        reasoning_effort: str = "low",
    ) -> List[Dict[str, Any]]:
        """
        Generate prompts using appropriate prompt builder.

        This ensures DRY principle - all prompt building logic is in prompt builders.

        Args:
            input_csv: Path to input CSV
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            species_units_file: Optional species units file for calibration targets
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            List of prompt dictionaries
        """
        if workflow_type == "parameter":
            # For parameters, pass storage_dir (though not used in immediate mode)
            return self.parameter_creator.process(input_csv, None, reasoning_effort)
        elif workflow_type == "test_statistic":
            return self.test_statistic_creator.process(input_csv, None, reasoning_effort)
        elif workflow_type == "calibration_target":
            return self.calibration_target_creator.process(
                input_csv, species_units_file, reasoning_effort
            )
        elif workflow_type == "isolated_system_target":
            # Requires model_structure_file and model_context_file
            if not self.model_structure_file:
                raise ValueError(
                    "model_structure_file required for isolated_system_target workflow"
                )
            if not self.model_context_file:
                raise ValueError("model_context_file required for isolated_system_target workflow")
            return self.isolated_system_target_creator.process(
                input_csv,
                self.model_structure_file,
                self.model_context_file,
                species_units_file,
                reasoning_effort,
            )
        elif workflow_type == "submodel_target":
            # Requires model_structure_file and model_context_file
            if not self.model_structure_file:
                raise ValueError("model_structure_file required for submodel_target workflow")
            if not self.model_context_file:
                raise ValueError("model_context_file required for submodel_target workflow")
            return self.submodel_target_creator.process(
                input_csv,
                self.model_structure_file,
                self.model_context_file,
                species_units_file,
                reasoning_effort,
            )
        else:
            return []

    async def process_single_request(
        self,
        request: Dict[str, Any],
        index: int,
        reasoning_effort: str,
        workflow_type: str = "parameter",
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a single extraction request using Pydantic AI.

        Args:
            request: Prompt dictionary from prompt builder
            index: Request index
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            workflow_type: Type of workflow for tool selection
            progress_callback: Optional callback for progress updates

        Returns:
            Result dictionary in workflow-compatible format
        """
        # Import here to avoid circular imports and allow lazy loading
        from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

        custom_id = request["custom_id"]
        prompt = request["prompt"]
        pydantic_model = request["pydantic_model"]

        # Extract item name from custom_id for logging
        # Format: cal_target_{id}_{i} or parameter_{name}_{cancer}_{i} or test_stat_{id}_{i}
        parts = custom_id.split("_")
        if len(parts) >= 3:
            # Remove prefix (first 2 parts) and suffix (last part which is index)
            item_name = "_".join(parts[2:-1])
        else:
            item_name = f"item_{index}"

        if progress_callback:
            progress_callback(f"  [{index + 1}] Processing {item_name}...")

        try:
            # Use Pydantic AI (supports discriminated unions via tool calling)
            model = OpenAIResponsesModel("gpt-5.1")
            settings = OpenAIResponsesModelSettings(
                openai_reasoning_summary="concise", openai_reasoning_effort=reasoning_effort
            )
            # Get validation context from request (for species_units)
            validation_context = request.get("validation_context", {})

            # Build tools list - web search and code execution
            builtin_tools = [WebSearchTool(), CodeExecutionTool()]

            agent = Agent(
                model,
                output_type=pydantic_model,
                model_settings=settings,
                builtin_tools=builtin_tools,
                validation_context=validation_context,
                retries=7,  # Increased for validation requirements
            )

            # Run agent with prompt
            result = await agent.run(prompt)
            parsed_data = result.output.model_dump()
            request_id = "pydantic_ai_" + custom_id

            if progress_callback:
                progress_callback(f"  [{index + 1}] ✓ Completed {item_name}")

            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": request_id,
                    "body": parsed_data,
                },
                "error": None,
            }

        except Exception as e:
            if progress_callback:
                progress_callback(f"  [{index + 1}] ✗ Failed {item_name}: {e}")

            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": 500,
                    "request_id": None,
                    "body": {"error": str(e)},
                },
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                },
            }

    async def process_all_requests(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[callable] = None,
        result_callback: Optional[callable] = None,
        reasoning_effort: str = "low",
    ) -> List[Dict[str, Any]]:
        """
        Process all requests from CSV file using Pydantic AI.

        Args:
            input_csv: Path to input CSV file
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            progress_callback: Optional callback for progress updates
            result_callback: Optional callback called immediately as each result completes
            reasoning_effort: Reasoning effort level

        Returns:
            List of results in standard format
        """
        # Get species_units_file for calibration/isolated system/submodel targets
        species_units_file = None
        if workflow_type in ("calibration_target", "isolated_system_target", "submodel_target"):
            species_units_file = self.base_dir / "jobs" / "input_data" / "species_units.json"

        # Generate prompts using prompt builder (DRY principle)
        prompts = self.get_prompts(input_csv, workflow_type, species_units_file, reasoning_effort)

        if progress_callback:
            progress_callback(f"Processing {len(prompts)} requests via Pydantic AI...\n")

        # Create tasks for all requests
        tasks = [
            self.process_single_request(
                prompt, i, reasoning_effort, workflow_type, progress_callback
            )
            for i, prompt in enumerate(prompts)
        ]

        # Process concurrently, calling result_callback as each completes
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)

            # Call result_callback synchronously (it's quick I/O)
            if result_callback:
                result_callback(result)

        if progress_callback:
            success_count = sum(1 for r in results if r.get("error") is None)
            progress_callback(f"\n✓ Completed {success_count}/{len(results)} requests")

        return results

    def run(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[callable] = None,
        result_callback: Optional[callable] = None,
        reasoning_effort: str = "low",
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for async processing via Pydantic AI.

        Args:
            input_csv: Path to input CSV file
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            progress_callback: Optional callback for progress updates
            result_callback: Optional callback called immediately as each result completes
            reasoning_effort: Reasoning effort level

        Returns:
            List of results in standard format
        """
        return asyncio.run(
            self.process_all_requests(
                input_csv, workflow_type, progress_callback, result_callback, reasoning_effort
            )
        )
