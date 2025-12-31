#!/usr/bin/env python3
"""
Direct immediate mode processing via OpenAI Responses API or Pydantic AI.

Processes CSV rows directly without creating intermediate batch files.

Two modes:
- OpenAI Responses API (default): Direct API calls with response_format
- Pydantic AI (optional): Uses tool calling for structured outputs, supports discriminated unions
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

from qsp_llm_workflows.core.batch_creator import (
    ParameterBatchCreator,
    TestStatisticBatchCreator,
    CalibrationTargetBatchCreator,
)
from qsp_llm_workflows.core.pydantic_models import (
    ParameterMetadata,
    TestStatistic,
)
from qsp_llm_workflows.core.calibration_target_models import CalibrationTarget


class ImmediateRequestProcessor:
    """Process extraction requests directly via OpenAI API or Pydantic AI."""

    def __init__(self, base_dir: Path, api_key: str, use_pydantic_ai: bool = False):
        """
        Initialize immediate request processor.

        Args:
            base_dir: Base directory for prompt assembly
            api_key: OpenAI API key
            use_pydantic_ai: Use Pydantic AI instead of direct OpenAI API (default: False)
        """
        self.base_dir = Path(base_dir)
        self.api_key = api_key
        self.use_pydantic_ai = use_pydantic_ai

        # Validate Pydantic AI availability if requested
        if use_pydantic_ai and not PYDANTIC_AI_AVAILABLE:
            raise ImportError(
                "Pydantic AI requested but not available. " "Install with: pip install pydantic-ai"
            )

        # Initialize OpenAI client (for non-Pydantic AI mode)
        if not use_pydantic_ai:
            self.client = AsyncOpenAI(api_key=api_key)

        # Initialize batch creators for prompt building (DRY principle)
        self.parameter_creator = ParameterBatchCreator(base_dir)
        self.test_statistic_creator = TestStatisticBatchCreator(base_dir)
        self.calibration_target_creator = CalibrationTargetBatchCreator(base_dir)

    def get_batch_requests(
        self,
        input_csv: Path,
        workflow_type: str,
        species_units_file: Optional[Path] = None,
        reasoning_effort: str = "high",
    ) -> List[Dict[str, Any]]:
        """
        Generate batch requests using appropriate batch creator.

        This ensures DRY principle - all prompt building logic is in batch creators.

        Args:
            input_csv: Path to input CSV
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            species_units_file: Optional species units file for calibration targets
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            List of batch request dictionaries
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
        else:
            return []

    def get_pydantic_model(self, workflow_type: str):
        """
        Get Pydantic model for workflow type.

        Args:
            workflow_type: "parameter", "test_statistic", or "calibration_target"

        Returns:
            Pydantic model class
        """
        if workflow_type == "parameter":
            return ParameterMetadata
        elif workflow_type == "test_statistic":
            return TestStatistic
        elif workflow_type == "calibration_target":
            return CalibrationTarget
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    async def process_single_request(
        self,
        request: Dict[str, Any],
        index: int,
        workflow_type: str,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a single extraction request.

        Args:
            request: Batch request dictionary (from batch creator)
            index: Request index
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            progress_callback: Optional callback for progress updates

        Returns:
            Result dictionary in batch-compatible format
        """
        custom_id = request["custom_id"]
        prompt = request["body"]["input"]
        reasoning_effort = request["body"]["reasoning"]["effort"]

        # Extract item name from custom_id for logging
        item_name = custom_id.split("_")[-2] if "_" in custom_id else f"item_{index}"

        if progress_callback:
            progress_callback(f"  [{index + 1}] Processing {item_name}...")

        try:
            # Get Pydantic model
            pydantic_model = self.get_pydantic_model(workflow_type)

            if self.use_pydantic_ai:
                # Use Pydantic AI (supports discriminated unions via tool calling)

                model = OpenAIResponsesModel("gpt-5.1")
                settings = OpenAIResponsesModelSettings(openai_reasoning_effort=reasoning_effort)
                agent = Agent(model, output_type=pydantic_model, model_settings=settings)

                # Run agent with prompt
                # Note: Pydantic AI doesn't yet support reasoning effort or web_search tools
                result = await agent.run(prompt)
                parsed_data = result.output.model_dump()
                request_id = "pydantic_ai_" + custom_id
            else:
                # Use OpenAI Responses API (default)
                response = await self.client.responses.parse(
                    model="gpt-5.2",
                    input=prompt,
                    reasoning={"effort": reasoning_effort},
                    tools=[{"type": "web_search"}],
                    text_format=pydantic_model,
                )
                parsed_data = response.output_parsed.model_dump()
                request_id = response.id

            if progress_callback:
                progress_callback(f"  [{index + 1}] ✓ Completed {item_name}")

            # Return in simple format (unpacker will handle both batch and immediate formats)
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

            # Return error in batch-compatible format
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
        reasoning_effort: str = "high",
    ) -> List[Dict[str, Any]]:
        """
        Process all requests from CSV file.

        Args:
            input_csv: Path to input CSV file
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            progress_callback: Optional callback for progress updates
            reasoning_effort: Reasoning effort (not used, kept for API compatibility)

        Returns:
            List of results in batch-compatible format
        """
        # Get species_units_file for calibration targets
        species_units_file = None
        if workflow_type == "calibration_target":
            species_units_file = self.base_dir / "batch_jobs" / "input_data" / "species_units.json"

        # Generate batch requests using batch creator (DRY principle)
        requests = self.get_batch_requests(
            input_csv, workflow_type, species_units_file, reasoning_effort
        )

        if progress_callback:
            progress_callback(f"Processing {len(requests)} requests via Responses API...\n")

        # Create tasks for all requests
        tasks = [
            self.process_single_request(request, i, workflow_type, progress_callback)
            for i, request in enumerate(requests)
        ]

        # Process concurrently
        results = await asyncio.gather(*tasks)

        if progress_callback:
            success_count = sum(1 for r in results if r.get("error") is None)
            progress_callback(f"\n✓ Completed {success_count}/{len(results)} requests")

        return results

    def run(
        self,
        input_csv: Path,
        workflow_type: str,
        progress_callback: Optional[callable] = None,
        reasoning_effort: str = "high",
    ) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for async processing.

        Args:
            input_csv: Path to input CSV file
            workflow_type: "parameter", "test_statistic", or "calibration_target"
            progress_callback: Optional callback for progress updates
            reasoning_effort: Reasoning effort level (not used, kept for API compatibility)

        Returns:
            List of results in batch-compatible format
        """
        return asyncio.run(
            self.process_all_requests(input_csv, workflow_type, progress_callback, reasoning_effort)
        )
