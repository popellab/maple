#!/usr/bin/env python3
"""
Direct immediate mode processing via OpenAI Responses API.

Processes CSV rows directly without creating intermediate batch files.
"""

import asyncio
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from qsp_llm_workflows.core.prompts import (
    build_parameter_extraction_prompt,
    build_test_statistic_prompt,
)
from qsp_llm_workflows.core.pydantic_models import (
    ParameterMetadata,
    TestStatistic,
)


class ImmediateRequestProcessor:
    """Process extraction requests directly via Responses API."""

    def __init__(self, base_dir: Path, api_key: str):
        """
        Initialize immediate request processor.

        Args:
            base_dir: Base directory for prompt assembly
            api_key: OpenAI API key
        """
        self.base_dir = Path(base_dir)
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)

    def read_csv_rows(self, input_csv: Path) -> List[Dict[str, str]]:
        """
        Read CSV file into list of row dictionaries.

        Args:
            input_csv: Path to input CSV file

        Returns:
            List of dictionaries, one per row
        """
        rows = []
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def create_custom_id(self, row: Dict[str, str], index: int, workflow_type: str) -> str:
        """
        Create custom ID for request.

        Args:
            row: CSV row data
            index: Row index
            workflow_type: "parameter" or "test_statistic"

        Returns:
            Custom ID string
        """
        if workflow_type == "parameter":
            cancer_type = row.get("cancer_type", "UNKNOWN")
            param_name = row.get("parameter_name", "UNKNOWN")
            return f"{cancer_type}_{param_name}_{index}"
        elif workflow_type == "test_statistic":
            test_stat_id = row.get("test_statistic_id", "UNKNOWN")
            return f"test_stat_{test_stat_id}_{index}"
        else:
            return f"request_{index}"

    def build_prompt(self, row: Dict[str, str], workflow_type: str) -> str:
        """
        Build prompt from CSV row data.

        Args:
            row: CSV row data
            workflow_type: "parameter" or "test_statistic"

        Returns:
            Assembled prompt string
        """
        if workflow_type == "parameter":
            # Extract parameter info from CSV
            param_name = row.get("parameter_name", "")
            param_units = row.get("parameter_units", "")
            param_desc = row.get("parameter_description", "")
            model_context = row.get("model_context", "")

            parameter_info = f"**Parameter Name:** {param_name}\n"
            parameter_info += f"**Units:** {param_units}\n"
            parameter_info += f"**Description:** {param_desc}\n"

            return build_parameter_extraction_prompt(
                parameter_info=parameter_info,
                model_context=model_context,
                used_primary_studies="",  # No used studies tracking in immediate mode yet
            )

        elif workflow_type == "test_statistic":
            # Extract test statistic info from CSV
            required_species = row.get("required_species", "")
            derived_desc = row.get("derived_species_description", "")
            model_context = row.get("model_context", "")
            scenario_context = row.get("scenario_context", "")

            return build_test_statistic_prompt(
                model_context=model_context,
                scenario_context=scenario_context,
                required_species_with_units=required_species,
                derived_species_description=derived_desc,
                used_primary_studies="",  # No used studies tracking yet
            )

        else:
            return ""

    def get_pydantic_model(self, workflow_type: str):
        """
        Get Pydantic model for workflow type.

        Args:
            workflow_type: "parameter" or "test_statistic"

        Returns:
            Pydantic model class
        """
        if workflow_type == "parameter":
            return ParameterMetadata
        elif workflow_type == "test_statistic":
            return TestStatistic
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

    async def process_single_request(
        self, row: Dict[str, str], index: int, workflow_type: str, reasoning_effort: str = "high"
    ) -> Dict[str, Any]:
        """
        Process a single extraction request.

        Args:
            row: CSV row data
            index: Row index
            workflow_type: "parameter" or "test_statistic"

        Returns:
            Result dictionary in batch-compatible format
        """
        custom_id = self.create_custom_id(row, index, workflow_type)

        try:
            # Build prompt from CSV data
            prompt = self.build_prompt(row, workflow_type)

            # Get Pydantic model
            pydantic_model = self.get_pydantic_model(workflow_type)

            # Call Responses API with structured outputs
            response = await self.client.responses.parse(
                model="gpt-5",
                input=prompt,
                reasoning={"effort": reasoning_effort},
                tools=[{"type": "web_search"}],
                text_format=pydantic_model,
            )

            # Use output_parsed and convert to dict
            parsed_data = response.output_parsed.model_dump()

            # Return in simple format (unpacker will handle both batch and immediate formats)
            return {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": response.id,
                    "body": parsed_data,
                },
                "error": None,
            }

        except Exception as e:
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
            workflow_type: "parameter" or "test_statistic"
            progress_callback: Optional callback for progress updates

        Returns:
            List of results in batch-compatible format
        """
        # Read CSV rows
        rows = self.read_csv_rows(input_csv)

        if progress_callback:
            progress_callback(f"Processing {len(rows)} requests via Responses API...")

        # Create tasks for all requests
        tasks = [
            self.process_single_request(row, i, workflow_type, reasoning_effort)
            for i, row in enumerate(rows)
        ]

        # Process concurrently
        results = await asyncio.gather(*tasks)

        if progress_callback:
            success_count = sum(1 for r in results if r.get("error") is None)
            progress_callback(f"✓ Completed {success_count}/{len(results)} requests")

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
            workflow_type: "parameter" or "test_statistic"
            progress_callback: Optional callback for progress updates
            reasoning_effort: Reasoning effort level ("low", "medium", "high")

        Returns:
            List of results in batch-compatible format
        """
        return asyncio.run(
            self.process_all_requests(input_csv, workflow_type, progress_callback, reasoning_effort)
        )
