#!/usr/bin/env python3
"""
Quick estimate CLI command.

Simple workflow: CSV in -> single LLM request -> CSV out.
"""

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

from qsp_llm_workflows.core.quick_estimate_models import QuickEstimateResponse


def build_prompt(csv_rows: list[dict]) -> str:
    """Build prompt from CSV rows."""
    targets_text = "\n".join(
        [
            f"""- **{row['calibration_target_id']}**
  - Cancer type: {row['cancer_type']}
  - Observable: {row['observable_description']}
  - Model context:
    - Species: {row['model_species']}
    - Indication: {row['model_indication']}
    - Compartment: {row['model_compartment']}
    - System: {row['model_system']}
    - Treatment history: {row['model_treatment_history']}
    - Stage/burden: {row['model_stage_burden']}
"""
            for row in csv_rows
        ]
    )

    prompt = f"""You are extracting quick ballpark estimates for calibration targets from scientific literature.

**IMPORTANT**: You must provide estimates for ALL {len(csv_rows)} calibration targets listed below.

**PAPER REUSE STRATEGY**: Prefer comprehensive PDAC characterization studies that report multiple measurements. If a single high-quality paper contains data for several calibration targets, use it for all applicable targets. This is preferred over using different papers for each target (as long as data quality is sufficient).

For EACH calibration target, you must:
1. Search for papers that MATCH the model context (species, indication, system, treatment history, stage)
2. Extract a numeric estimate value with units AND uncertainty (both required)
3. Find the exact text snippet containing that value
4. Record the paper title and DOI (reuse papers across targets when possible)
5. Describe the measurement threshold/context in words

**CRITICAL MATCHING REQUIREMENTS**:
- Species MUST match (e.g., human vs mouse)
- Indication MUST match (e.g., PDAC vs other cancer types)
- Prioritize matching system/compartment/treatment history
- If you cannot find a study matching the model context, explain this clearly in threshold_description
- Do NOT substitute data from different cancer types unless absolutely necessary

**VALIDATION REQUIREMENTS** (responses will be automatically validated):
1. **DOI must be valid and resolve**: Use real DOIs from actual papers (format: 10.xxxx/journal.year.id)
   - Verify DOI exists before submitting - search PubMed, Google Scholar, or journal websites
   - Common DOI prefixes: 10.1038 (Nature), 10.1126 (Science), 10.1371 (PLOS), 10.1200 (JCO)
2. **Paper title must match DOI**: Use the EXACT title from the paper (will be cross-checked with CrossRef)
   - Copy the title character-for-character from the paper or CrossRef metadata
3. **Estimate value must appear in snippet**: The numeric value MUST be present in the value_snippet
   - Include enough context so the number is clearly visible in the snippet text

**Units formatting**:
- Use Pint-parseable format: 'cell / millimeter**2', 'nanomolarity', 'day', etc.
- Use ** for exponents (not ^)
- Common units: cell, nanomolarity, micromolarity, millimolarity, millimeter, day, dimensionless

**Calibration Targets:**
{targets_text}

Search the literature and provide estimates for ALL targets. Strictly prioritize studies matching the model context.
If you cannot find valid sources that pass validation, explain the difficulty in threshold_description.
"""
    return prompt


async def process_quick_estimates(input_csv: Path, output_csv: Path, api_key: str) -> None:
    """Process quick estimates via single LLM request."""
    # Read input CSV
    with open(input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} calibration targets from {input_csv}")

    # Build prompt
    prompt = build_prompt(rows)

    # Make LLM request
    print("Sending request to LLM...")
    print(f"\nPrompt preview (first 500 chars):\n{prompt[:500]}...\n")

    model = OpenAIResponsesModel("gpt-5.1")
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort="high")
    agent = Agent(
        model,
        output_type=QuickEstimateResponse,
        model_settings=settings,
        builtin_tools=[WebSearchTool()],
        retries=7,  # Increased for validation requirements
    )

    result = await agent.run(prompt)
    response = result.output

    print(f"Received {len(response.estimates)} estimates from LLM")
    if len(response.estimates) == 0:
        print("WARNING: LLM returned empty list. Check if the prompt is clear enough.")

    # Write output CSV
    fieldnames = [
        "calibration_target_id",
        "estimate",
        "units",
        "uncertainty",
        "uncertainty_type",
        "value_snippet",
        "paper_name",
        "doi",
        "threshold_description",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for estimate in response.estimates:
            writer.writerow(estimate.model_dump())

    print(f"✓ Wrote {len(response.estimates)} estimates to {output_csv}")


def main():
    """Main entry point for qsp-quick-estimate command."""
    # Load environment variables from .env file
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Quick calibration target estimate extraction (CSV -> CSV)"
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV with calibration targets")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CSV path")

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Run async processing
    asyncio.run(process_quick_estimates(args.input_csv, args.output, api_key))


if __name__ == "__main__":
    main()
