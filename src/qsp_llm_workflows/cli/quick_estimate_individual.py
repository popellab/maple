#!/usr/bin/env python3
"""
Quick estimate CLI command - individual processing.

Processes each calibration target as a separate LLM request.
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

from qsp_llm_workflows.core.quick_estimate_models import QuickTargetEstimate


def build_prompt_for_target(row: dict) -> str:
    """Build prompt for a single calibration target."""
    prompt = f"""You are extracting a quick ballpark estimate for a calibration target from scientific literature.

**Calibration Target:**
- **ID**: {row['calibration_target_id']}
- **Cancer type**: {row['cancer_type']}
- **Observable**: {row['observable_description']}
- **Model context**:
  - Species: {row['model_species']}
  - Indication: {row['model_indication']}
  - Compartment: {row['model_compartment']}
  - System: {row['model_system']}
  - Treatment history: {row['model_treatment_history']}
  - Stage/burden: {row['model_stage_burden']}

Your task:
1. Search THOROUGHLY for papers that MATCH the model context (species, indication, system, treatment history, stage)
   - Try multiple search strategies and keywords
   - Look for quantitative image analysis studies, flow cytometry studies, tissue microarray studies
   - Check recent reviews for cited quantitative data
   - If values appear only in figures/tables, describe what you found and note that digitization may be needed
2. Extract a numeric estimate value with units AND uncertainty (both required)
3. Find the exact text snippet containing that value
4. Record the paper title and DOI
5. Describe the measurement threshold/context in words

**CRITICAL MATCHING REQUIREMENTS**:
- **Species MUST match exactly** (human vs mouse - NO exceptions)
- **Indication MUST match exactly** (PDAC vs other cancers - NO exceptions)
- **Compartment MUST match exactly** (tumor tissue vs serum/plasma - do NOT substitute different biological compartments)
- **Tissue type MUST match** (PDAC tumor vs normal pancreas - do NOT substitute)
- **Analyte MUST match exactly** (do NOT substitute related but distinct molecular markers, cell types, or measurement modalities)
- **Units MUST match the requested physical quantity**:
  - If units are specified, do NOT report measurements in incompatible units
  - Cell density (cells/mm²) ≠ percent of total cells (dimensionless)
  - Do NOT convert between fundamentally different measurement types
- **Source MUST be in vivo patient data**:
  - Do NOT use cell culture, media formulations, or in vitro experimental conditions
  - ONLY use measurements from actual patient samples (tissue, blood, etc.)
- **Do NOT use statistical effect sizes as calibration values**:
  - NO hazard ratios, odds ratios, regression coefficients, or p-values
- **If you cannot find valid matching data**:
  - Do NOT fabricate placeholder values or use fake DOIs
  - Validation will fail and the request will retry with different search strategies

**VALIDATION REQUIREMENTS** (response will be automatically validated):
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

**Pint usage in model_output_code** (CRITICAL):
When writing the `model_output_code` function, follow the GOLDEN RULE:
**"Keep values tethered to their units throughout calculations."**

✓ CORRECT approach:
```python
def compute_test_statistic(time, species_dict, ureg):
    import numpy as np
    cd8 = species_dict['V_T.CD8']  # Already has units (cell)
    V_T = species_dict['V_T']  # Already has units (milliliter)
    density_3d = (cd8 / V_T).to('cell / millimeter**3')  # Units propagate
    section_thickness = 0.005 * ureg.millimeter  # Tethered to units
    density_2d = density_3d * section_thickness  # Pint handles dimensions
    return density_2d.to('cell / millimeter**2')  # Final conversion
```

✗ INCORRECT approach:
```python
def compute_test_statistic(time, species_dict, ureg):
    cd8_value = species_dict['V_T.CD8'].magnitude  # ✗ Extracting magnitude too early!
    vt_value = species_dict['V_T'].magnitude  # ✗ Units lost!
    density = cd8_value / vt_value  # ✗ No dimensional checking!
    return density * ureg('cell / millimeter**2')  # ✗ Units reattached at end
```

**Why the golden rule matters:**
- Pint catches dimensional errors automatically (e.g., adding cells to volume)
- Unit conversions happen correctly (.to() method)
- Code is self-documenting (units make intent clear)

**When to extract .magnitude:**
ONLY when absolutely necessary (e.g., passing to numpy functions that don't support Pint).
Immediately reattach units after the operation.

Search the literature and provide an estimate for this calibration target. Strictly prioritize studies matching the model context.
If you cannot find valid sources that pass validation, explain the difficulty in threshold_description.
"""
    return prompt


async def process_single_target(
    row: dict, agent: Agent, target_num: int, total_targets: int
) -> QuickTargetEstimate:
    """Process a single calibration target."""
    target_id = row["calibration_target_id"]
    print(f"\nProcessing {target_num}/{total_targets}: {target_id}")

    prompt = build_prompt_for_target(row)
    result = await agent.run(prompt)
    estimate = result.output

    print(f"  ✓ Found estimate: {estimate.estimate} {estimate.units}")
    print(f"  ✓ Source: {estimate.paper_name[:60]}...")
    print(f"  ✓ DOI: {estimate.doi}")

    return estimate


async def process_quick_estimates_individual(
    input_csv: Path, output_csv: Path, api_key: str
) -> None:
    """Process quick estimates via individual LLM requests in parallel."""
    # Read input CSV
    with open(input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} calibration targets from {input_csv}")
    print(f"Submitting {len(rows)} requests in parallel...\n")

    # Set up agent (reuse for all requests)
    model = OpenAIResponsesModel("gpt-5.1")
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort="high")
    agent = Agent(
        model,
        output_type=QuickTargetEstimate,
        model_settings=settings,
        builtin_tools=[WebSearchTool()],
        retries=10,  # Increased for validation requirements
    )

    # Create tasks for all targets
    tasks = [process_single_target(row, agent, i, len(rows)) for i, row in enumerate(rows, 1)]

    # Process all targets in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful estimates
    estimates = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  ✗ Failed to process {rows[i]['calibration_target_id']}: {result}")
        else:
            estimates.append(result)

    print(f"\n{'='*60}")
    print(f"Completed {len(estimates)}/{len(rows)} targets")
    print(f"{'='*60}\n")

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
        "model_output_code",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(estimate.model_dump())

    print(f"✓ Wrote {len(estimates)} estimates to {output_csv}")


def main():
    """Main entry point for qsp-quick-estimate-individual command."""
    # Load environment variables from .env file
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Quick calibration target estimate extraction - individual processing (CSV -> CSV)"
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
    asyncio.run(process_quick_estimates_individual(args.input_csv, args.output, api_key))


if __name__ == "__main__":
    main()
