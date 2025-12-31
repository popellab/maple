#!/usr/bin/env python3
"""
Upload requests immediately using the Responses API with concurrent processing.
"""

import asyncio
import json
import sys
from pathlib import Path
from openai import AsyncOpenAI
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic


def load_api_key():
    """Load API key from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise ValueError("OPENAI_API_KEY not found in .env file")


async def process_request(client, request, index, total, pydantic_model):
    """Process a single request asynchronously with structured outputs."""
    print(f"Starting {index}/{total}: {request['custom_id']}")

    # Use responses.parse() with Pydantic model for structured outputs
    response = await client.responses.parse(
        model=request["body"]["model"],
        input=request["body"]["input"],
        reasoning=request["body"]["reasoning"],
        tools=[{"type": "web_search"}],
        text_format=pydantic_model,
    )

    print(f"Completed {index}/{total}: {request['custom_id']}")

    # Format result to match standard output format
    result = {
        "custom_id": request["custom_id"],
        "response": {"status_code": 200, "request_id": response.id, "body": response.model_dump()},
    }
    return result


def detect_pydantic_model_from_request(request):
    """Detect which Pydantic model to use based on request custom_id."""
    custom_id = request["custom_id"]

    # Check if it's a test statistic request
    if custom_id.startswith("test_stat_"):
        return TestStatistic

    # Default to parameter metadata
    return ParameterMetadata


async def main():
    if len(sys.argv) != 2:
        print("Usage: upload_immediate.py requests.jsonl")
        sys.exit(1)

    jsonl_file = Path(sys.argv[1])
    client = AsyncOpenAI(api_key=load_api_key())

    # Read all requests
    requests = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            requests.append(json.loads(line))

    print(f"Processing {len(requests)} requests concurrently...")

    # Create tasks for all requests
    tasks = [
        process_request(
            client, request, i + 1, len(requests), detect_pydantic_model_from_request(request)
        )
        for i, request in enumerate(requests)
    ]

    # Execute all requests concurrently
    results = await asyncio.gather(*tasks)

    # Save results to file
    output_file = jsonl_file.parent / f"{jsonl_file.stem}_immediate_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Results saved to: {output_file}")
    print(f"Completed {len(results)} requests")


if __name__ == "__main__":
    asyncio.run(main())
