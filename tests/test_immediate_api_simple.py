#!/usr/bin/env python3
"""
Simple, fast test for immediate mode API contract.

Tests that the API call works correctly with minimal complexity.
"""

import asyncio
import json
import os
from pathlib import Path
from openai import AsyncOpenAI
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata


async def test_simple_api_call():
    """Test API call with simple prompt and low reasoning effort."""
    # Load API key from .env file first (override environment variable)
    api_key = None
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False

    print(f"Using API key: {api_key[:20]}...")

    client = AsyncOpenAI(api_key=api_key)

    # Very simple test prompt
    simple_prompt = """
Extract parameter metadata from this text:

"The cancer cell growth rate k_C_growth was measured at 0.15 per day in the PDAC model."

Return a complete ParameterMetadata structure with:
- parameter_name: k_C_growth
- cancer_type: PDAC
- parameter_value: use value_type "point_estimate" with value 0.15
- units: 1/day
- Fill in other required fields with reasonable defaults
- For primary_data_sources, use: title="Test Study", first_author="Smith", year=2023, doi="10.1234/test"
    """

    try:
        print("Testing API call with ParameterMetadata model...")
        print("  Model: gpt-5")
        print("  Reasoning: low")
        print("  Web search: disabled")

        response = await client.responses.parse(
            model="gpt-5",
            input=simple_prompt,
            reasoning={"effort": "low"},  # Fast!
            text_format=ParameterMetadata,
        )

        print(f"\n✓ API call successful")
        print(f"  Response ID: {response.id}")

        # Test using output_parsed (the standard pattern)
        if hasattr(response, "output_parsed"):
            parsed = response.output_parsed
            print(f"\n✓ output_parsed available")
            print(f"  Type: {type(parsed).__name__}")

            # Test serialization
            parsed_dict = parsed.model_dump()
            json_str = json.dumps(parsed_dict, indent=2)
            print(f"\n✓ Serialization successful")
            print(f"  JSON length: {len(json_str)} bytes")

            # Show a few fields
            if "mathematical_role" in parsed_dict:
                print(f"  Sample field (mathematical_role): {parsed_dict['mathematical_role'][:100]}...")
            if "parameter_estimates" in parsed_dict:
                print(f"  Parameter estimates count: {len(parsed_dict['parameter_estimates'])}")

            print(f"\n✅ All tests passed!")
            return True
        else:
            print("❌ No output_parsed attribute found")
            return False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_simple_api_call())
    exit(0 if result else 1)
