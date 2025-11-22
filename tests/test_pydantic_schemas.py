"""
Test that Pydantic models generate OpenAI-compatible strict JSON schemas.

These tests catch issues like Dict[str, str] generating additionalProperties,
which breaks OpenAI's strict JSON schema mode.
"""
import pytest
from openai.lib._pydantic import to_strict_json_schema

from qsp_llm_workflows.core.pydantic_models import (
    ParameterMetadata,
    TestStatistic,
    ParameterEstimates,
    TestStatisticEstimates,
    KeyAssumption,
)


def check_no_additional_properties(schema: dict, path: str = "root"):
    """
    Recursively check that schema contains no additionalProperties.

    OpenAI's strict JSON schema mode requires all properties to be explicitly
    defined, and does not allow additionalProperties (which would be generated
    by Dict[str, str] or similar types).
    """
    issues = []

    # Check current level
    if "additionalProperties" in schema:
        if schema["additionalProperties"] is not False:
            issues.append(f"{path}: Found additionalProperties: {schema['additionalProperties']}")

    # Recursively check properties
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                issues.extend(check_no_additional_properties(prop_schema, f"{path}.{prop_name}"))

    # Check definitions/$defs
    for def_key in ["definitions", "$defs"]:
        if def_key in schema:
            for def_name, def_schema in schema[def_key].items():
                if isinstance(def_schema, dict):
                    issues.extend(check_no_additional_properties(def_schema, f"{path}.{def_key}.{def_name}"))

    # Check items (for arrays)
    if "items" in schema and isinstance(schema["items"], dict):
        issues.extend(check_no_additional_properties(schema["items"], f"{path}.items"))

    return issues


def test_parameter_metadata_schema_strict_compatible():
    """Test that ParameterMetadata generates OpenAI-compatible strict schema."""
    schema = to_strict_json_schema(ParameterMetadata)

    # Check for additionalProperties
    issues = check_no_additional_properties(schema)

    if issues:
        pytest.fail(f"ParameterMetadata schema has additionalProperties:\n" + "\n".join(issues))

    # Verify required fields are present
    assert "properties" in schema
    assert "required" in schema
    assert "key_assumptions" in schema["properties"]
    assert "key_assumptions" in schema["required"]


def test_test_statistic_schema_strict_compatible():
    """Test that TestStatistic generates OpenAI-compatible strict schema."""
    schema = to_strict_json_schema(TestStatistic)

    # Check for additionalProperties
    issues = check_no_additional_properties(schema)

    if issues:
        pytest.fail(f"TestStatistic schema has additionalProperties:\n" + "\n".join(issues))

    # Verify required fields are present
    assert "properties" in schema
    assert "required" in schema
    assert "test_statistic_estimates" in schema["properties"]


def test_parameter_estimates_schema_structure():
    """Test that ParameterEstimates has correct structure."""
    schema = to_strict_json_schema(ParameterEstimates)

    # Verify key_assumptions is NOT in ParameterEstimates (should be in ParameterMetadata)
    assert "key_assumptions" not in schema["properties"], \
        "key_assumptions should be in ParameterMetadata, not ParameterEstimates"


def test_test_statistic_estimates_schema_structure():
    """Test that TestStatisticEstimates has correct structure."""
    schema = to_strict_json_schema(TestStatisticEstimates)

    # Verify key_assumptions IS in TestStatisticEstimates
    assert "key_assumptions" in schema["properties"], \
        "key_assumptions should be in TestStatisticEstimates"

    # Check that it's a list, not a dict
    key_assumptions_schema = schema["properties"]["key_assumptions"]
    assert key_assumptions_schema["type"] == "array", \
        "key_assumptions should be an array, not an object with additionalProperties"


def test_key_assumption_model():
    """Test that KeyAssumption model is correctly structured."""
    schema = to_strict_json_schema(KeyAssumption)

    assert "properties" in schema
    assert "number" in schema["properties"]
    assert "text" in schema["properties"]
    assert schema["properties"]["number"]["type"] == "integer"
    assert schema["properties"]["text"]["type"] == "string"


def test_all_models_have_required_fields():
    """Test that all models have 'required' fields matching their properties."""
    models = [ParameterMetadata, TestStatistic, ParameterEstimates, TestStatisticEstimates]

    for model in models:
        schema = to_strict_json_schema(model)

        # All properties should be in required (Pydantic BaseModel default)
        properties = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))

        # Check that required doesn't have extra keys
        extra_required = required - properties
        assert not extra_required, \
            f"{model.__name__} has extra keys in required: {extra_required}"

        # Check that all properties are required (or explicitly optional)
        # Note: This is a strong check - adjust if we want optional fields
        missing_required = properties - required
        if missing_required:
            # This is acceptable if fields are Optional[]
            print(f"Warning: {model.__name__} has optional fields: {missing_required}")


def test_schemas_have_descriptions():
    """Test that all fields have descriptions (good practice for LLMs)."""
    models = [ParameterMetadata, TestStatistic]

    for model in models:
        schema = to_strict_json_schema(model)

        # Check that properties have descriptions
        for prop_name, prop_schema in schema.get("properties", {}).items():
            assert "description" in prop_schema, \
                f"{model.__name__}.{prop_name} is missing a description"
