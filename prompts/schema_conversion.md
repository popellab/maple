# Goal

You are a data migration assistant. Your task is to convert parameter metadata to the latest schema version, preserving all data while adapting to the new structure.

You will receive:
- Target schema template (JSON format) - the structure to match
- Current data to convert (JSON format) - contains old schema implicitly
- Migration instructions - guidance on the conversion

You must return:
- Converted data (JSON format) matching the target schema structure

# Instructions

Convert the provided JSON data to match the target schema following these guidelines:

## Data Preservation
1. **Preserve all information**: Every piece of data must be preserved in the converted output
2. **Map fields accurately**: Infer the old schema structure from the data and map to new schema fields
3. **Restructure as needed**: Adapt nested structures, field names, and organization to match the target schema
4. **Maintain data types**: Keep numeric values as numbers, strings as strings, etc.

## Schema Adaptation Rules
1. **Follow target schema exactly**: The output must conform precisely to the target schema structure
2. **Handle renamed fields**: Map old field names to new field names (e.g., `expected_distribution` → `test_statistic_estimates`)
3. **Handle moved fields**: Data may need to move between sections (e.g., flat to nested structure)
4. **Handle new required fields**: If the target schema requires fields not in current data, use sensible defaults or derive from existing data
5. **Handle deprecated fields**: If current data has fields not in target schema, preserve them in appropriate sections or note their removal

## Quality Checks
1. **Completeness**: Verify all current data is represented in the converted output
2. **Validity**: Ensure the output is valid JSON and matches the target schema structure
3. **Semantic equivalence**: The meaning and scientific content should be identical

# Target Schema (Template)

Note: Header fields are excluded from the template for clarity. These fields (parameter_name, parameter_units, parameter_definition, cancer_type, tags, model_context, context_hash, schema_version, derivation_id, derivation_timestamp) are metadata and will be preserved separately.

Focus on transforming the content fields to match this structure:

```json
{{NEW_SCHEMA}}
```

# Migration Instructions

{{MIGRATION_NOTES}}

# Current Data

The following is the data to convert, shown in JSON format. The current schema structure is implicit in this data - infer it and adapt to the target schema:

```json
{{JSON_CONTENT}}
```

# Output Format

Return ONLY the converted content in a ```json code block. Do not include explanatory text before or after the JSON.

Requirements:
- Wrap your response in ```json code block tags
- Output must be valid JSON
- Output must conform to the new schema structure
- All data from the old schema must be preserved
- Use proper JSON formatting (proper escaping, quoted strings, etc.)
- Use `\n` for line breaks in multi-line strings
- Numeric values should be actual numbers, not strings
