#!/usr/bin/env python3
"""
Unpack batch results to parameter folders.

Extracts JSON from LLM responses, adds header fields from input CSV,
and saves as properly-formatted YAML files.
"""

import argparse
import json
import re
import sys
import yaml
import csv
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional


def file_content_changed(file_path: Path, new_content: str) -> bool:
    """
    Check if new content differs from existing file.

    Args:
        file_path: Path to the file to check
        new_content: New content to compare

    Returns:
        True if file doesn't exist or content has changed, False if identical
    """
    if not file_path.exists():
        return True

    try:
        existing_content = file_path.read_text(encoding='utf-8')
        # Compare SHA256 hashes for efficiency
        existing_hash = hashlib.sha256(existing_content.encode('utf-8')).hexdigest()
        new_hash = hashlib.sha256(new_content.encode('utf-8')).hexdigest()
        return existing_hash != new_hash
    except Exception:
        # If we can't read the file, assume content changed
        return True


def find_matching_derivation(directory: Path, base_filename: str, new_content: str) -> Optional[Path]:
    """
    Find existing derivation file with matching content.

    For files with _deriv### suffixes, checks all derivation files to see if any
    have matching content (for idempotency).

    Args:
        directory: Directory to search in
        base_filename: Filename pattern (e.g., "k_C1_growth_PDAC_04e798b1_deriv001.yaml")
        new_content: Content to compare against

    Returns:
        Path to matching file if found, None otherwise
    """
    # Strip _deriv### suffix if present to get base pattern
    import re
    match = re.match(r'(.+)_deriv\d+\.yaml$', base_filename)
    if not match:
        # Not a derivation file, check the exact filename
        exact_path = directory / base_filename
        if exact_path.exists() and not file_content_changed(exact_path, new_content):
            return exact_path
        return None

    base_pattern = match.group(1)

    # Find all derivation files matching this pattern
    deriv_files = list(directory.glob(f"{base_pattern}_deriv*.yaml"))

    # Check each for matching content
    for deriv_file in deriv_files:
        if not file_content_changed(deriv_file, new_content):
            return deriv_file

    return None


def extract_json_from_content(content):
    """Extract JSON section from content."""
    # First try to find JSON code block
    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find any code block that might be JSON
    code_block_match = re.search(r'```\n(\{.*?\})\n```', content, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)

    # If no code block found, try parsing the entire content as JSON
    # This handles cases where LLM forgot the ```json tags but generated valid JSON
    try:
        # Test if the entire content can be parsed as JSON
        json.loads(content.strip())
        return content.strip()
    except json.JSONDecodeError:
        pass

    return None

def extract_yaml_from_content(content):
    """Extract YAML section from content."""
    # Try to find YAML code block
    yaml_match = re.search(r'```yaml\n(.*?)\n```', content, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1)

    # Try to find generic code block
    code_block_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)

    # If no code block found, assume entire content is YAML
    return content.strip()

def find_parameter_json(data):
    """
    Recursively search for parameter metadata JSON object.

    Looks for an object containing characteristic parameter metadata fields
    like mathematical_role, parameter_estimates, sources, etc.

    Args:
        data: JSON data (dict, list, or primitive)

    Returns:
        Parameter metadata dict if found, None otherwise
    """
    # Required fields that indicate this is parameter metadata
    required_fields = {'parameter_estimates'}
    # At least one source field required
    source_fields = {'sources', 'data_sources'}
    # Common fields that suggest parameter metadata
    common_fields = {'mathematical_role', 'parameter_range', 'study_overview',
                     'technical_details', 'derivation_code_r', 'pooling_weights'}

    if isinstance(data, dict):
        # Check if this object looks like parameter metadata
        keys = set(data.keys())

        # Must have required fields, at least one source field, and at least 3 common fields
        has_sources = len(keys & source_fields) > 0
        if required_fields.issubset(keys) and has_sources and len(keys & common_fields) >= 3:
            return data

        # Otherwise, recursively search in values
        for value in data.values():
            result = find_parameter_json(value)
            if result:
                return result

    elif isinstance(data, list):
        # Search in list items
        for item in data:
            result = find_parameter_json(item)
            if result:
                return result

    return None

def extract_checklist_summary(content):
    """Extract checklist review summary from content (everything before the JSON block)."""
    # Find the JSON code block
    json_match = re.search(r'```json\n.*?\n```', content, re.DOTALL)
    if json_match:
        # Return everything before the JSON block
        return content[:json_match.start()].strip()
    return None

def extract_first_source_tag(content, is_json=True):
    """Extract the first source tag from JSON or YAML content."""
    try:
        if is_json:
            data = json.loads(content)
        else:
            data = yaml.safe_load(content)
        # Check data_sources first (new format), then sources (legacy)
        if 'data_sources' in data and isinstance(data['data_sources'], dict):
            # Get first source key
            return list(data['data_sources'].keys())[0]
        elif 'sources' in data and isinstance(data['sources'], dict):
            # Get first source key
            return list(data['sources'].keys())[0]
    except Exception:
        pass
    return None

def get_unique_filename(base_path, overwrite=False):
    """Get a unique filename by adding v2, v3, etc. if file exists (unless overwrite is True)."""
    if overwrite or not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 2
    while True:
        new_name = f"{stem}_v{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def load_parameter_metadata(input_csv: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load parameter metadata from input CSV.

    Returns dict keyed by (cancer_type, parameter_name) with metadata values.
    """
    metadata = {}

    if not input_csv or not input_csv.exists():
        return metadata

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cancer_type = row['cancer_type']
            parameter_name = row['parameter_name']

            key = (cancer_type, parameter_name)
            metadata[key] = {
                'parameter_name': parameter_name,
                'parameter_units': row.get('parameter_units', ''),
                'parameter_definition': row.get('parameter_description', ''),
                'cancer_type': cancer_type,
                'tags': [],  # Could be populated if available in CSV
                'model_context': row.get('model_context', ''),
                'context_hash': row.get('definition_hash', '')
            }

    return metadata

def load_test_statistic_metadata(input_csv: Path) -> Dict[str, Dict]:
    """
    Load test statistic metadata from input CSV.

    Returns dict keyed by test_statistic_id with metadata values.
    """
    metadata = {}

    if not input_csv or not input_csv.exists():
        return metadata

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_statistic_id = row['test_statistic_id']

            metadata[test_statistic_id] = {
                'test_statistic_id': test_statistic_id,
                'cancer_type': row.get('cancer_type', ''),
                'scenario_context': row.get('scenario_context', ''),
                'required_species': row.get('required_species', ''),
                'derived_species_description': row.get('derived_species_description', ''),
                'tags': ['test-statistic'],
                'context_hash': row.get('context_hash', '')
            }

    return metadata

def prepend_header_fields(json_content: str, header_data: Dict, additional_tags: list = None, schema_header_fields: list = None) -> tuple[str, str]:
    """
    Prepend header fields to LLM-generated JSON content and convert to YAML.

    Args:
        json_content: JSON from LLM response
        header_data: Header fields to prepend
        additional_tags: Optional list of additional tags to add (e.g., ['ai-reviewed'])
        schema_header_fields: List of header field names to add (if None, uses default v1 fields)

    Returns:
        Tuple of (YAML string with header fields, author_year from sources)
    """
    # Parse the LLM JSON
    try:
        llm_data = json.loads(json_content)
    except Exception as e:
        print(f"Warning: Could not parse LLM JSON: {e}")
        return json_content, None

    # Extract author_year from sources before converting to YAML
    author_year = None
    if 'sources' in llm_data and isinstance(llm_data['sources'], dict):
        try:
            author_year = list(llm_data['sources'].keys())[0]
        except:
            pass
    # v2 schema uses data_sources instead of sources
    elif 'data_sources' in llm_data and isinstance(llm_data['data_sources'], dict):
        try:
            author_year = list(llm_data['data_sources'].keys())[0]
        except:
            pass

    # Build complete document with header fields first
    complete_data = {}

    # Add header section comment
    header_comment = "# PARAMETER DEFINITION (from model context)\n"
    header_comment += "# " + "=" * 76 + "\n"

    # Use provided schema fields or default to v1
    if schema_header_fields is None:
        schema_header_fields = [
            'parameter_name', 'parameter_units', 'parameter_definition',
            'cancer_type', 'tags', 'model_context', 'context_hash'
        ]

    # Add header fields in order (only those in schema)
    for field in schema_header_fields:
        if field == 'tags':
            # Special handling for tags
            tags = header_data.get('tags', [])
            if not isinstance(tags, list):
                tags = []
            if 'ai-generated' not in tags:
                tags.append('ai-generated')
            # Add any additional tags
            if additional_tags:
                for tag in additional_tags:
                    if tag not in tags:
                        tags.append(tag)
            complete_data['tags'] = tags
        elif field == 'model_context':
            # Parse and add model_context (it's JSON in the CSV)
            if header_data.get('model_context'):
                try:
                    model_context = json.loads(header_data['model_context'])
                    complete_data['model_context'] = model_context
                except json.JSONDecodeError:
                    complete_data['model_context'] = header_data['model_context']
        elif field in header_data:
            complete_data[field] = header_data[field]

    # Add all LLM-generated fields
    complete_data.update(llm_data)

    # Custom representer for multi-line strings
    def str_representer(dumper, data):
        if '\n' in data:
            # Use literal block style for multi-line strings
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)

    # Convert to YAML string with header comment
    yaml_str = yaml.dump(complete_data,
                        Dumper=yaml.SafeDumper,
                        default_flow_style=False,
                        allow_unicode=True,
                        sort_keys=False,
                        width=1000)

    return header_comment + yaml_str, author_year

def extract_header_fields_from_yaml(yaml_file: Path) -> Optional[Dict]:
    """
    Extract header fields from an existing YAML file.

    Returns dict with header fields, or None if file not found/invalid.
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        header_fields = {}
        field_names = [
            'parameter_name', 'parameter_units', 'parameter_definition',
            'cancer_type', 'tags', 'model_context', 'context_hash',
            'schema_version', 'derivation_id', 'derivation_timestamp'
        ]

        for field in field_names:
            if field in data:
                header_fields[field] = data[field]

        return header_fields if header_fields else None
    except Exception as e:
        print(f"Warning: Could not extract header fields from {yaml_file}: {e}")
        return None

def prepend_header_fields_to_json(json_content: str, header_fields: Dict, schema_header_fields: list) -> str:
    """
    Prepend header fields to JSON content and convert to YAML.

    Args:
        json_content: JSON string with converted content
        header_fields: Dict with header fields to prepend
        schema_header_fields: List of header field names to add

    Returns:
        YAML string with header fields prepended
    """
    try:
        # Parse the JSON content
        content_data = json.loads(json_content)

        # Build complete data with headers first
        complete_data = {}

        # Add header fields in order (only those in schema)
        for field in schema_header_fields:
            if field in header_fields:
                complete_data[field] = header_fields[field]

        # Add converted content
        complete_data.update(content_data)

        # Convert to YAML with header comment
        header_comment = "# PARAMETER DEFINITION (from model context)\n"
        header_comment += "# " + "=" * 76 + "\n"

        # Custom representer for multi-line strings
        def str_representer(dumper, data):
            if '\n' in data:
                # Use literal block style for multi-line strings
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)

        yaml_str = yaml.dump(complete_data,
                           Dumper=yaml.SafeDumper,
                           default_flow_style=False,
                           allow_unicode=True,
                           sort_keys=False,
                           width=1000)

        return header_comment + yaml_str
    except Exception as e:
        print(f"Warning: Could not prepend header fields: {e}")
        # Return as-is converted to YAML
        try:
            content_data = json.loads(json_content)
            return yaml.dump(content_data, Dumper=yaml.SafeDumper,
                           default_flow_style=False, allow_unicode=True,
                           sort_keys=False, width=1000)
        except:
            return json_content

def parse_header_fields_from_template(template_path: Path) -> list:
    """
    Parse header field names from a schema template file.

    Header fields are defined as all fields that come before the first
    "content" field (mathematical_role, parameter_range, study_overview, etc.)

    Args:
        template_path: Path to schema template YAML file

    Returns:
        List of header field names in order
    """
    import yaml

    # Known content fields that mark the end of header section
    content_fields = {
        'mathematical_role', 'parameter_range', 'study_overview',
        'technical_details', 'parameter_estimates', 'derivation_explanation',
        'derivation_code_r', 'pooling_weights', 'key_study_limitations',
        'sources', 'data_sources', 'methodological_sources',
        # Test statistic fields
        'test_statistic', 'test_statistic_definition', 'model_output',
        'expected_distribution', 'validation_weights'
    }

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return []

        # Extract fields in order, stop at first content field
        header_fields = []
        for field in data.keys():
            if field in content_fields:
                break
            header_fields.append(field)

        return header_fields
    except Exception as e:
        print(f"Warning: Could not parse header fields from template: {e}")
        # Fallback to v1 fields
        return [
            'parameter_name', 'parameter_units', 'parameter_definition',
            'cancer_type', 'tags', 'model_context', 'context_hash'
        ]

def get_schema_header_fields(schema_template_path: Path) -> list:
    """
    Get header field names from a schema template.

    Args:
        schema_template_path: Path to schema template file

    Returns:
        List of header field names
    """
    return parse_header_fields_from_template(schema_template_path)

def find_next_derivation_number(target_dir: Path, base_filename: str) -> int:
    """
    Find the next available derivation number for a given base filename.

    Args:
        target_dir: Directory to search for existing files
        base_filename: Base filename (e.g., "k_CD8_pro_MARCHINGO2014_PDAC_a1b2c3d4")

    Returns:
        Next available derivation number (e.g., 1, 2, 3...)
    """
    import re

    # Find all files matching pattern: base_filename_deriv*.yaml
    pattern = f"{base_filename}_deriv*.yaml"
    existing_files = list(target_dir.glob(pattern))

    if not existing_files:
        return 1

    # Extract derivation numbers
    deriv_numbers = []
    for file in existing_files:
        match = re.search(r'_deriv(\d+)\.yaml$', file.name)
        if match:
            deriv_numbers.append(int(match.group(1)))

    return max(deriv_numbers) + 1 if deriv_numbers else 1

def generate_derivation_id(param_name: str, author_year: str, cancer_type: str, context_hash: str, deriv_num: int) -> str:
    """
    Generate derivation ID for v2 schema.

    Format: {param_name}_{author_year}_{cancer_type}_{context_hash}_deriv{num:03d}
    This is the v1 filename base plus deriv{num:03d}

    Args:
        param_name: Parameter name
        author_year: Author and year (e.g., "MARCHINGO2014")
        cancer_type: Cancer type (e.g., "PDAC")
        context_hash: Context hash
        deriv_num: Derivation number

    Returns:
        Derivation ID string
    """
    return f"{param_name}_{author_year}_{cancer_type}_{context_hash}_deriv{deriv_num:03d}"

def main():
    if len(sys.argv) < 3:
        print("Usage: unpack_results.py results.jsonl target_project_dir [input_csv] [source_yaml_dir] [schema_template]")
        print("       results.jsonl: Batch results file")
        print("       target_project_dir: path to the QSP project directory")
        print("       input_csv: (optional) Input CSV with parameter metadata for header fields")
        print("       source_yaml_dir: (optional) Source directory for schema conversion (to extract header fields)")
        print("       schema_template: (optional) Path to schema template YAML (determines header fields and filename format)")
        sys.exit(1)

    results_file = sys.argv[1]
    target_project_dir = sys.argv[2]
    input_csv = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    source_yaml_dir = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    schema_template = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None

    target_project_path = Path(target_project_dir).resolve()

    # Detect batch type and load appropriate metadata
    param_metadata = {}
    test_stat_metadata = {}
    is_test_stat_batch = False

    if input_csv:
        # Check if this is a test statistic batch by examining results file
        try:
            with open(results_file, 'r') as f_check:
                first_lines = [next(f_check) for _ in range(min(10, sum(1 for _ in open(results_file))))]
                if any('test_stat_' in line for line in first_lines):
                    is_test_stat_batch = True
        except StopIteration:
            pass

        # Load appropriate metadata based on batch type
        if is_test_stat_batch:
            test_stat_metadata = load_test_statistic_metadata(input_csv)
        else:
            param_metadata = load_parameter_metadata(input_csv)

    # Parse header fields from schema template
    schema_header_fields = []
    uses_derivation_id = False
    if schema_template and schema_template.exists():
        schema_header_fields = get_schema_header_fields(schema_template)
        uses_derivation_id = 'derivation_id' in schema_header_fields
        print(f"Using schema template: {schema_template.name}")
        print(f"Header fields: {', '.join(schema_header_fields)}")
    else:
        # Default to v1 schema
        schema_header_fields = [
            'parameter_name', 'parameter_units', 'parameter_definition',
            'cancer_type', 'tags', 'model_context', 'context_hash'
        ]
        print(f"Using default v1 schema")

    # Prepare checklist summary file for checklist batches
    checklist_summaries = []

    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            
            custom_id = result["custom_id"]
            response = result["response"]
            
            if response["status_code"] != 200:
                continue

            # GPT-5 response structure: find message in output array (may have multiple reasoning blocks)
            output_items = response["body"]["output"]
            message_item = None
            for item in output_items:
                if item.get("type") == "message":
                    message_item = item
                    break

            if not message_item or "content" not in message_item:
                print(f"Warning: Could not find message content for {custom_id}")
                continue

            content = message_item["content"][0]["text"]
            
            # Parse custom_id and determine directory structure
            parts = custom_id.split('_')

            # Handle different batch types based on custom_id prefix
            use_quick_estimate_format = False  # Initialize flag
            use_test_statistic_format = False  # Initialize flag

            if parts[0] == "defn":
                # Parameter definition: defn_CANCER_TYPE_PARAMETER_NAME_INDEX
                cancer_type = parts[1]
                parameter_name = '_'.join(parts[2:-1])  # Everything between cancer_type and index
                # Put in parameter-definitions directory
                param_dir = target_project_path / "parameter-definitions" / cancer_type / parameter_name
                param_dir.mkdir(parents=True, exist_ok=True)
                filename_default = "definition.yaml"
                use_flat_structure = False
                additional_tags = None
            elif parts[0] == "quick":
                # Quick estimate: quick_CANCER_TYPE_PARAMETER_NAME_INDEX
                cancer_type = parts[1]
                parameter_name = '_'.join(parts[2:-1])  # Everything between cancer_type and index
                # Save directly to target directory with flat structure
                param_dir = target_project_path  # Flat structure at quick_estimates level
                param_dir.mkdir(parents=True, exist_ok=True)
                filename_default = None  # Will be computed from metadata
                use_flat_structure = True
                use_quick_estimate_format = True  # Special flag for quick estimates
                additional_tags = ['quick-estimate']
            elif parts[0] == "checklist" and parts[1] == "json":
                # Checklist review: checklist_json_ORIGINAL_CUSTOM_ID
                # Parse original custom_id to get cancer_type and parameter_name
                original_parts = parts[2:]  # Skip "checklist_json_" prefix
                cancer_type = original_parts[0]
                parameter_name = '_'.join(original_parts[1:-1])  # Everything except first and last
                # Save directly to parameter_estimates with flat structure
                param_dir = target_project_path  # Flat structure at parameter_estimates level
                filename_default = None  # Will be computed from metadata
                use_flat_structure = True
                additional_tags = ['ai-reviewed']
            elif parts[0] == "validate" and parts[1] == "json":
                # JSON validation: validate_json_ORIGINAL_CUSTOM_ID
                # Parse original custom_id to get cancer_type and parameter_name
                original_parts = parts[2:]  # Skip "validate_json_" prefix
                cancer_type = original_parts[0]
                parameter_name = '_'.join(original_parts[1:-1])  # Everything except first and last
                # Save directly to parameter_estimates with flat structure
                param_dir = target_project_path  # Flat structure at parameter_estimates level
                filename_default = None  # Will be computed from metadata
                use_flat_structure = True
                additional_tags = ['ai-validated']
            elif parts[0] == "test" and parts[1] == "stat":
                # Test statistic: test_stat_TEST_STATISTIC_ID_INDEX
                test_statistic_id = '_'.join(parts[2:-1])  # Everything between test_stat_ and index
                # Save directly to test_statistics with flat structure
                param_dir = target_project_path  # Flat structure at test_statistics level
                param_dir.mkdir(parents=True, exist_ok=True)
                filename_default = None  # Will be computed from metadata
                use_flat_structure = True
                use_test_statistic_format = True  # Special flag for test statistics
                additional_tags = ['test-statistic']
                # For unpacking, we'll reuse the parameter_name variable for test_statistic_id
                # Extract cancer_type from metadata
                cancer_type = test_stat_metadata.get(test_statistic_id, {}).get('cancer_type', 'unknown')
                parameter_name = test_statistic_id  # Reuse parameter_name variable
            elif parts[0] == "schema" and parts[1] == "convert":
                # Schema conversion: schema_convert_ORIGINAL_FILENAME
                # Extract original filename (everything after schema_convert_)
                original_file_stem = '_'.join(parts[2:])
                # Save to same directory (will overwrite or add _v2)
                param_dir = target_project_path
                filename_default = f"{original_file_stem}.yaml"
                use_flat_structure = False
                additional_tags = None
                # For schema conversion, we skip cancer_type/parameter_name parsing
                cancer_type = None
                parameter_name = None
            else:
                # Regular parameter extraction: CANCER_TYPE_PARAMETER_NAME_INDEX
                # Save directly to parameter_estimates with flat structure
                cancer_type = parts[0]
                parameter_name = '_'.join(parts[1:-1])  # Everything except first and last
                param_dir = target_project_path  # Flat structure at parameter_estimates level
                filename_default = None  # Will be computed from metadata
                use_flat_structure = True
                additional_tags = None

            # Extract content (JSON for all types, including schema conversion)
            if parts[0] == "schema" and parts[1] == "convert":
                # Schema conversion returns JSON that we convert to YAML
                json_content = extract_json_from_content(content)
                if json_content:
                    try:
                        # Try to get header fields from original YAML file
                        header_fields = None
                        if source_yaml_dir:
                            # Reconstruct original filename from custom_id
                            original_file = source_yaml_dir / f"{original_file_stem}.yaml"
                            if original_file.exists():
                                header_fields = extract_header_fields_from_yaml(original_file)
                            else:
                                print(f"Warning: Could not find original file {original_file}")

                        # Convert JSON to YAML with header fields if available
                        if header_fields:
                            # Parse the JSON to check for derivation_id logic
                            # Handle potential LaTeX escape sequences by using strict=False
                            try:
                                converted_data = json.loads(json_content)
                            except json.JSONDecodeError:
                                # Try fixing common LaTeX escape issues
                                json_content_fixed = json_content.replace('\\', '\\\\')
                                converted_data = json.loads(json_content_fixed)

                            # If using derivation_id schema, generate new derivation_id
                            if uses_derivation_id:
                                # Extract author_year from converted data
                                author_year_sources = None
                                if 'sources' in converted_data and isinstance(converted_data['sources'], dict) and converted_data['sources']:
                                    author_year_sources = list(converted_data['sources'].keys())[0]
                                elif 'data_sources' in converted_data and isinstance(converted_data['data_sources'], dict) and converted_data['data_sources']:
                                    author_year_sources = list(converted_data['data_sources'].keys())[0]

                                if author_year_sources:
                                    # Use parameter_name and other fields from header_fields (more reliable than parsing filename)
                                    param_name_orig = header_fields.get('parameter_name', '')
                                    cancer_type_orig = header_fields.get('cancer_type', '')
                                    definition_hash = header_fields.get('context_hash', '')

                                    if param_name_orig and cancer_type_orig and definition_hash:
                                        # Generate base filename for v1
                                        base_filename = f"{param_name_orig}_{author_year_sources}_{cancer_type_orig}_{definition_hash}"

                                        # Find next derivation number
                                        deriv_num = find_next_derivation_number(param_dir, base_filename)

                                        # Generate derivation_id
                                        from datetime import datetime
                                        derivation_id = generate_derivation_id(param_name_orig, author_year_sources, cancer_type_orig, definition_hash, deriv_num)

                                        # Add derivation tracking fields
                                        header_fields['derivation_id'] = derivation_id
                                        header_fields['derivation_timestamp'] = datetime.now().isoformat()
                                        header_fields['schema_version'] = schema_template.stem if schema_template else 'v2'

                                        # Use derivation_id for filename
                                        filename_default = f"{derivation_id}.yaml"
                                    else:
                                        # Fallback: couldn't extract all needed fields
                                        print(f"Warning: Could not generate derivation_id for {original_file_stem} - missing header fields")
                                else:
                                    # No sources found - use original filename
                                    print(f"Warning: No sources found in converted data for {original_file_stem}")

                            # Use same formatting as parameter extraction (with multiline strings)
                            yaml_str = prepend_header_fields_to_json(json_content, header_fields, schema_header_fields)
                        else:
                            # No header fields - just convert JSON to YAML
                            data = json.loads(json_content)

                            # Custom representer for multi-line strings
                            def str_representer(dumper, data):
                                if '\n' in data:
                                    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
                                return dumper.represent_scalar('tag:yaml.org,2002:str', data)
                            yaml.add_representer(str, str_representer, Dumper=yaml.SafeDumper)

                            yaml_str = yaml.dump(data,
                                               Dumper=yaml.SafeDumper,
                                               default_flow_style=False,
                                               allow_unicode=True,
                                               sort_keys=False,
                                               width=1000)

                        # Save as YAML
                        # Check if any existing derivation file has identical content (idempotent)
                        matching_file = find_matching_derivation(param_dir, filename_default, yaml_str)
                        if matching_file:
                            print(f"Skipped (unchanged): {matching_file.name}")
                        else:
                            # Get unique filename if needed (for new extractions with different content)
                            file_path = param_dir / filename_default
                            if not uses_derivation_id or 'derivation_id' not in (header_fields or {}):
                                file_path = get_unique_filename(file_path)

                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(yaml_str)
                            print(f"Saved: {file_path.name}")
                    except Exception as e:
                        print(f"Error converting schema for {custom_id}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"No JSON found: {custom_id}")
                continue  # Skip regular JSON processing

            # Extract JSON content
            json_content = extract_json_from_content(content)
            if json_content:
                # For checklist batches, find the actual parameter JSON (might be wrapped)
                if parts[0] == "checklist" and parts[1] == "json":
                    try:
                        # Parse the JSON response
                        response_data = json.loads(json_content)

                        # Find the parameter metadata JSON (handles any wrapper structure)
                        param_data = find_parameter_json(response_data)

                        if param_data:
                            # Found parameter data, use it
                            json_content = json.dumps(param_data)

                            # Try to extract review summary from text before JSON or from wrapper
                            summary = extract_checklist_summary(content)
                            if summary:
                                checklist_summaries.append(f"## {custom_id}\n\n{summary}\n\n---\n")
                            else:
                                # Look for summary field in response_data
                                for key, value in response_data.items():
                                    if isinstance(value, str) and ('summary' in key.lower() or 'review' in key.lower()):
                                        checklist_summaries.append(f"## {custom_id}\n\n{value}\n\n---\n")
                                        break
                        else:
                            print(f"Warning: Could not find parameter metadata in checklist response for {custom_id}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse checklist JSON for {custom_id}: {e}")

                # Prepend header fields if metadata available
                key = (cancer_type, parameter_name)
                author_year = None
                header_data = {}
                definition_hash = ''

                # Check if this is a test statistic
                if use_test_statistic_format and parameter_name in test_stat_metadata:
                    # Use test statistic metadata
                    header_data = test_stat_metadata[parameter_name].copy()
                    definition_hash = test_stat_metadata[parameter_name].get('context_hash', '')
                    # Extract author_year from JSON for filename
                    author_year = extract_first_source_tag(json_content, is_json=True)
                    json_content, author_year = prepend_header_fields(json_content, header_data, additional_tags, schema_header_fields)
                elif key in param_metadata:
                    # For schemas with derivation_id, add the derivation tracking fields
                    header_data = param_metadata[key].copy()
                    definition_hash = param_metadata[key].get('context_hash', '')

                    if uses_derivation_id:
                        # Extract author_year first (needed for derivation_id)
                        temp_json = json.loads(json_content)
                        author_year_sources = None
                        if 'sources' in temp_json and isinstance(temp_json['sources'], dict):
                            author_year_sources = list(temp_json['sources'].keys())[0]
                        elif 'data_sources' in temp_json and isinstance(temp_json['data_sources'], dict):
                            author_year_sources = list(temp_json['data_sources'].keys())[0]

                        if author_year_sources:
                            # Generate base filename for v1 (used as derivation_id base)
                            base_filename = f"{parameter_name}_{author_year_sources}_{cancer_type}_{definition_hash}"

                            # Find next derivation number
                            deriv_num = find_next_derivation_number(param_dir, base_filename)

                            # Generate derivation_id
                            derivation_id = generate_derivation_id(parameter_name, author_year_sources, cancer_type, definition_hash, deriv_num)

                            # Add derivation tracking fields to header data
                            from datetime import datetime
                            header_data['derivation_id'] = derivation_id
                            header_data['derivation_timestamp'] = datetime.now().isoformat()
                            header_data['schema_version'] = schema_template.stem if schema_template else 'v2'

                    json_content, author_year = prepend_header_fields(json_content, header_data, additional_tags, schema_header_fields)
                else:
                    # Extract author_year from JSON even without metadata
                    author_year = extract_first_source_tag(json_content, is_json=True)

                # Determine filename based on structure type
                if use_flat_structure:
                    # Special handling for test statistics
                    if use_test_statistic_format and definition_hash:
                        # Test statistic format: {test_statistic_id}_{cancer_type}_{context_hash}.yaml
                        if cancer_type:
                            file_name = f"{parameter_name}_{cancer_type}_{definition_hash}.yaml"
                        else:
                            # Fallback if no cancer_type found
                            file_name = f"{parameter_name}_unknown_{definition_hash}.yaml"
                    # Special handling for quick estimates (no author_year, use deriv number)
                    elif use_quick_estimate_format and definition_hash:
                        # Quick estimate format: {param}_{cancer}_{hash}_deriv001.yaml
                        base_filename = f"{parameter_name}_{cancer_type}_{definition_hash}"
                        deriv_num = find_next_derivation_number(param_dir, base_filename)
                        file_name = f"{base_filename}_deriv{deriv_num:03d}.yaml"
                    # Check if using derivation_id-based filenames (v2 schema)
                    elif uses_derivation_id and 'derivation_id' in header_data:
                        # v2 format: {derivation_id}.yaml
                        file_name = f"{header_data['derivation_id']}.yaml"
                    elif author_year and definition_hash:
                        # v1 format: {param_name}_{author_year}_{cancer_type}_{definition_hash}.yaml
                        file_name = f"{parameter_name}_{author_year}_{cancer_type}_{definition_hash}.yaml"
                    else:
                        # Fallback if missing components
                        file_name = f"{parameter_name}_{cancer_type}_unknown.yaml"
                else:
                    # Legacy format: use first source tag or default
                    first_source = extract_first_source_tag(json_content, is_json=True)
                    if first_source:
                        file_name = f"{first_source}.yaml"
                    else:
                        # Keep yaml extension for legacy batch types (defn, quick)
                        file_name = filename_default

                # Check if any existing derivation file has identical content (idempotent)
                matching_file = find_matching_derivation(param_dir, file_name, json_content)
                if matching_file:
                    if use_flat_structure:
                        print(f"Skipped (unchanged): {matching_file.name}")
                    else:
                        print(f"Skipped (unchanged): {cancer_type}/{parameter_name}/{matching_file.name}")
                else:
                    # Get unique filename if needed (for new extractions with different content)
                    file_path = get_unique_filename(param_dir / file_name)

                    # Save as YAML file (converted from JSON)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(json_content)

                    if use_flat_structure:
                        print(f"Saved: {file_path.name}")
                    else:
                        print(f"Saved: {cancer_type}/{parameter_name}/{file_path.name}")
            else:
                print(f"No JSON found: {cancer_type}/{parameter_name}")

    # Write checklist summaries to markdown file if any were collected
    if checklist_summaries:
        # Get base directory for this repository
        base_dir = Path(results_file).parent.parent
        scratch_dir = base_dir / "scratch"
        scratch_dir.mkdir(exist_ok=True)

        # Create timestamp-based filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checklist_file = scratch_dir / f"checklist_reviews_{timestamp}.md"

        # Write all summaries to file
        with open(checklist_file, 'w', encoding='utf-8') as f:
            f.write("# Parameter Checklist Reviews\n\n")
            f.write(f"Generated from: {Path(results_file).name}\n\n")
            f.write("---\n\n")
            for summary in checklist_summaries:
                f.write(summary)

        print(f"\nChecklist reviews saved to: {checklist_file}")

    # Print next steps based on batch type
    if is_test_stat_batch and input_csv:
        print(f"\n{'='*70}")
        print(f"Next: Aggregate test statistics to create validation CSV")
        print(f"{'='*70}")
        print(f"  python ../qspio-pdac/metadata/aggregate_test_statistics.py \\")
        print(f"    {input_csv} \\")
        print(f"    {target_project_path} \\")
        print(f"    ../qsp-metadata-storage/scratch/")
    elif input_csv and str(input_csv).endswith('extraction_input'):
        # Check if this might be a quick estimate batch
        if 'quick' in str(target_project_path):
            print(f"\n{'='*70}")
            print(f"Next: Aggregate quick estimates to create parameter CSV")
            print(f"{'='*70}")
            print(f"  python ../qspio-pdac/metadata/aggregate_quick_estimates.py \\")
            print(f"    {input_csv} \\")
            print(f"    {target_project_path} \\")
            print(f"    parameters/")

if __name__ == "__main__":
    main()
