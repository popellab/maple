#!/usr/bin/env python3
"""
Convert legacy parameter format to v2 schema.

Legacy format uses:
- parameter_estimates.mu, s2, natural_scale_mean, natural_scale_ci95
- sources (dict)

V2 format uses:
- parameter_estimates.mean, variance, ci95, units
- data_sources (dict)
- Additional required fields: parameter_name, parameter_units, parameter_definition, etc.

Usage:
    python scripts/process/convert_legacy_to_v2.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        data/simbio_parameters.csv \\
        ../qsp-metadata-storage/parameter_estimates_v2 \\
        --dry-run
"""
import argparse
import sys
import os
from pathlib import Path
import yaml
import csv
import re
import hashlib
from datetime import datetime

def extract_parameter_name_from_filename(filename: str) -> str:
    """Extract parameter name from legacy filename."""
    base = filename.replace('.yaml', '')
    # Remove _legacy suffix
    if base.endswith('_legacy'):
        base = base[:-7]
    return base


def load_parameter_definitions(csv_path: str) -> dict:
    """Load parameter definitions from simbio_parameters.csv."""
    params = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('Name')
                if name:
                    params[name] = {
                        'units': row.get('Units', ''),
                        'definition': row.get('Definition', ''),
                        'references': row.get('References', '')
                    }
    except Exception as e:
        print(f"Warning: Could not load parameter definitions: {e}")

    return params


def convert_r_code(legacy_code: str, param_name: str) -> str:
    """
    Convert legacy R code variable names to v2 format.

    Legacy: mu, s2, natural_scale_mean, natural_scale_ci95
    V2: mean_param, variance_param, ci95_param (on natural scale)
    """
    if not legacy_code:
        return None

    # Remove markdown code fences if present
    code = re.sub(r'^```r?\s*\n', '', legacy_code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)

    # Add comment about conversion
    header = f"# Converted from legacy format for parameter: {param_name}\n"
    header += "# Variable names updated to v2 schema\n\n"

    # Replace variable names at the end of the code
    # Legacy code computes: mu, s2, natural_scale_mean, natural_scale_ci95
    # V2 needs: mean_param, variance_param, ci95_param (all on natural scale)

    replacements = [
        (r'\nmu <-', '\nmean_param <-'),  # Use natural scale mean
        (r'\ns2 <-', '\nvariance_param <-'),  # Use natural scale variance
        (r'\nnatural_scale_mean <-', '\n# natural_scale_mean <-'),  # Comment out
        (r'\nnatural_scale_ci95 <-', '\nci95_param <-'),  # Rename to ci95_param
    ]

    for pattern, replacement in replacements:
        code = re.sub(pattern, replacement, code)

    # For legacy code, we actually want natural scale values, not canonical
    # So replace mu with natural_scale_mean in final assignment
    code = re.sub(
        r'# Required summary statistics\nmean_param <-',
        '# Required summary statistics\nmean_param <- mean(mc_draws_natural)  # Use natural scale\nvariance_param <- var(mc_draws_natural)  # Use natural scale\n# Canonical scale values (not used in v2)\nmu_canonical <-',
        code
    )

    return header + code


def extract_units_from_study_overview(study_overview: str) -> str:
    """
    Extract units from legacy study_overview field.
    Format: "Legacy parameter NAME with units UNITS. ..."
    """
    if not study_overview:
        return "UNKNOWN_UNITS"

    # Try to match "with units XXX"
    match = re.search(r'with units\s+([^.]+)', study_overview)
    if match:
        return match.group(1).strip()

    # Check if it's a calculated/derived parameter
    if 'Calculated parameter' in study_overview or 'derived from factors' in study_overview:
        return "DERIVED_UNITS"

    return "UNKNOWN_UNITS"


def convert_legacy_to_v2(legacy_data: dict, param_name: str, param_defs: dict) -> dict:
    """
    Convert legacy parameter format to v2 schema.

    Args:
        legacy_data: Legacy YAML data
        param_name: Parameter name
        param_defs: Parameter definitions from simbio_parameters.csv

    Returns:
        V2 format dictionary
    """
    param_info = param_defs.get(param_name, {})

    # Try to extract units from study_overview
    study_overview = legacy_data.get('study_overview', '')
    units = extract_units_from_study_overview(study_overview)

    # Fallback to CSV if available
    if units == "UNKNOWN_UNITS" and param_info.get('units'):
        units = param_info['units']

    # Extract legacy values
    legacy_estimates = legacy_data.get('parameter_estimates', {})
    natural_scale_mean = legacy_estimates.get('natural_scale_mean')
    natural_scale_ci95 = legacy_estimates.get('natural_scale_ci95', [None, None])

    # Convert R code
    legacy_r_code = legacy_estimates.get('derivation_code_r', '')
    converted_r_code = convert_r_code(legacy_r_code, param_name)

    # Build v2 structure
    v2_data = {
        'schema_version': 'v2',

        # Parameter definition
        'parameter_name': param_name,
        'parameter_units': units,
        'parameter_definition': param_info.get('definition', 'Legacy parameter - definition not available'),
        'cancer_type': 'LEGACY',  # Mark as legacy
        'tags': ['legacy', 'converted'],

        # Derivation identity
        'derivation_id': f"{param_name}_LEGACY_001",
        'derivation_timestamp': datetime.now().isoformat() + 'Z',

        # Model context (minimal for legacy)
        'model_context': {
            'derived_from_context': [
                {
                    'name': param_name,
                    'description': f'Legacy parameter from historical model'
                }
            ],
            'reactions_and_rules': []
        },
        'context_hash': hashlib.md5(f"legacy_{param_name}".encode()).hexdigest()[:8],

        'mathematical_role': 'Legacy parameter - mathematical role not documented',
        'parameter_range': 'positive_reals',  # Default assumption

        # Study information (from legacy)
        'study_overview': study_overview if study_overview else 'Legacy parameter - study details not available',
        'technical_details': legacy_data.get('technical_details', 'Legacy parameter - technical details not available'),

        # Parameter estimates (converted from legacy)
        'parameter_estimates': {
            'mean': natural_scale_mean,
            'variance': None,  # Not available in legacy format
            'ci95': natural_scale_ci95 if isinstance(natural_scale_ci95, list) else [None, None],
            'units': units,
        },

        # Derivation explanation
        'derivation_explanation': 'Legacy parameter converted from historical model format. Original derivation details not available.',

        # R code (converted)
        'derivation_code_r': converted_r_code if converted_r_code else 'Legacy parameter - R code not available',

        # Pooling weights (from legacy)
        'pooling_weights': legacy_data.get('pooling_weights', {}),

        # Study limitations (from legacy)
        'key_study_limitations': legacy_data.get('key_study_limitations', 'Legacy parameter - limitations not documented'),

        # Data sources (convert from legacy 'sources')
        'data_sources': {},

        # Methodological sources
        'methodological_sources': {}
    }

    # Convert sources to data_sources
    legacy_sources = legacy_data.get('sources', {})
    for source_key, source_data in legacy_sources.items():
        v2_data['data_sources'][source_key] = {
            'citation': source_data.get('citation', 'Legacy source - citation not available'),
            'doi': source_data.get('doi_or_url'),
            'data_extracted': [
                {
                    'description': 'Legacy parameter value',
                    'value': natural_scale_mean,
                    'units': units,
                    'figure_or_table': source_data.get('figure_or_table'),
                    'text_snippet': source_data.get('text_snippet', 'Legacy parameter - text not available'),
                    'weight_in_synthesis': 1.0
                }
            ]
        }

    return v2_data


def main():
    parser = argparse.ArgumentParser(description="Convert legacy parameters to v2 schema")
    parser.add_argument(
        "legacy_dir",
        help="Directory with legacy parameter YAML files"
    )
    parser.add_argument(
        "param_defs_csv",
        help="Path to simbio_parameters.csv with parameter definitions"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for v2 format files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversions without writing files"
    )

    args = parser.parse_args()

    # Load parameter definitions
    print(f"Loading parameter definitions from {args.param_defs_csv}...")
    param_defs = load_parameter_definitions(args.param_defs_csv)
    print(f"  Loaded {len(param_defs)} parameter definitions")

    # Create output directory
    if not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process legacy files
    legacy_dir = Path(args.legacy_dir)
    legacy_files = list(legacy_dir.glob('*_legacy.yaml'))

    print(f"\nFound {len(legacy_files)} legacy parameter files")

    converted_count = 0
    error_count = 0
    errors_list = []
    missing_units = []
    missing_mean = []
    missing_sources = []
    derived_units = []

    for legacy_file in legacy_files:
        param_name = extract_parameter_name_from_filename(legacy_file.name)

        try:
            # Load legacy YAML
            try:
                with open(legacy_file, 'r') as f:
                    legacy_data = yaml.safe_load(f)
            except yaml.YAMLError as yaml_err:
                # Try to fix common YAML issues (unquoted colons in strings)
                print(f"  Warning: YAML parsing error in {legacy_file.name}, attempting to fix...")
                with open(legacy_file, 'r') as f:
                    content = f.read()

                # Fix study_overview field - replace with proper quoted multiline string
                # Find study_overview and replace everything until next top-level key
                def fix_study_overview(match):
                    field_name = match.group(1)
                    field_value = match.group(2)
                    # Remove any existing quotes and newlines, collapse to single line
                    cleaned = field_value.strip().replace('\n', ' ').replace('  ', ' ')
                    # Remove any existing quotes
                    cleaned = cleaned.strip('"').strip("'")
                    return f'{field_name} "{cleaned}"'

                # Match study_overview: ... until next non-indented line or end
                content = re.sub(
                    r'^(study_overview:)\s*(.+?)(?=\n\w|\Z)',
                    fix_study_overview,
                    content,
                    flags=re.MULTILINE | re.DOTALL
                )

                legacy_data = yaml.safe_load(content)

            # Convert to v2
            v2_data = convert_legacy_to_v2(legacy_data, param_name, param_defs)

            # Track issues (only count truly unknown, not derived)
            if v2_data['parameter_units'] == 'UNKNOWN_UNITS':
                missing_units.append(param_name)
            elif v2_data['parameter_units'] == 'DERIVED_UNITS':
                # Track derived parameters separately
                derived_units.append(param_name)
            if v2_data['parameter_estimates']['mean'] is None:
                missing_mean.append(param_name)
            if not v2_data['data_sources']:
                missing_sources.append(param_name)

            if args.dry_run:
                print(f"\n{'='*60}")
                print(f"Would convert: {legacy_file.name} -> {param_name}_legacy.yaml")
                print(f"  Parameter: {param_name}")
                print(f"  Units: {v2_data['parameter_units']}")
                print(f"  Mean: {v2_data['parameter_estimates']['mean']}")

                # Show first source citation if available
                if v2_data['data_sources']:
                    first_source = next(iter(v2_data['data_sources'].values()))
                    print(f"  Source: {first_source.get('citation', 'N/A')}")
                else:
                    print(f"  Source: No sources available")
            else:
                # Write v2 YAML (keep _legacy suffix for easy identification)
                output_file = output_dir / f"{param_name}_legacy.yaml"
                with open(output_file, 'w') as f:
                    yaml.dump(v2_data, f, default_flow_style=False, sort_keys=False)

                print(f"✓ Converted: {legacy_file.name} -> {output_file.name}")

            converted_count += 1

        except Exception as e:
            error_msg = f"{legacy_file.name}: {str(e)}"
            print(f"✗ Error converting {error_msg}")
            errors_list.append(error_msg)
            error_count += 1

    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  Converted: {converted_count}")
    print(f"  Errors: {error_count}")

    if derived_units:
        print(f"\n  Derived/calculated parameters: {len(derived_units)}")
        for param in derived_units[:10]:  # Show first 10
            print(f"    - {param}")
        if len(derived_units) > 10:
            print(f"    ... and {len(derived_units) - 10} more")

    if missing_units:
        print(f"\n  Missing units: {len(missing_units)}")
        for param in missing_units[:10]:  # Show first 10
            print(f"    - {param}")
        if len(missing_units) > 10:
            print(f"    ... and {len(missing_units) - 10} more")

    if missing_mean:
        print(f"\n  Missing mean values: {len(missing_mean)}")
        for param in missing_mean[:10]:
            print(f"    - {param}")
        if len(missing_mean) > 10:
            print(f"    ... and {len(missing_mean) - 10} more")

    if missing_sources:
        print(f"\n  Missing sources: {len(missing_sources)}")
        for param in missing_sources[:10]:
            print(f"    - {param}")
        if len(missing_sources) > 10:
            print(f"    ... and {len(missing_sources) - 10} more")

    if errors_list:
        print(f"\n  Errors encountered:")
        for error in errors_list[:10]:
            print(f"    - {error}")
        if len(errors_list) > 10:
            print(f"    ... and {len(errors_list) - 10} more")

    if args.dry_run:
        print(f"\n  DRY RUN - No files written")
        print(f"  Remove --dry-run to write files")
    else:
        print(f"\n  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
