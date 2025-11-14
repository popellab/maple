#!/usr/bin/env python3
"""Enrich simple parameter CSV with model definitions for LLM extraction."""

import json
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Enrich parameter CSV with model definitions",
        epilog="Example: python scripts/prepare/enrich_parameter_csv.py input.csv defs.json PDAC -o output.csv"
    )
    parser.add_argument('simple_csv', type=Path, help='CSV with parameter_name column')
    parser.add_argument('definitions_json', type=Path, help='Model definitions JSON')
    parser.add_argument('cancer_type', help='Cancer type (e.g., PDAC)')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Output CSV path')
    args = parser.parse_args()

    # Load definitions
    with open(args.definitions_json, 'r') as f:
        definitions = json.load(f)

    # Load and enrich CSV
    df = pd.read_csv(args.simple_csv)

    if 'parameter_name' not in df.columns:
        raise ValueError("Input CSV must have 'parameter_name' column")

    enriched = []
    missing = []

    for _, row in df.iterrows():
        param = row['parameter_name']

        if param not in definitions:
            missing.append(param)
            continue

        info = definitions[param]
        param_def = info['definition']['parameter_definition']

        enriched.append({
            'cancer_type': args.cancer_type.upper(),
            'parameter_name': param,
            'definition_hash': info['hash'],
            'parameter_units': param_def.get('units', ''),
            'parameter_description': param_def.get('description', ''),
            'model_context': json.dumps(param_def.get('model_context', {}), separators=(',', ':'))
        })

    if missing:
        print(f"Warning: {len(missing)} parameters not found in definitions: {', '.join(missing)}")

    if not enriched:
        raise ValueError("No parameters enriched - all missing from definitions")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(enriched).to_csv(args.output, index=False)
    print(f"Enriched {len(enriched)} parameters → {args.output}")


if __name__ == "__main__":
    main()
