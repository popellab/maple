#!/usr/bin/env python3
"""Enrich partial test statistic CSV with model and scenario context."""

import argparse
import hashlib
from pathlib import Path
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Enrich test statistic CSV with model and scenario context",
        epilog="Example: python scripts/prepare/enrich_test_statistic_csv.py partial.csv model.txt scenario.yaml -o output.csv",
    )
    parser.add_argument(
        "partial_csv",
        type=Path,
        help="CSV with test_statistic_id, required_species, derived_species_description",
    )
    parser.add_argument("model_context_file", type=Path, help="Text file with model context")
    parser.add_argument(
        "scenario_yaml", type=Path, help="Scenario YAML with scenario_context and indication"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    # Load context
    model_context = args.model_context_file.read_text(encoding="utf-8").strip()

    with open(args.scenario_yaml, "r") as f:
        scenario = yaml.safe_load(f)

    scenario_context = scenario["scenario_context"].strip()
    cancer_type = scenario.get("indication", "unknown")
    context_hash = hashlib.md5(f"{model_context}_{scenario_context}".encode()).hexdigest()[:8]

    # Load and enrich CSV
    df = pd.read_csv(args.partial_csv)

    required_cols = ["test_statistic_id", "required_species", "derived_species_description"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add enriched columns
    df["cancer_type"] = cancer_type
    df["model_context"] = model_context
    df["scenario_context"] = scenario_context
    df["context_hash"] = context_hash

    # Reorder
    df = df[
        [
            "test_statistic_id",
            "cancer_type",
            "model_context",
            "scenario_context",
            "required_species",
            "derived_species_description",
            "context_hash",
        ]
    ]

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Enriched {len(df)} test statistics → {args.output}")


if __name__ == "__main__":
    main()
