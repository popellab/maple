#!/usr/bin/env python3
"""
Monitor OpenAI batch job and download results.
"""

import json
import sys
from pathlib import Path
from openai import OpenAI

def load_api_key():
    """Load API key from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise ValueError("OPENAI_API_KEY not found in .env file")

def main():
    if len(sys.argv) != 2:
        print("Usage: batch_monitor.py batch_id")
        sys.exit(1)

    batch_id = sys.argv[1]

    # Try to load batch metadata from .batch_id file
    script_dir = Path(__file__).parent
    batch_jobs_dir = script_dir.parent.parent / "batch_jobs"  # Project root / batch_jobs

    batch_metadata = None
    for batch_id_file in batch_jobs_dir.glob("*.batch_id"):
        try:
            with open(batch_id_file, 'r') as f:
                content = f.read().strip()
                # Check if it's JSON or plain batch_id
                if content.startswith('{'):
                    metadata = json.loads(content)
                    if metadata.get("batch_id") == batch_id:
                        batch_metadata = metadata
                        break
                elif content == batch_id:
                    # Old format - just batch_id as string
                    batch_metadata = {
                        "batch_id": batch_id,
                        "batch_type": batch_id_file.stem.replace('_requests', ''),
                        "source_csv": None
                    }
                    break
        except Exception:
            continue

    client = OpenAI(api_key=load_api_key())
    batch = client.batches.retrieve(batch_id)

    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"Completed: {batch.request_counts.completed}/{batch.request_counts.total}")

    if batch.status == "completed":
        if batch.output_file_id:
            content = client.files.content(batch.output_file_id)

            # Save to batch_jobs directory
            batch_jobs_dir.mkdir(exist_ok=True)

            output_file = batch_jobs_dir / f"{batch_id}_results.jsonl"
            with open(output_file, 'wb') as f:
                f.write(content.content)
            print(f"Downloaded: {output_file}")

            # Generate unpacking command based on batch type
            if batch_metadata:
                batch_type = batch_metadata.get("batch_type", "parameter")
                source_csv = batch_metadata.get("source_csv", "input_csv")

                # Determine target directory based on batch type
                if batch_type == "test_stat":
                    target_dir = "../qsp-metadata-storage/test_statistics"
                    template = "templates/test_statistic_template.yaml"
                elif batch_type.startswith("checklist") or batch_type.startswith("validate"):
                    target_dir = "../qsp-metadata-storage/parameter_estimates"
                    template = "templates/parameter_metadata_template.yaml"
                else:
                    target_dir = "../qsp-metadata-storage/parameter_estimates"
                    template = "templates/parameter_metadata_template.yaml"

                # Print next command
                print(f"\nNext: Unpack results to {target_dir.split('/')[-1]}:")
                if batch_type == "test_stat":
                    # Test statistics need template for header fields
                    if source_csv:
                        print(f"  python scripts/process/unpack_results.py {output_file} {target_dir} {source_csv} \"\" {template}")
                        print("\nThen aggregate test statistics:")
                        print(f"  python ../qspio-pdac/metadata/aggregate_test_statistics.py {source_csv} {target_dir} ../qsp-metadata-storage/scratch/")
                    else:
                        print(f"  python scripts/process/unpack_results.py {output_file} {target_dir} input_csv \"\" {template}")
                        print("\nNote: Replace 'input_csv' with path to CSV used to create this batch")
                        print("\nThen aggregate test statistics:")
                        print(f"  python ../qspio-pdac/metadata/aggregate_test_statistics.py input_csv {target_dir} ../qsp-metadata-storage/scratch/")
                else:
                    # Other batch types need template for header fields
                    if source_csv:
                        print(f"  python scripts/process/unpack_results.py {output_file} {target_dir} {source_csv} \"\" {template}")
                    else:
                        print(f"  python scripts/process/unpack_results.py {output_file} {target_dir} input_csv \"\" {template}")
                        print("\nNote: Replace 'input_csv' with path to CSV used to create this batch")
            else:
                # Fallback if no metadata found
                print("\nNext: Unpack results to parameter_estimates:")
                print(f"  python scripts/process/unpack_results.py {output_file} ../qsp-metadata-storage/parameter_estimates input_csv")
                print("\nNote: input_csv is required for header fields (parameter_name, units, definition, etc.)")

if __name__ == "__main__":
    main()
