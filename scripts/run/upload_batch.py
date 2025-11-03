#!/usr/bin/env python3
"""
Upload batch to OpenAI API.
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
        print("Usage: upload_batch.py requests.jsonl")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    client = OpenAI(api_key=load_api_key())
    
    # Upload file
    with open(jsonl_file, 'rb') as f:
        batch_input_file = client.files.create(file=f, purpose="batch")
    
    # Create batch
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        tools=[{"type": "web_search"}],
        completion_window="24h"
    )
    
    print(f"Batch ID: {batch.id}")

    # Determine batch type from filename
    jsonl_path = Path(jsonl_file)
    batch_type = jsonl_path.stem.replace('_requests', '')

    # Try to find source CSV in batch_jobs/input_data/
    source_csv = None
    input_data_dir = jsonl_path.parent / "input_data"
    if input_data_dir.exists():
        # Look for CSV files - use most recent one as heuristic
        csv_files = sorted(input_data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if csv_files:
            source_csv = str(csv_files[0].resolve())

    # Save batch metadata as JSON
    batch_id_file = jsonl_path.parent / f"{jsonl_path.stem}.batch_id"
    metadata = {
        "batch_id": batch.id,
        "batch_type": batch_type,
        "source_csv": source_csv,
        "jsonl_file": str(jsonl_path.resolve())
    }
    with open(batch_id_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print next command
    print(f"\nNext: Monitor batch progress and download results when complete:")
    print(f"  python scripts/run/batch_monitor.py {batch.id}")

if __name__ == "__main__":
    main()
