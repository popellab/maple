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
        completion_window="24h"
    )
    
    print(f"Batch ID: {batch.id}")
    print(f"\nTo monitor this batch, run:")
    print(f"python scripts/batch_monitor.py {batch.id}")

    # Save batch info to batch_jobs directory
    jsonl_path = Path(jsonl_file)
    batch_id_file = jsonl_path.parent / f"{jsonl_path.stem}.batch_id"
    with open(batch_id_file, 'w') as f:
        f.write(batch.id)

if __name__ == "__main__":
    main()
