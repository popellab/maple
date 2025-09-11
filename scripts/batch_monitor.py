#!/usr/bin/env python3
"""
Monitor OpenAI batch job and download results.
"""

import sys
from pathlib import Path
from openai import OpenAI

def load_api_key():
    """Load API key from .env file."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise ValueError("OPENAI_API_KEY not found in .env file")

def main():
    if len(sys.argv) < 2:
        print("Usage: monitor_batch.py batch_id [--download]")
        sys.exit(1)
    
    batch_id = sys.argv[1]
    download = "--download" in sys.argv
    
    client = OpenAI(api_key=load_api_key())
    batch = client.batches.retrieve(batch_id)
    
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
    
    if batch.status == "completed" and download:
        if batch.output_file_id:
            content = client.files.content(batch.output_file_id)
            
            # Save to batch_jobs directory  
            script_dir = Path(__file__).parent
            batch_jobs_dir = script_dir.parent / "batch_jobs"
            batch_jobs_dir.mkdir(exist_ok=True)
            
            output_file = batch_jobs_dir / f"{batch_id}_results.jsonl"
            with open(output_file, 'wb') as f:
                f.write(content.content)
            print(f"Downloaded: {output_file}")

if __name__ == "__main__":
    main()