#!/usr/bin/env python3
"""
Create batch requests to add pooling-ready metadata fields to study YAML files.

Scans to-review/ directory for YAML files missing pooling metadata fields:
- parameter_estimates.link_function
- parameter_estimates.link_center
- parameter_estimates.link_sd  
- parameter_estimates.context_weight

Creates batch requests to have LLM analyze existing metadata and add missing fields.
"""

import sys
from pathlib import Path

from batch_creator import PoolingMetadataBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # Parse arguments - now requires to_review_dir
    if len(sys.argv) < 2:
        print("Usage: create_pooling_metadata_batch.py to_review_dir [output.jsonl]")
        print("       to_review_dir: path to the QSP project's to-review directory")
        print("       output.jsonl: optional output file (defaults to batch_jobs/pooling_metadata_requests.jsonl)")
        sys.exit(1)
    elif len(sys.argv) == 2:
        to_review_dir = Path(sys.argv[1])
        output_jsonl = None  # Use default
    elif len(sys.argv) == 3:
        to_review_dir = Path(sys.argv[1]) 
        output_jsonl = Path(sys.argv[2])
    else:
        print("Usage: create_pooling_metadata_batch.py to_review_dir [output.jsonl]")
        print("       to_review_dir: path to the QSP project's to-review directory")
        print("       output.jsonl: optional output file (defaults to batch_jobs/pooling_metadata_requests.jsonl)")
        sys.exit(1)
    
    if not to_review_dir.exists():
        print(f"Error: Directory not found: {to_review_dir}")
        sys.exit(1)
    
    # Create batch creator and process
    print(f"Scanning {to_review_dir} for YAML files missing pooling metadata...")
    creator = PoolingMetadataBatchCreator(base_dir)
    creator.run(output_jsonl, to_review_dir)

if __name__ == "__main__":
    main()
