#!/usr/bin/env python3
"""
Extract and save prompts from batch JSONL files for examination.

This script reads a batch JSONL file and extracts the prompt text from the 
body.input field, saving it to a markdown file in the scratch/ directory.
"""

import sys
import json
from pathlib import Path


def extract_prompt(jsonl_file: Path, request_index: int = 0) -> str:
    """
    Extract prompt from a specific request in the JSONL file.
    
    Args:
        jsonl_file: Path to the JSONL batch file
        request_index: Index of the request to extract (0-based)
        
    Returns:
        The prompt text from body.input
    """
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == request_index:
                request = json.loads(line)
                return request.get('body', {}).get('input', '')
    
    raise IndexError(f"Request index {request_index} not found in file")


def save_prompt_to_file(prompt: str, output_path: Path, custom_id: str = None) -> None:
    """
    Save prompt text to a markdown file.
    
    Args:
        prompt: The prompt text to save
        output_path: Path where to save the file
        custom_id: Optional custom ID to include in filename
    """
    # Create scratch directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if custom_id:
            f.write(f"# Extracted Prompt - {custom_id}\n\n")
        else:
            f.write("# Extracted Prompt\n\n")
        f.write(prompt)
    
    print(f"Prompt saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_prompt.py <jsonl_file> [request_index]")
        print("       jsonl_file: Path to batch JSONL file")
        print("       request_index: Index of request to extract (default: 0)")
        print("\nExamples:")
        print("  extract_prompt.py batch_jobs/parameter_definition_requests.jsonl")
        print("  extract_prompt.py batch_jobs/parameter_requests.jsonl 2")
        sys.exit(1)
    
    jsonl_file = Path(sys.argv[1])
    request_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    if not jsonl_file.exists():
        print(f"Error: File not found: {jsonl_file}")
        sys.exit(1)
    
    try:
        # Extract the prompt
        prompt = extract_prompt(jsonl_file, request_index)
        
        # Get custom_id for filename
        custom_id = None
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == request_index:
                    request = json.loads(line)
                    custom_id = request.get('custom_id', f'request_{request_index}')
                    break
        
        # Determine output filename
        base_name = jsonl_file.stem  # e.g., "parameter_definition_requests"
        if custom_id:
            output_filename = f"{base_name}_{custom_id}.md"
        else:
            output_filename = f"{base_name}_request_{request_index}.md"
        
        # Create output path in scratch directory
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent
        output_path = base_dir / "scratch" / output_filename
        
        # Save the prompt
        save_prompt_to_file(prompt, output_path, custom_id)
        
        print(f"Extracted request #{request_index} from {jsonl_file}")
        print(f"Custom ID: {custom_id}")
        print(f"Prompt length: {len(prompt)} characters")
        
    except IndexError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()