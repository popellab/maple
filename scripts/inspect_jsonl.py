#!/usr/bin/env python3
"""
Utility to pretty print specific lines from JSONL files for debugging.
"""

import json
import sys
from pathlib import Path

def pretty_print_jsonl_line(jsonl_file: Path, line_number: int = 1):
    """Pretty print a specific line from a JSONL file."""
    if not jsonl_file.exists():
        print(f"Error: File not found: {jsonl_file}")
        return
    
    try:
        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f, 1):
                if i == line_number:
                    try:
                        data = json.loads(line.strip())
                        print(f"=== Line {line_number} of {jsonl_file.name} ===")
                        print(json.dumps(data, indent=2))
                        
                        # If it's a batch request, also show the prompt excerpt
                        if "body" in data and "input" in data["body"]:
                            prompt = data["body"]["input"]
                            print(f"\n=== Prompt Preview (first 500 chars) ===")
                            print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
                            
                            # Look for YAML content specifically
                            if "```yaml" in prompt:
                                yaml_start = prompt.find("```yaml") + 7
                                yaml_end = prompt.find("```", yaml_start)
                                if yaml_end > yaml_start:
                                    yaml_content = prompt[yaml_start:yaml_end]
                                    print(f"\n=== YAML Content ===")
                                    print(yaml_content[:1000] + ("..." if len(yaml_content) > 1000 else ""))
                        
                        return data
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_number}: {e}")
                        print(f"Raw line: {line}")
                        return None
        
        print(f"Error: Line {line_number} not found in {jsonl_file}")
        return None
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: inspect_jsonl.py <file.jsonl> [line_number]")
        print("       line_number defaults to 1")
        sys.exit(1)
    
    jsonl_file = Path(sys.argv[1])
    line_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    pretty_print_jsonl_line(jsonl_file, line_number)

if __name__ == "__main__":
    main()