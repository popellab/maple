#!/usr/bin/env python3
"""
Unpack batch results to parameter folders.
"""

import json
import re
import sys
import yaml
from pathlib import Path

def extract_yaml_from_content(content):
    """Extract YAML section from content."""
    # Find YAML code block
    yaml_match = re.search(r'```yaml\n(.*?)\n```', content, re.DOTALL)
    if yaml_match:
        return yaml_match.group(1)
    return None

def extract_first_source_tag(yaml_content):
    """Extract the first source tag from YAML content."""
    try:
        data = yaml.safe_load(yaml_content)
        if 'sources' in data and isinstance(data['sources'], dict):
            # Get first source key
            return list(data['sources'].keys())[0]
    except Exception:
        pass
    return None

def get_unique_filename(base_path):
    """Get a unique filename by adding v2, v3, etc. if file exists."""
    if not base_path.exists():
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    counter = 2
    while True:
        new_name = f"{stem}_v{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1

def main():
    if len(sys.argv) != 3:
        print("Usage: unpack_results.py results.jsonl target_project_dir")
        print("       target_project_dir: path to the QSP project directory")
        sys.exit(1)
    
    results_file = sys.argv[1]
    target_project_dir = sys.argv[2]
    target_project_path = Path(target_project_dir).resolve()
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            
            custom_id = result["custom_id"]
            response = result["response"]
            
            if response["status_code"] != 200:
                continue
            
            # GPT-5 response structure: body.output[1].content[0].text
            content = response["body"]["output"][1]["content"][0]["text"]
            
            # Parse custom_id: cancer_type_parameter_name_index
            parts = custom_id.split('_')
            cancer_type = parts[0]
            parameter_name = '_'.join(parts[1:-1])  # Everything except first and last
            
            # Create parameter directory in to-review (in main QSP project)
            param_dir = target_project_path / "to-review" / cancer_type / parameter_name
            param_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract YAML content
            yaml_content = extract_yaml_from_content(content)
            if yaml_content:
                # Get first source tag for filename
                first_source = extract_first_source_tag(yaml_content)
                if first_source:
                    yaml_filename = f"{first_source}.yaml"
                else:
                    yaml_filename = "metadata.yaml"
                
                # Get unique filename
                yaml_path = get_unique_filename(param_dir / yaml_filename)
                
                # Save YAML file
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    f.write(yaml_content)
                
                print(f"Saved: {cancer_type}/{parameter_name}/{yaml_path.name}")
            else:
                print(f"No YAML found: {cancer_type}/{parameter_name}")

if __name__ == "__main__":
    main()