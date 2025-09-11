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

import json
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

def load_yaml_safe(filepath: Path) -> Dict[str, Any]:
    """Load YAML file safely with error handling."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return {}

def has_pooling_metadata(yaml_data: Dict[str, Any]) -> bool:
    """Check if YAML already has required pooling metadata."""
    if not isinstance(yaml_data, dict):
        return False
        
    estimates = yaml_data.get("parameter_estimates", {})
    if not isinstance(estimates, dict):
        return False
    
    # Check for required fields  
    required_fields = ["link_function", "link_center", "link_sd", "context_weight"]
    return all(field in estimates and estimates[field] is not None for field in required_fields)

def scan_for_yamls_needing_update(base_dir: Path) -> List[Dict[str, Any]]:
    """Scan to-review directory for YAML files missing pooling metadata."""
    yamls_to_update = []
    
    # Scan cancer-specific directories
    for cancer_dir in ["base", "PDAC", "TNBC", "CRC", "UM", "NSCLC", "HCC"]:
        cancer_path = base_dir / cancer_dir
        if not cancer_path.exists():
            continue
            
        print(f"Scanning {cancer_path}...")
        
        for param_dir in cancer_path.iterdir():
            if not param_dir.is_dir():
                continue
                
            param_name = param_dir.name
            
            # Find YAML files (excluding prior_metadata.yaml)
            yaml_files = [f for f in param_dir.glob("*.yaml") if f.name != "prior_metadata.yaml"]
            
            for yaml_file in yaml_files:
                yaml_data = load_yaml_safe(yaml_file)
                
                if yaml_data and not has_pooling_metadata(yaml_data):
                    yamls_to_update.append({
                        "file_path": str(yaml_file),
                        "relative_path": str(yaml_file.relative_to(base_dir)),
                        "parameter_name": param_name,
                        "cancer_type": cancer_dir,
                        "study_id": yaml_file.stem,
                        "yaml_data": yaml_data
                    })
                    print(f"  Need to update: {yaml_file.name}")
                elif yaml_data and has_pooling_metadata(yaml_data):
                    print(f"  Already has metadata: {yaml_file.name}")
    
    return yamls_to_update

def create_pooling_metadata_prompt(yaml_info: Dict[str, Any]) -> str:
    """Create prompt for LLM to add pooling metadata to YAML."""
    yaml_data = yaml_info["yaml_data"]
    
    # Convert YAML data to string for the prompt
    yaml_str = yaml.dump(yaml_data, default_flow_style=False, width=80)
    
    prompt = f"""You are helping to add statistical metadata to a parameter study YAML file for meta-analysis pooling.

**Current YAML file content:**
```yaml
{yaml_str}
```

**Task:** Add the following required fields to the `parameter_estimates` section:

1. **`link_function`**: The statistical link function used for the parameter. Choose from:
   - "identity" (for parameters on natural scale, like rates, concentrations)  
   - "log" (for positive parameters that vary over orders of magnitude)
   - "logit" (for probabilities/fractions between 0 and 1)

2. **`link_center`**: The center/location parameter on the link scale.
   - For identity link: same as parameter_location_value
   - For log link: ln(parameter_location_value) 
   - For logit link: logit(parameter_location_value)

3. **`link_sd`**: The standard deviation of the parameter on the link scale. 
   - If you have confidence intervals, convert to standard deviation
   - If you have coefficient of variation (CV), calculate: link_sd ≈ CV for log link, complex for others
   - If only point estimates available, estimate reasonable uncertainty (typically 0.1-0.5 on link scale)

4. **`context_weight`**: Study quality/relevance weight between 0-1. Calculate as:
   - context_weight = overall_quality × relevance_to_target
   - Where both overall_quality and relevance_to_target come from the existing quality_metrics section
   - If quality_metrics don't exist, assess and add them, then calculate context_weight

**Guidelines:**
- Analyze the existing `parameter_estimates.data_description` and study context
- Consider the parameter units and typical values when choosing link function
- Be conservative with context_weight (0.4-0.8 typical range)
- If derivation code exists, consider its sophistication for link_sd estimation

**Output:** Return ONLY the complete updated YAML with the new fields added. Preserve all existing structure and content exactly."""

    return prompt

def create_request(custom_id: str, prompt: str) -> Dict[str, Any]:
    """Create a batch API request for pooling metadata update."""
    return {
        "custom_id": custom_id,
        "method": "POST", 
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5",
            "input": prompt,
            "reasoning": {"effort": "medium"}
        }
    }

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
        output_jsonl = base_dir / "batch_jobs" / "pooling_metadata_requests.jsonl"
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
    
    # Scan for YAML files needing updates
    print(f"Scanning {to_review_dir} for YAML files missing pooling metadata...")
    yamls_to_update = scan_for_yamls_needing_update(to_review_dir)
    
    if not yamls_to_update:
        print("No YAML files found that need pooling metadata updates.")
        return
    
    print(f"Found {len(yamls_to_update)} YAML files needing updates")
    
    # Create batch requests
    requests = []
    for i, yaml_info in enumerate(yamls_to_update):
        prompt = create_pooling_metadata_prompt(yaml_info)
        
        custom_id = f"pooling_{yaml_info['cancer_type']}_{yaml_info['parameter_name']}_{yaml_info['study_id']}_{i}"
        request = create_request(custom_id, prompt)
        
        # Add metadata for processing results later
        request["metadata"] = {
            "file_path": yaml_info["file_path"],
            "relative_path": yaml_info["relative_path"],
            "parameter_name": yaml_info["parameter_name"],
            "cancer_type": yaml_info["cancer_type"], 
            "study_id": yaml_info["study_id"]
        }
        
        requests.append(request)
    
    # Create output directory and write requests
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_jsonl, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created {len(requests)} batch requests in {output_jsonl}")
    print("Files to be updated:")
    for yaml_info in yamls_to_update:
        print(f"  - {yaml_info['relative_path']}")

if __name__ == "__main__":
    main()
