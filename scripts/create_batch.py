#!/usr/bin/env python3
"""
Create batch requests for parameter extraction.
"""

import csv
import json
import sys
from pathlib import Path

# Import functions from generate_prompts.py in same directory
from generate_prompts import (
    load_inputs, index_param_info, render_parameter_to_search,
    build_model_context, fill_template, clean_template
)

def create_prompt(cancer_type, parameter_name, params_df, reactions_df, template_text, param_info):
    """Create a prompt for the given cancer type and parameter."""
    param_row = params_df[params_df["Name"].astype(str) == parameter_name].iloc[0]
    
    name = str(param_row.get("Name", "")).strip()
    units = str(param_row.get("Units", "")).strip()
    definition = str(param_row.get("Definition", "")).strip()
    
    parameter_block = render_parameter_to_search(name, units, definition)
    
    rxns = reactions_df[reactions_df["Parameter"].astype(str) == name]
    model_context_block = build_model_context(name, rxns, param_info)
    model_context_block += f"\n**Target Cancer Type:** {cancer_type}\n"
    
    return fill_template(template_text, parameter_block, model_context_block)

def create_request(custom_id, prompt):
    """Create a batch API request."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-5",
            "input": prompt,
            "reasoning": {"effort": "high"}
        }
    }

def main():
    # Set defaults
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    defaults = {
        'input_csv': base_dir / 'examples' / 'input_format.csv',
        'params_csv': base_dir / 'data' / 'simbio_parameters.csv', 
        'reactions_csv': base_dir / 'data' / 'model_context.csv',
        'template_md': base_dir / 'prompts' / 'qsp_parameter_extraction_prompt.md',
        'output_jsonl': str(base_dir / 'batch_jobs' / 'batch_requests.jsonl')
    }
    
    # Parse arguments with defaults
    if len(sys.argv) == 1:
        # Use all defaults
        input_csv, params_csv, reactions_csv, template_md, output_jsonl = [
            str(defaults['input_csv']), str(defaults['params_csv']), 
            str(defaults['reactions_csv']), str(defaults['template_md']), 
            defaults['output_jsonl']
        ]
    elif len(sys.argv) == 2:
        # Only input CSV provided
        input_csv = sys.argv[1]
        params_csv, reactions_csv, template_md, output_jsonl = [
            str(defaults['params_csv']), str(defaults['reactions_csv']), 
            str(defaults['template_md']), defaults['output_jsonl']
        ]
    elif len(sys.argv) == 6:
        # All arguments provided
        input_csv, params_csv, reactions_csv, template_md, output_jsonl = sys.argv[1:6]
    else:
        print("Usage: create_batch.py [input.csv] [params.csv] [reactions.csv] [template.md] [output.jsonl]")
        print("       create_batch.py input.csv  # uses defaults for other files")
        print("       create_batch.py            # uses all defaults")
        sys.exit(1)
    
    # Load data
    params_df, reactions_df, template_text = load_inputs(
        Path(params_csv), Path(reactions_csv), Path(template_md)
    )
    param_info = index_param_info(params_df)
    cleaned_template = clean_template(template_text)
    
    # Process CSV and create requests
    requests = []
    with open(input_csv, 'r') as f:
        for i, row in enumerate(csv.DictReader(f)):
            cancer_type = row['cancer_type']
            parameter_name = row['parameter_name']
            
            prompt = create_prompt(
                cancer_type, parameter_name, 
                params_df, reactions_df, cleaned_template, param_info
            )
            
            custom_id = f"{cancer_type}_{parameter_name}_{i}"
            request = create_request(custom_id, prompt)
            requests.append(request)
    
    # Create batch_jobs directory if it doesn't exist
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    with open(output_jsonl, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created {len(requests)} requests in {output_jsonl}")

if __name__ == "__main__":
    main()
