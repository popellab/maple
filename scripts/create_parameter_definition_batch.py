#!/usr/bin/env python3
"""
Create batch requests for parameter definition generation using the class-based batch creator system.
"""

import sys
from pathlib import Path

from batch_creator import ParameterDefinitionBatchCreator


def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # Require all arguments - no complex backward compatibility logic
    if len(sys.argv) != 4:
        print("Usage: create_parameter_definition_batch.py input.csv params.csv reactions.csv")
        print("       input.csv: CSV with cancer_type and parameter_name columns")
        print("       params.csv: Parameter definitions (Name, Units, Definition)")
        print("       reactions.csv: Model context (Parameter, Reaction, etc.)")
        sys.exit(1)
    
    input_csv = Path(sys.argv[1])
    params_csv = Path(sys.argv[2])
    reactions_csv = Path(sys.argv[3])
    
    # Validate input files exist
    for file_path in [input_csv, params_csv, reactions_csv]:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    # Create batch creator and process
    creator = ParameterDefinitionBatchCreator(base_dir)
    output_path = creator.run(None, input_csv, params_csv, reactions_csv)  # Use default output path
    
    print(f"Batch file created: {output_path}")


if __name__ == "__main__":
    main()