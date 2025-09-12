#!/usr/bin/env python3
"""
Base class for creating batch requests for different types of LLM processing tasks.

This module provides a common framework for generating OpenAI batch API requests
with consistent patterns for request creation, file output, and error handling.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional

from prompt_assembly import PromptAssembler


class BatchCreator(ABC):
    """
    Abstract base class for creating OpenAI batch API requests.
    
    Provides common functionality for request creation, file I/O, and batch processing
    while allowing subclasses to implement specific logic for different prompt types.
    """
    
    def __init__(self, base_dir: Path, prompt_assembler: Optional[PromptAssembler] = None):
        """
        Initialize the batch creator.
        
        Args:
            base_dir: Base directory of the project (used for relative path resolution)
            prompt_assembler: Optional PromptAssembler instance (created if not provided)
        """
        self.base_dir = Path(base_dir)
        self.prompt_assembler = prompt_assembler or PromptAssembler(self.base_dir)
        
    def create_request(self, custom_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Create a standardized batch API request.
        
        Args:
            custom_id: Unique identifier for this request
            prompt: The prompt text to send to the model
            **kwargs: Additional request parameters (model, reasoning effort, etc.)
            
        Returns:
            Dictionary representing a batch API request
        """
        # Set defaults
        model = kwargs.get("model", "gpt-5")
        reasoning_effort = kwargs.get("reasoning_effort", "high")
        
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "input": prompt,
                "reasoning": {"effort": reasoning_effort}
            }
        }
        
        # Add metadata if provided
        if "metadata" in kwargs:
            request["metadata"] = kwargs["metadata"]
            
        return request
    
    def write_batch_file(self, requests: List[Dict], output_path: Path) -> None:
        """
        Write batch requests to JSONL file.
        
        Args:
            requests: List of batch request dictionaries
            output_path: Path where to write the JSONL file
        """
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
                
        print(f"Created {len(requests)} requests in {output_path}")
    
    def get_default_output_path(self) -> Path:
        """
        Get the default output path for this batch creator type.
        
        Returns:
            Default path for batch request files
        """
        return self.base_dir / "batch_jobs" / f"{self.get_batch_type()}_requests.jsonl"
    
    @abstractmethod
    def get_batch_type(self) -> str:
        """
        Get the type identifier for this batch creator.
        
        Used for default file naming and logging.
        
        Returns:
            String identifier for this batch type (e.g., "parameter", "pooling_metadata")
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Process input data and generate batch requests.
        
        This method should be implemented by subclasses to handle their specific
        input processing and prompt generation logic.
        
        Returns:
            List of batch request dictionaries ready for API submission
        """
        pass
    
    def run(self, output_path: Optional[Path] = None, *args, **kwargs) -> Path:
        """
        Execute the complete batch creation process.
        
        Args:
            output_path: Optional output path (uses default if not provided)
            *args, **kwargs: Arguments passed to the process() method
            
        Returns:
            Path to the created batch file
        """
        # Generate requests
        requests = self.process(*args, **kwargs)
        
        # Determine output path
        if output_path is None:
            output_path = self.get_default_output_path()
        else:
            output_path = Path(output_path)
            
        # Write batch file
        self.write_batch_file(requests, output_path)
        
        # Log batch type information
        print(f"Batch type: {self.get_batch_type()}")
        
        return output_path


class ParameterBatchCreator(BatchCreator):
    """
    Creates batch requests for parameter extraction from scientific literature.
    
    Processes CSV input with cancer_type and parameter_name columns, generating
    prompts that combine parameter definitions with model context information.
    """
    
    def get_batch_type(self) -> str:
        return "parameter"
    
    def process(self, input_csv: Path, params_csv: Path, reactions_csv: Path) -> List[Dict[str, Any]]:
        """
        Process parameter extraction inputs and generate batch requests.
        
        Args:
            input_csv: CSV file with cancer_type and parameter_name columns
            params_csv: CSV file with parameter definitions
            reactions_csv: CSV file with reaction/model context
            
        Returns:
            List of batch request dictionaries
        """
        import csv
        from parameter_utils import (
            load_inputs, index_param_info, render_parameter_to_search,
            build_model_context
        )
        
        # Load parameter and reaction data
        params_df, reactions_df, _ = load_inputs(
            Path(params_csv), Path(reactions_csv), None
        )
        param_info = index_param_info(params_df)
        
        # Process CSV and create requests
        requests = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                cancer_type = row['cancer_type']
                parameter_name = row['parameter_name']
                
                # Get parameter info
                param_row = params_df[params_df["Name"].astype(str) == parameter_name].iloc[0]
                name = str(param_row.get("Name", "")).strip()
                units = str(param_row.get("Units", "")).strip()
                definition = str(param_row.get("Definition", "")).strip()
                
                # Build parameter info block
                parameter_block = render_parameter_to_search(name, units, definition)
                
                # Build model context
                rxns = reactions_df[reactions_df["Parameter"].astype(str) == name]
                model_context_block = build_model_context(name, rxns, param_info)
                model_context_block += f"\n**Target Cancer Type:** {cancer_type}\n"
                
                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "PARAMETER_INFO": parameter_block,
                    "MODEL_CONTEXT": model_context_block
                }
                
                # Assemble the prompt
                prompt = self.prompt_assembler.assemble_prompt("parameter_extraction", runtime_data)
                
                # Create batch request
                custom_id = f"{cancer_type}_{parameter_name}_{i}"
                request = self.create_request(custom_id, prompt)
                requests.append(request)
        
        return requests


class PoolingMetadataBatchCreator(BatchCreator):
    """
    Creates batch requests for adding pooling metadata to existing study YAML files.
    
    Scans directory structures for YAML files missing required pooling metadata fields
    and generates prompts to add statistical metadata for meta-analysis.
    """
    
    def get_batch_type(self) -> str:
        return "pooling_metadata"
    
    def process(self, to_review_dir: Path) -> List[Dict[str, Any]]:
        """
        Process YAML files needing pooling metadata updates.
        
        Args:
            to_review_dir: Directory containing study YAML files to process
            
        Returns:
            List of batch request dictionaries
        """
        import yaml
        from typing import Dict, Any
        
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
        
        # Scan for YAML files needing updates
        yamls_to_update = []
        
        # Scan cancer-specific directories
        for cancer_dir in ["base", "PDAC", "TNBC", "CRC", "UM", "NSCLC", "HCC"]:
            cancer_path = to_review_dir / cancer_dir
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
                            "relative_path": str(yaml_file.relative_to(to_review_dir)),
                            "parameter_name": param_name,
                            "cancer_type": cancer_dir,
                            "study_id": yaml_file.stem,
                            "yaml_data": yaml_data
                        })
                        print(f"  Need to update: {yaml_file.name}")
                    elif yaml_data and has_pooling_metadata(yaml_data):
                        print(f"  Already has metadata: {yaml_file.name}")
        
        # Create batch requests
        requests = []
        for i, yaml_info in enumerate(yamls_to_update):
            prompt = create_pooling_metadata_prompt(yaml_info)
            
            custom_id = f"pooling_{yaml_info['cancer_type']}_{yaml_info['parameter_name']}_{yaml_info['study_id']}_{i}"
            request = self.create_request(
                custom_id, 
                prompt, 
                reasoning_effort="medium",
                metadata={
                    "file_path": yaml_info["file_path"],
                    "relative_path": yaml_info["relative_path"],
                    "parameter_name": yaml_info["parameter_name"],
                    "cancer_type": yaml_info["cancer_type"], 
                    "study_id": yaml_info["study_id"]
                }
            )
            
            requests.append(request)
        
        if yamls_to_update:
            print(f"Found {len(yamls_to_update)} YAML files needing updates")
            print("Files to be updated:")
            for yaml_info in yamls_to_update:
                print(f"  - {yaml_info['relative_path']}")
        else:
            print("No YAML files found that need pooling metadata updates.")
        
        return requests