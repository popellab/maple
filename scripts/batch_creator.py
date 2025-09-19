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
    
    def load_parameter_definition(self, cancer_type: str, parameter_name: str, 
                                parameter_storage_dir: Path = None) -> Optional[Dict[str, Any]]:
        """
        Load parameter definition from storage directory.
        
        Args:
            cancer_type: Cancer type for the parameter
            parameter_name: Name of the parameter
            parameter_storage_dir: Path to parameter storage directory
            
        Returns:
            Parameter definition dict or None if not found
        """
        if parameter_storage_dir is None:
            # Default to sibling directory
            parameter_storage_dir = self.base_dir.parent / "qsp-parameter-storage"
            
        definition_path = (parameter_storage_dir / "parameter-definitions" / 
                          cancer_type / parameter_name / "definition.yaml")
        
        if not definition_path.exists():
            print(f"Warning: Parameter definition not found at {definition_path}")
            return None
            
        try:
            import yaml
            with open(definition_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load parameter definition from {definition_path}: {e}")
            return None
    
    def process(self, input_csv: Path, parameter_storage_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Process parameter extraction inputs and generate batch requests using stored parameter definitions.

        Args:
            input_csv: CSV file with cancer_type and parameter_name columns
            parameter_storage_dir: Optional path to parameter storage directory

        Returns:
            List of batch request dictionaries
        """
        import csv
        from parameter_utils import render_parameter_to_search, collect_existing_studies

        # Process CSV and create requests
        requests = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                cancer_type = row['cancer_type']
                parameter_name = row['parameter_name']

                # Load parameter definition (required for extraction)
                param_def = self.load_parameter_definition(cancer_type, parameter_name, parameter_storage_dir)
                if not param_def:
                    print(f"Warning: No parameter definition found for {cancer_type}/{parameter_name}, skipping")
                    continue

                # Extract all needed info from parameter definition
                name = param_def.get("parameter_name", parameter_name)
                units = param_def.get("parameter_units", "")
                definition = param_def.get("parameter_definition", "")
                canonical_scale = param_def.get("canonical_scale", "NOT_PROVIDED")
                mathematical_role = param_def.get("mathematical_role", "")

                # Build parameter info block with cancer type
                parameter_block = render_parameter_to_search(name, units, definition, cancer_type)

                # Use mathematical role as model context
                model_context_block = mathematical_role

                # Collect existing studies for this parameter
                existing_studies = collect_existing_studies(cancer_type, parameter_name, parameter_storage_dir)

                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "EXISTING_STUDIES": existing_studies,
                    "PARAMETER_INFO": parameter_block,
                    "CANONICAL_SCALE": canonical_scale,
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


class QuickEstimateBatchCreator(BatchCreator):
    """
    Creates batch requests for quick parameter estimation from scientific literature.

    Processes CSV input with cancer_type and parameter_name columns, generating
    simplified prompts that ask for ballpark estimates with sources rather than
    comprehensive metadata extraction.
    """

    def get_batch_type(self) -> str:
        return "quick_estimate"

    def load_parameter_definition(self, cancer_type: str, parameter_name: str,
                                parameter_storage_dir: Path = None) -> Optional[Dict[str, Any]]:
        """
        Load parameter definition from storage directory.

        Args:
            cancer_type: Cancer type for the parameter
            parameter_name: Name of the parameter
            parameter_storage_dir: Path to parameter storage directory

        Returns:
            Parameter definition dict or None if not found
        """
        if parameter_storage_dir is None:
            # Default to sibling directory
            parameter_storage_dir = self.base_dir.parent / "qsp-parameter-storage"

        definition_path = (parameter_storage_dir / "parameter-definitions" /
                          cancer_type / parameter_name / "definition.yaml")

        if not definition_path.exists():
            print(f"Warning: Parameter definition not found at {definition_path}")
            return None

        try:
            import yaml
            with open(definition_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load parameter definition from {definition_path}: {e}")
            return None

    def process(self, input_csv: Path, parameter_storage_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Process quick estimation inputs and generate batch requests using stored parameter definitions.

        Args:
            input_csv: CSV file with cancer_type and parameter_name columns
            parameter_storage_dir: Optional path to parameter storage directory

        Returns:
            List of batch request dictionaries
        """
        import csv
        from parameter_utils import render_parameter_to_search, collect_existing_studies

        # Process CSV and create requests
        requests = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                cancer_type = row['cancer_type']
                parameter_name = row['parameter_name']

                # Load parameter definition (required for extraction)
                param_def = self.load_parameter_definition(cancer_type, parameter_name, parameter_storage_dir)
                if not param_def:
                    print(f"Warning: No parameter definition found for {cancer_type}/{parameter_name}, skipping")
                    continue

                # Extract all needed info from parameter definition
                name = param_def.get("parameter_name", parameter_name)
                units = param_def.get("parameter_units", "")
                definition = param_def.get("parameter_definition", "")
                canonical_scale = param_def.get("canonical_scale", "NOT_PROVIDED")
                mathematical_role = param_def.get("mathematical_role", "")

                # Build parameter info block with cancer type
                parameter_block = render_parameter_to_search(name, units, definition, cancer_type)

                # Use mathematical role as model context
                model_context_block = mathematical_role

                # Collect existing studies for this parameter
                existing_studies = collect_existing_studies(cancer_type, parameter_name, parameter_storage_dir)

                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "EXISTING_STUDIES": existing_studies,
                    "PARAMETER_INFO": parameter_block,
                    "CANONICAL_SCALE": canonical_scale,
                    "MODEL_CONTEXT": model_context_block
                }

                # Assemble the prompt using the quick_estimation prompt type
                prompt = self.prompt_assembler.assemble_prompt("quick_estimation", runtime_data)

                # Create batch request
                custom_id = f"quick_{cancer_type}_{parameter_name}_{i}"
                request = self.create_request(custom_id, prompt, reasoning_effort="medium")
                requests.append(request)

        return requests


class ParameterDefinitionBatchCreator(BatchCreator):
    """
    Creates batch requests for parameter definition generation.
    
    Processes parameter/cancer_type combinations from input CSV, generating
    prompts that create standardized parameter definitions with canonical scales.
    """
    
    def get_batch_type(self) -> str:
        return "parameter_definition"
    
    def process(self, input_csv: Path, params_csv: Path, reactions_csv: Path,
                skip_existing: bool = True) -> List[Dict[str, Any]]:
        """
        Process parameter definition inputs and generate batch requests.

        Args:
            input_csv: CSV file with cancer_type and parameter_name columns
            params_csv: CSV file with parameter names/units (simbio_parameters.csv)
            reactions_csv: CSV file with reaction/model context
            skip_existing: If True, skip parameters that already have definitions (default: True)

        Returns:
            List of batch request dictionaries
        """
        import csv
        import pandas as pd
        from parameter_utils import render_parameter_to_search, build_model_context, index_param_info

        # Load simbio parameters for name/units
        simbio_df = pd.read_csv(params_csv)

        # Load parameter definitions
        definitions_path = self.base_dir / "data" / "parameter_definitions.csv"
        if definitions_path.exists():
            definitions_df = pd.read_csv(definitions_path)
        else:
            print(f"Warning: Parameter definitions file not found at {definitions_path}")
            definitions_df = pd.DataFrame(columns=['cancer_type', 'parameter_name', 'definition'])

        # Load reactions for model context
        reactions_df = pd.read_csv(reactions_csv)

        # Create parameter info index for model context building
        # Simple index just with name->units mapping since we don't have full definitions yet
        param_info = {}
        for _, row in simbio_df.iterrows():
            name = str(row.get("Name", "")).strip()
            units = str(row.get("Units", "")).strip()
            param_info[name] = {
                "Units": units,
                "Definition": "",
                "References": ""
            }

        # Check parameter storage directory for existing definitions
        parameter_storage_dir = self.base_dir.parent / "qsp-parameter-storage"

        # Process CSV and create requests
        requests = []
        skipped_count = 0
        with open(input_csv, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                cancer_type = row['cancer_type']
                parameter_name = row['parameter_name']

                # Check if definition already exists
                if skip_existing:
                    definition_path = (parameter_storage_dir / "parameter-definitions" /
                                     cancer_type / parameter_name / "definition.yaml")
                    if definition_path.exists():
                        print(f"Skipping {cancer_type}/{parameter_name} - definition already exists")
                        skipped_count += 1
                        continue

                # Get parameter name/units from simbio_parameters
                simbio_matches = simbio_df[simbio_df["Name"].astype(str) == parameter_name]
                if simbio_matches.empty:
                    print(f"Warning: Parameter '{parameter_name}' not found in simbio_parameters.csv, skipping")
                    continue

                simbio_row = simbio_matches.iloc[0]
                name = str(simbio_row.get("Name", "")).strip()
                units = str(simbio_row.get("Units", "")).strip()

                # Get definition from parameter_definitions.csv
                # First try exact cancer_type match
                def_matches = definitions_df[
                    (definitions_df["cancer_type"].astype(str) == cancer_type) &
                    (definitions_df["parameter_name"].astype(str) == parameter_name)
                ]

                # If no exact match, try any cancer type for this parameter
                if def_matches.empty:
                    def_matches = definitions_df[
                        definitions_df["parameter_name"].astype(str) == parameter_name
                    ]

                if def_matches.empty:
                    # Check if Notes column exists in simbio_parameters and use as fallback
                    notes = str(simbio_row.get("Notes", "")).strip()
                    if notes:
                        print(f"Info: Using Notes from simbio_parameters.csv as definition for '{parameter_name}' (not found in parameter_definitions.csv)")
                        definition = notes
                    else:
                        print(f"Warning: No definition found for parameter '{parameter_name}' in either parameter_definitions.csv or simbio_parameters Notes column, skipping")
                        continue
                else:
                    definition = str(def_matches.iloc[0].get("definition", "")).strip()
                    if len(def_matches) > 1 and definitions_df[
                        (definitions_df["cancer_type"].astype(str) == cancer_type) &
                        (definitions_df["parameter_name"].astype(str) == parameter_name)
                    ].empty:
                        # Used fallback definition from different cancer type
                        fallback_cancer = str(def_matches.iloc[0].get("cancer_type", ""))
                        print(f"Info: Using definition for '{parameter_name}' from {fallback_cancer} (no {cancer_type}-specific definition found)")

                # Build parameter info block with cancer type
                parameter_block = render_parameter_to_search(name, units, definition, cancer_type)

                # Build model context
                rxns = reactions_df[reactions_df["Parameter"].astype(str) == name]
                model_context_block = build_model_context(name, rxns, param_info)

                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "PARAMETER_INFO": parameter_block,
                    "MODEL_CONTEXT": model_context_block
                }

                # Assemble the prompt
                prompt = self.prompt_assembler.assemble_prompt("parameter_definition", runtime_data)

                # Create batch request
                custom_id = f"defn_{cancer_type}_{parameter_name}_{i}"
                request = self.create_request(custom_id, prompt)
                requests.append(request)

        # Print summary
        if skip_existing and skipped_count > 0:
            print(f"Skipped {skipped_count} parameters with existing definitions")
        print(f"Created {len(requests)} parameter definition requests")

        return requests


class ParameterChecklistBatchCreator(BatchCreator):
    """
    Creates batch requests for parameter checklist auditing.

    Combines parameter definitions (for context) with study metadata YAMLs (for auditing)
    to generate checklist prompts for quality assurance of parameter extractions.
    """

    def get_batch_type(self) -> str:
        return "checklist"

    def load_parameter_definition(self, cancer_type: str, parameter_name: str,
                                parameter_storage_dir: Path = None) -> Optional[str]:
        """
        Load parameter definition YAML as text for context.

        Args:
            cancer_type: Cancer type for the parameter
            parameter_name: Name of the parameter
            parameter_storage_dir: Path to parameter storage directory

        Returns:
            Parameter definition YAML as string or None if not found
        """
        if parameter_storage_dir is None:
            # Default to sibling directory
            parameter_storage_dir = self.base_dir.parent / "qsp-parameter-storage"

        definition_path = (parameter_storage_dir / "parameter-definitions" /
                          cancer_type / parameter_name / "definition.yaml")

        if not definition_path.exists():
            print(f"Warning: Parameter definition not found at {definition_path}")
            return None

        try:
            with open(definition_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not load parameter definition from {definition_path}: {e}")
            return None

    def load_study_yaml(self, yaml_path: Path) -> Optional[str]:
        """
        Load study YAML file as text for auditing.

        Args:
            yaml_path: Path to the study YAML file

        Returns:
            YAML content as string or None if error
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not load study YAML from {yaml_path}: {e}")
            return None

    def load_checklist_prompt_template(self) -> str:
        """
        Load the parameter checklist prompt template.

        Returns:
            Checklist prompt template with placeholders
        """
        checklist_path = self.base_dir / "prompts" / "parameter_checklist.md"
        with open(checklist_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_checklist_prompt(self, parameter_definition: str, study_yaml: str) -> str:
        """
        Create checklist prompt by filling placeholders.

        Args:
            parameter_definition: Parameter definition YAML as string
            study_yaml: Study metadata YAML as string

        Returns:
            Complete checklist prompt
        """
        template = self.load_checklist_prompt_template()

        # Fill placeholders
        prompt = template.replace("{{PARAMETER_DEFINITION}}", parameter_definition)
        prompt = prompt.replace("{{DOCUMENTATION}}", study_yaml)

        return prompt

    def process(self, input_csv: Path, parameter_storage_dir: Path = None,
                to_review_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Process checklist inputs and generate batch requests.

        Args:
            input_csv: CSV file with cancer_type and parameter_name columns
            parameter_storage_dir: Optional path to parameter storage directory
            to_review_dir: Optional path to to-review directory

        Returns:
            List of batch request dictionaries
        """
        import csv

        # Set default directories
        if parameter_storage_dir is None:
            parameter_storage_dir = self.base_dir.parent / "qsp-parameter-storage"
        if to_review_dir is None:
            # Use parameter storage to-review first (standard workflow), then local as fallback
            param_storage_to_review = parameter_storage_dir / "to-review"
            local_to_review = self.base_dir / "to-review"
            if param_storage_to_review.exists():
                to_review_dir = param_storage_to_review
            elif local_to_review.exists():
                to_review_dir = local_to_review
            else:
                # Default to parameter storage path even if it doesn't exist (will give clear error)
                to_review_dir = param_storage_to_review

        requests = []

        with open(input_csv, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                cancer_type = row['cancer_type']
                parameter_name = row['parameter_name']

                # Load parameter definition for context
                param_definition = self.load_parameter_definition(
                    cancer_type, parameter_name, parameter_storage_dir
                )
                if not param_definition:
                    print(f"Warning: No parameter definition found for {cancer_type}/{parameter_name}, skipping")
                    continue

                # Find study YAML files to audit
                study_dir = to_review_dir / cancer_type / parameter_name
                if not study_dir.exists():
                    print(f"Warning: Study directory not found: {study_dir}, skipping")
                    continue

                # Find YAML files (excluding prior_metadata.yaml)
                yaml_files = [f for f in study_dir.glob("*.yaml")
                             if f.name != "prior_metadata.yaml"]

                if not yaml_files:
                    print(f"Warning: No study YAML files found in {study_dir}, skipping")
                    continue

                # Create request for each study YAML
                for j, yaml_file in enumerate(yaml_files):
                    study_yaml = self.load_study_yaml(yaml_file)
                    if not study_yaml:
                        continue

                    # Create checklist prompt
                    prompt = self.create_checklist_prompt(param_definition, study_yaml)

                    # Create batch request
                    study_id = yaml_file.stem
                    custom_id = f"checklist_{cancer_type}_{parameter_name}_{study_id}_{i}_{j}"

                    request = self.create_request(
                        custom_id,
                        prompt,
                        reasoning_effort="high",
                        metadata={
                            "cancer_type": cancer_type,
                            "parameter_name": parameter_name,
                            "study_id": study_id,
                            "yaml_file": str(yaml_file),
                            "relative_path": str(yaml_file.relative_to(to_review_dir))
                        }
                    )

                    requests.append(request)
                    print(f"  Created checklist request for {cancer_type}/{parameter_name}/{study_id}")

        return requests


class ConstraintValidationBatchCreator(BatchCreator):
    """
    Creates batch requests for constraint validation test generation from biological expectations.

    Processes CSV input with constraint descriptions or biological expectations, generating
    prompts that create MATLAB unit test-style constraint validation definitions.
    """

    def get_batch_type(self) -> str:
        return "constraint_validation"

    def process(self, input_csv: Path, model_context_csv: Path = None) -> List[Dict[str, Any]]:
        """
        Process constraint validation inputs and generate batch requests.

        Args:
            input_csv: CSV file with constraint_id and constraint_description columns
            model_context_csv: Optional CSV file with model structure information

        Returns:
            List of batch request dictionaries
        """
        import csv
        import pandas as pd

        # Load model context if provided
        model_context_info = {}
        if model_context_csv and Path(model_context_csv).exists():
            model_df = pd.read_csv(model_context_csv)
            # Create a lookup for model variables and their descriptions
            for _, row in model_df.iterrows():
                var_name = str(row.get("Variable", "")).strip()
                if var_name:
                    model_context_info[var_name] = {
                        "description": str(row.get("Description", "")).strip(),
                        "units": str(row.get("Units", "")).strip(),
                        "compartment": str(row.get("Compartment", "")).strip()
                    }

        # Process CSV and create requests
        requests = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.DictReader(f)):
                constraint_id = row.get('constraint_id', f'constraint_{i}')
                constraint_description = row.get('constraint_description', '')

                if not constraint_description.strip():
                    print(f"Warning: Empty constraint description for {constraint_id}, skipping")
                    continue

                # Build constraint info block
                constraint_info = f"**Constraint ID:** {constraint_id}\n\n"
                constraint_info += f"**Biological Expectation:**\n{constraint_description}"

                # Add cancer type and parameter context if available
                cancer_type = row.get('cancer_type', '')
                parameter_context = row.get('parameter_context', '')
                if cancer_type:
                    constraint_info += f"\n\n**Cancer Type Context:** {cancer_type}"
                if parameter_context:
                    constraint_info += f"\n\n**Parameter Context:** {parameter_context}"

                # Build model context block
                model_context_block = "**Model Structure Information:**\n\n"
                if model_context_info:
                    model_context_block += "Available model variables and their descriptions:\n\n"
                    for var_name, info in model_context_info.items():
                        model_context_block += f"- `{var_name}`: {info['description']}"
                        if info['units']:
                            model_context_block += f" (units: {info['units']})"
                        if info['compartment']:
                            model_context_block += f" [compartment: {info['compartment']}]"
                        model_context_block += "\n"
                else:
                    model_context_block += "No specific model context provided. Use standard SimBiology variable naming conventions."

                # Collect existing constraints (placeholder for now)
                existing_constraints = "No existing constraint validation tests provided for comparison."

                # Prepare runtime data for prompt assembly
                runtime_data = {
                    "EXISTING_CONSTRAINTS": existing_constraints,
                    "CONSTRAINT_INFO": constraint_info,
                    "MODEL_CONTEXT": model_context_block
                }

                # Assemble the prompt
                prompt = self.prompt_assembler.assemble_prompt("constraint_validation", runtime_data)

                # Create batch request
                custom_id = f"constraint_{constraint_id}_{i}"
                request = self.create_request(custom_id, prompt, reasoning_effort="high")
                requests.append(request)

        return requests