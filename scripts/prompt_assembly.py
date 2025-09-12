#!/usr/bin/env python3
"""
Prompt assembly system for QSP LLM workflows.
Handles loading, assembling, and generating prompts from modular components.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class PromptAssembler:
    """Assembles prompts from modular components based on configuration."""
    
    def __init__(self, base_dir: Path):
        """Initialize the prompt assembler with base directory."""
        self.base_dir = Path(base_dir)
        self.config = None
        
    def load_config(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load prompt assembly configuration."""
        if config_path is None:
            config_path = self.base_dir / "templates" / "configs" / "prompt_assembly.yaml"
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        return self.config
        
    def load_template(self, template_path: Path) -> str:
        """Load a template file."""
        with open(self.base_dir / template_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def load_example(self, example_path: Path) -> str:
        """Load an example file."""
        with open(self.base_dir / example_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def format_content(self, content: str, source_config: Dict[str, str]) -> str:
        """Format content based on source configuration."""
        format_type = source_config.get("format", "raw")
        prefix = source_config.get("prefix", "")
        
        if format_type == "yaml_code_block":
            return f"{prefix}```yaml\n{content}\n```"
        elif format_type == "raw":
            return f"{prefix}{content}"
        else:
            return content
            
    def assemble_prompt(self, prompt_type: str, runtime_data: Dict[str, str]) -> str:
        """Assemble a complete prompt from components."""
        if self.config is None:
            self.load_config()
            
        if prompt_type not in self.config["prompt_types"]:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        prompt_config = self.config["prompt_types"][prompt_type]
        
        # Load base prompt
        base_prompt_path = self.base_dir / prompt_config["base_prompt"]
        with open(base_prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
            
        # Process placeholders
        for placeholder_config in prompt_config["placeholders"]:
            placeholder_name = placeholder_config["name"]
            placeholder_tag = f"{{{{{placeholder_name}}}}}"
            source = placeholder_config["source"]
            
            if placeholder_tag not in prompt_text:
                continue  # Skip if placeholder not found
                
            replacement_content = ""
            
            if source == "template_file":
                template_path = prompt_config["template"]
                template_content = self.load_template(template_path)
                source_config = self.config["placeholder_sources"]["template_file"]
                replacement_content = self.format_content(template_content, source_config)
                
            elif source == "example_files":
                examples = prompt_config.get("examples", [])
                example_contents = []
                for example_path in examples:
                    example_content = self.load_example(example_path)
                    example_contents.append(example_content)
                
                combined_examples = "\n\n".join(example_contents)
                source_config = self.config["placeholder_sources"]["example_files"]
                replacement_content = self.format_content(combined_examples, source_config)
                
            elif source == "runtime":
                if placeholder_name in runtime_data:
                    source_config = self.config["placeholder_sources"]["runtime"]
                    replacement_content = self.format_content(runtime_data[placeholder_name], source_config)
                else:
                    replacement_content = f"[{placeholder_name} - TO BE PROVIDED]"
                    
            # Replace placeholder in prompt
            prompt_text = prompt_text.replace(placeholder_tag, replacement_content)
            
        return prompt_text
        
    def get_available_prompt_types(self) -> List[str]:
        """Get list of available prompt types."""
        if self.config is None:
            self.load_config()
        return list(self.config["prompt_types"].keys())
        
    def validate_runtime_data(self, prompt_type: str, runtime_data: Dict[str, str]) -> bool:
        """Validate that required runtime data is provided."""
        if self.config is None:
            self.load_config()
            
        prompt_config = self.config["prompt_types"][prompt_type]
        required_runtime_placeholders = [
            p["name"] for p in prompt_config["placeholders"] 
            if p["source"] == "runtime"
        ]
        
        missing = [key for key in required_runtime_placeholders if key not in runtime_data]
        if missing:
            raise ValueError(f"Missing required runtime data for {prompt_type}: {missing}")
            
        return True
