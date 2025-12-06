#!/usr/bin/env python3
"""
Create batch requests to fix validation errors.

Loads validation reports, groups errors by file, and creates fix requests
using the validation_fix_prompt template.

Headers are stripped before sending to LLM and will be preserved from original file.
"""

import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List

from qsp_llm_workflows.core.batch_creator import BatchCreator
from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic
from qsp_llm_workflows.core.prompts import build_validation_fix_prompt
from qsp_llm_workflows.core.header_utils import HeaderManager

logger = logging.getLogger(__name__)


class ValidationFixBatchCreator(BatchCreator):
    """
    Create batch requests to fix files with validation errors.
    """

    def __init__(
        self,
        data_dir: str,
        validation_results_dir: str,
        output_file: str,
        model_class,
        **kwargs,
    ):
        """
        Initialize validation fix batch creator.

        Args:
            data_dir: Directory containing YAML files to fix
            validation_results_dir: Directory with validation JSON reports
            output_file: Path to output batch JSONL file
            model_class: Pydantic model class (ParameterMetadata or TestStatistic)
            **kwargs: Additional config options
        """
        super().__init__(Path(data_dir).parent)  # BatchCreator only takes base_dir
        self.data_dir = Path(data_dir)
        self.validation_results_dir = Path(validation_results_dir)
        self.output_file = Path(output_file)
        self.model_class = model_class

        # Initialize header manager
        self.header_manager = HeaderManager()

        # Determine workflow type from model class
        if model_class == ParameterMetadata:
            self.workflow_type = "parameter"
        elif model_class == TestStatistic:
            self.workflow_type = "test_statistic"
        else:
            raise ValueError(f"Unknown model class: {model_class}")

    def load_validation_reports(self) -> Dict[str, List[str]]:
        """
        Load validation reports and group errors by filename.

        Returns:
            Dict mapping filename to list of error messages
        """
        errors_by_file = {}

        # Load all individual validation report files
        for report_file in self.validation_results_dir.glob("*.json"):
            # Skip master summary
            if report_file.name == "master_validation_summary.json":
                continue

            try:
                with open(report_file) as f:
                    report = json.load(f)

                # Extract failed items
                for failure in report.get("failed", []):
                    filename = failure["item"]
                    reason = failure["reason"]

                    # Extract just the filename if it includes path or extra info
                    if "/" in filename:
                        filename = filename.split("/")[-1]

                    # Group by file
                    if filename not in errors_by_file:
                        errors_by_file[filename] = []

                    # Format error with validator name
                    validator_name = report["summary"]["name"]
                    error_msg = f"[{validator_name}] {reason}"
                    errors_by_file[filename].append(error_msg)

            except Exception as e:
                logger.warning(f"Could not load validation report {report_file}: {e}")

        return errors_by_file

    def get_batch_type(self) -> str:
        """Get batch type identifier for file naming."""
        return f"validation_fix_{self.workflow_type}"

    def process(self) -> List[Dict]:
        """
        Process validation reports and generate fix requests.

        Returns:
            List of batch request dictionaries
        """
        return self.create_batch_requests()

    def create_batch_requests(self) -> List[Dict]:
        """
        Create batch requests for files with validation errors.

        Returns:
            List of batch request dictionaries
        """
        # Load validation errors
        errors_by_file = self.load_validation_reports()

        if not errors_by_file:
            logger.info("No validation errors found - nothing to fix")
            return []

        logger.info(f"Found validation errors in {len(errors_by_file)} files")

        requests = []
        yaml_files = list(self.data_dir.glob("*.yaml"))

        for yaml_file in yaml_files:
            filename = yaml_file.name

            # Skip files without errors
            if filename not in errors_by_file:
                continue

            # Strip headers from YAML (LLM should only see content fields)
            try:
                headers, content = self.header_manager.strip_headers_from_yaml(
                    yaml_file, self.model_class
                )
            except Exception as e:
                logger.warning(f"Could not parse {filename}: {e}")
                continue

            # Convert content dict to YAML string for prompt
            content_yaml = yaml.dump(
                content, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120
            )

            # Format validation errors
            errors = errors_by_file[filename]
            error_text = "\n".join(f"- {error}" for error in errors)

            # Build prompt (with content only, headers preserved separately)
            prompt = build_validation_fix_prompt(
                yaml_content=content_yaml,
                validation_errors=error_text,
            )

            # Create request with "fix_" prefix for custom_id
            # The unpacker expects validation fix IDs to start with "fix_"
            request = self.create_request(
                custom_id=f"fix_{filename.replace('.yaml', '')}",
                prompt=prompt,
                pydantic_model=self.model_class,
            )

            requests.append(request)

        logger.info(f"Created {len(requests)} fix requests")
        return requests


def main():
    """CLI entry point for creating validation fix batches."""
    import argparse

    parser = argparse.ArgumentParser(description="Create validation fix batch requests")
    parser.add_argument("workflow_type", choices=["parameter", "test_statistic"])
    parser.add_argument("--data-dir", required=True, help="Directory with YAML files to fix")
    parser.add_argument(
        "--validation-results-dir",
        default="validation-outputs",
        help="Directory with validation reports",
    )
    parser.add_argument(
        "--output", default="batch_jobs/validation_fix_batch.jsonl", help="Output batch file"
    )

    args = parser.parse_args()

    # Determine model class
    model_class = ParameterMetadata if args.workflow_type == "parameter" else TestStatistic

    # Create batch
    creator = ValidationFixBatchCreator(
        data_dir=args.data_dir,
        validation_results_dir=args.validation_results_dir,
        output_file=args.output,
        model_class=model_class,
    )

    creator.create_and_save_batch()
    print(f"✓ Validation fix batch created: {args.output}")


if __name__ == "__main__":
    main()
