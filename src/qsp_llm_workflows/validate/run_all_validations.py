#!/usr/bin/env python3
"""
Master validation runner - executes all core validation checks.

Runs:
1. Template compliance validation
2. Code execution testing
3. Text snippet validation
4. Source reference validation
5. DOI resolution validation
6. Value consistency checking (vs legacy and same-context derivations)
7. Duplicate primary sources check (prevents duplicate extractions)
8. Automated snippet source verification (via Europe PMC, with manual fallback)

Usage:
    python scripts/validate/run_all_validations.py test_statistics
    python scripts/validate/run_all_validations.py parameter_estimates
"""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic
from qsp_llm_workflows.validate.orchestrator import ValidationOrchestrator
from qsp_llm_workflows.validate.check_schema_compliance import SchemaValidator
from qsp_llm_workflows.validate.test_code_execution import CodeExecutionValidator
from qsp_llm_workflows.validate.check_text_snippets import TextSnippetValidator
from qsp_llm_workflows.validate.check_source_references import SourceReferenceValidator
from qsp_llm_workflows.validate.check_doi_validity import DOIValidator
from qsp_llm_workflows.validate.check_value_consistency import ValueConsistencyChecker
from qsp_llm_workflows.validate.check_duplicate_primary_sources import (
    DuplicatePrimarySourceChecker,
)
from qsp_llm_workflows.validate.check_snippet_sources_automated import (
    AutomatedSnippetVerifier,
)
from qsp_llm_workflows.validate.tag_validation_results import tag_directory

# Load environment variables from .env file
load_dotenv()


def get_validators(data_dir: str, model_class):
    """
    Get list of validators to run.

    Args:
        data_dir: Directory containing YAML files
        model_class: Pydantic model class for schema validation

    Returns:
        List of Validator instances
    """
    return [
        SchemaValidator(data_dir, model_class=model_class),
        CodeExecutionValidator(data_dir, threshold_pct=5.0),
        TextSnippetValidator(data_dir),
        SourceReferenceValidator(data_dir),
        DOIValidator(data_dir, rate_limit=1.0),
        ValueConsistencyChecker(data_dir),
        DuplicatePrimarySourceChecker(data_dir),
        AutomatedSnippetVerifier(data_dir, rate_limit=0.5, fuzzy_threshold=0.8),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run all validation checks on metadata files",
        epilog="""
Examples:
    qsp-validate test_statistics --dir metadata-storage/to-review/test_statistics
    qsp-validate parameter_estimates --dir metadata-storage/to-review/parameter_estimates
        """,
    )
    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of workflow to validate",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory to validate (e.g., metadata-storage/to-review/parameter_estimates)",
    )

    args = parser.parse_args()

    # Determine model class based on workflow type
    if args.workflow_type == "test_statistics":
        model_class = TestStatistic
    else:  # parameter_estimates
        model_class = ParameterMetadata

    data_dir = Path(args.dir).resolve()

    output_dir = Path("validation-outputs")

    # Validate paths exist
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator
    orchestrator = ValidationOrchestrator(str(data_dir), output_dir, model_class.__name__)

    # Get validators
    validators = get_validators(str(data_dir), model_class)

    # Run all validations
    result = orchestrator.run_all_validations(validators)

    # Print master summary
    orchestrator.print_master_summary(result)

    # Tag files with validation results
    print(f"\n{'='*60}")
    print("TAGGING FILES WITH VALIDATION RESULTS")
    print(f"{'='*60}\n")

    # Get validation tags for passed validations
    validation_tags = [
        name.lower().replace(" ", "_").replace("-", "_")
        for name in result.get_passed_validation_names()
    ]

    if validation_tags:
        print(f"Validation tags to add: {', '.join(validation_tags)}")
        try:
            tag_directory(str(data_dir), validation_tags)
            print("✓ Files tagged successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not tag files: {e}")
    else:
        print("No validations passed - skipping tagging")

    print()

    # Check if any validations failed
    if result.has_failures:
        # Provide instructions for running validation fix workflow
        print("\n" + "=" * 60)
        print("VALIDATION FAILURES DETECTED")
        print("=" * 60)
        print("\nYou can automatically fix validation errors by submitting")
        print("failed YAMLs back to OpenAI for correction.")
        print("\nTo run validation fix workflow:")

        # Construct fix command with --dir if custom directory was used
        if args.dir:
            fix_cmd = f"qsp-fix {args.workflow_type} --immediate --dir {args.dir}"
        else:
            fix_cmd = f"qsp-fix {args.workflow_type} --immediate"

        print(f"  {fix_cmd}")
        print("\nThis will:")
        print("  1. Create fix batch requests from validation failures")
        print("  2. Upload to OpenAI API")
        print("  3. Monitor until completion")
        print("  4. Unpack fixed YAMLs (overwrites originals)")
        print("  5. Prompt you to re-run validation")
        print()

        sys.exit(1)
    else:
        print("\n✓ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
