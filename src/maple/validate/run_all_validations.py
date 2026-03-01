#!/usr/bin/env python3
"""
Master validation runner - executes all core validation checks.

Runs:
1. Template compliance validation
2. Code execution testing (derivation_code)
3. Model output code validation (test statistics only - compute_test_statistic function)
4. Text snippet validation
5. Source reference validation
6. DOI resolution validation
7. Value consistency checking (vs legacy and same-context derivations)
8. Duplicate primary sources check (prevents duplicate extractions)
9. Automated snippet source verification (via Europe PMC, with manual fallback)

Usage:
    python scripts/validate/run_all_validations.py test_statistics
    python scripts/validate/run_all_validations.py parameter_estimates
"""
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

from maple.core.pydantic_models import ParameterMetadata, TestStatistic
from maple.validate.orchestrator import ValidationOrchestrator
from maple.validate.check_schema_compliance import SchemaValidator
from maple.validate.test_code_execution import CodeExecutionValidator
from maple.validate.check_text_snippets import TextSnippetValidator
from maple.validate.check_source_references import SourceReferenceValidator
from maple.validate.check_doi_validity import DOIValidator
from maple.validate.check_value_consistency import ValueConsistencyChecker
from maple.validate.check_duplicate_primary_sources import (
    DuplicatePrimarySourceChecker,
)
from maple.validate.check_snippet_sources_automated import (
    AutomatedSnippetVerifier,
)
from maple.validate.check_model_output_code import ModelOutputCodeValidator
from maple.validate.tag_validation_results import tag_files_individually

# Load environment variables from .env file
load_dotenv()


def get_validators(data_dir: str, model_class, species_units_file: str | None = None):
    """
    Get list of validators to run.

    Args:
        data_dir: Directory containing YAML files
        model_class: Pydantic model class for schema validation
        species_units_file: Optional path to species_units.json for model output code validation

    Returns:
        List of Validator instances
    """
    return [
        SchemaValidator(data_dir, model_class=model_class),
        CodeExecutionValidator(data_dir, threshold_pct=5.0),
        ModelOutputCodeValidator(data_dir, species_units_file=species_units_file),
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
    parser.add_argument(
        "--species-units-file",
        type=str,
        default=None,
        help="Path to species_units.json for model output code validation (from qsp-export-model)",
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
    validators = get_validators(str(data_dir), model_class, args.species_units_file)

    # Run all validations
    result = orchestrator.run_all_validations(validators)

    # Print master summary
    orchestrator.print_master_summary(result)

    # Tag files with validation results (per-file based on which validations each passed)
    print(f"\n{'='*60}")
    print("TAGGING FILES WITH VALIDATION RESULTS")
    print(f"{'='*60}\n")

    # Get per-file validation tags
    file_tags = result.get_per_file_tags()

    if file_tags:
        try:
            tagged_count = tag_files_individually(str(data_dir), file_tags)
            print(f"✓ Tagged {tagged_count} files with their passed validations")
        except Exception as e:
            print(f"⚠ Warning: Could not tag files: {e}")
    else:
        print("No files passed any validations - skipping tagging")

    print()

    # Exit with appropriate code
    if result.has_failures:
        sys.exit(1)
    else:
        print("✓ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
