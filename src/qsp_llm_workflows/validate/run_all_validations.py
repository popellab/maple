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
8. Manual snippet source verification (interactive)

Usage:
    python scripts/validate/run_all_validations.py test_statistics
    python scripts/validate/run_all_validations.py parameter_estimates
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from qsp_llm_workflows.core.pydantic_models import ParameterMetadata, TestStatistic
from qsp_llm_workflows.validate.check_schema_compliance import SchemaValidator
from qsp_llm_workflows.validate.test_code_execution import CodeExecutionValidator
from qsp_llm_workflows.validate.check_text_snippets import TextSnippetValidator
from qsp_llm_workflows.validate.check_source_references import SourceReferenceValidator
from qsp_llm_workflows.validate.check_doi_validity import DOIValidator
from qsp_llm_workflows.validate.check_value_consistency import ValueConsistencyChecker
from qsp_llm_workflows.validate.check_duplicate_primary_sources import DuplicatePrimarySourceChecker
from qsp_llm_workflows.validate.check_snippet_sources_manual_verify import (
    SnippetSourceManualVerifier,
)
from qsp_llm_workflows.validate.tag_validation_results import tag_directory

# Load environment variables from .env file
load_dotenv()


def run_validation(validator, description: str) -> dict:
    """
    Run a validation and return results.

    Args:
        validator: Validator instance with validate() method
        description: Human-readable description

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")

    try:
        results = validator.validate()
        success = results.get("summary", {}).get("passed", 0) == results.get("summary", {}).get(
            "total", 0
        )

        return {"validation": description, "success": success, "results": results}

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        return {"validation": description, "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Run all validation checks on metadata files",
        epilog="""
Examples:
    python scripts/validate/run_all_validations.py test_statistics
    python scripts/validate/run_all_validations.py parameter_estimates
        """,
    )
    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of workflow to validate",
    )

    args = parser.parse_args()

    # Determine paths and model based on workflow type
    if args.workflow_type == "test_statistics":
        data_dir = Path("../qsp-metadata-storage/to-review/test_statistics")
        model_class = TestStatistic
    else:  # parameter_estimates
        data_dir = Path("../qsp-metadata-storage/to-review/parameter_estimates")
        model_class = ParameterMetadata

    output_dir = Path("output/validation_results")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths exist
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print("# CORE AUTOMATED VALIDATION SUITE")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Data directory: {data_dir}")
    print(f"# Pydantic model: {model_class.__name__}")
    print(f"# Output directory: {output_dir}")
    print(f"{'#'*60}")

    all_results = []

    # 1. Schema compliance (using Pydantic models)
    validator = SchemaValidator(str(data_dir), model_class)
    result = run_validation(validator, "Template Compliance Validation")
    all_results.append(result)
    if result.get("results"):
        # Save to JSON
        with open(output_dir / "schema_compliance.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        # Print summary
        summary = result["results"].get("summary", {})
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

    # 2. Code execution
    validator = CodeExecutionValidator(str(data_dir))
    result = run_validation(validator, "Code Execution Testing")
    all_results.append(result)
    if result.get("results"):
        with open(output_dir / "code_execution.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        summary = result["results"].get("summary", {})
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

    # 3. Text snippet validation
    validator = TextSnippetValidator(str(data_dir))
    result = run_validation(validator, "Text Snippet Validation")
    all_results.append(result)
    if result.get("results"):
        with open(output_dir / "text_snippets.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        summary = result["results"].get("summary", {})
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

    # 4. Source reference validation
    validator = SourceReferenceValidator(str(data_dir))
    result = run_validation(validator, "Source Reference Validation")
    all_results.append(result)
    if result.get("results"):
        with open(output_dir / "source_references.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        summary = result["results"].get("summary", {})
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

    # 5. DOI resolution
    validator = DOIValidator(str(data_dir))
    result = run_validation(validator, "DOI Resolution Validation")
    all_results.append(result)
    if result.get("results"):
        with open(output_dir / "doi_validity.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        summary = result["results"].get("summary", {})
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

    # 6. Value consistency
    validator = ValueConsistencyChecker(str(data_dir))
    result = run_validation(validator, "Value Consistency Checking")
    all_results.append(result)
    if result.get("results"):
        with open(output_dir / "value_consistency.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        summary = result["results"].get("summary", {})
        print(f"  Comparisons: {summary.get('n_comparisons', 0)}")

    # 7. Duplicate primary sources
    validator = DuplicatePrimarySourceChecker(str(data_dir))
    result = run_validation(validator, "Duplicate Primary Sources Check")
    all_results.append(result)
    if result.get("results"):
        with open(output_dir / "duplicate_primary_sources.json", "w") as f:
            json.dump(result["results"], f, indent=2)
        summary = result["results"].get("summary", {})
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

    # 8. Manual snippet source verification (interactive)
    print(f"\n{'='*60}")
    print("Running: Manual Snippet Source Verification")
    print(f"{'='*60}")

    try:
        verifier = SnippetSourceManualVerifier(str(data_dir))
        results = verifier.verify_interactive()

        success = results.get("user_verified", False)
        all_results.append(
            {
                "validation": "Manual Snippet Source Verification",
                "success": success,
                "results": results,
            }
        )

        with open(output_dir / "snippet_sources.json", "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        all_results.append(
            {"validation": "Manual Snippet Source Verification", "success": False, "error": str(e)}
        )

    # Generate master summary
    print(f"\n{'='*60}")
    print("MASTER VALIDATION SUMMARY")
    print(f"{'='*60}")

    master_summary = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "template": str(template),
        "validations": [],
    }

    for result in all_results:
        validation_name = result["validation"]
        success = result.get("success", False)
        status = "✓ PASSED" if success else "✗ FAILED"

        print(f"\n{validation_name}: {status}")

        validation_entry = {
            "name": validation_name,
            "success": success,
            "summary": result.get("results", {}).get("summary"),
        }

        summary_data = result.get("results", {}).get("summary")
        if summary_data:
            if "total" in summary_data:
                print(f"  Total: {summary_data['total']}")
                print(
                    f"  Passed: {summary_data['passed']} ({summary_data.get('pass_rate', 0)*100:.1f}%)"
                )
                print(f"  Failed: {summary_data['failed']}")
            elif "n_comparisons" in summary_data:
                print(f"  Comparisons: {summary_data['n_comparisons']}")
                if "pearson_r" in summary_data:
                    print(f"  Correlation: {summary_data['pearson_r']:.3f}")

        master_summary["validations"].append(validation_entry)

    # Save master summary
    summary_path = output_dir / "master_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Master summary saved to: {summary_path}")
    print(f"Individual reports in: {output_dir}")
    print(f"{'='*60}\n")

    # Tag files with validation results
    print(f"\n{'='*60}")
    print("TAGGING FILES WITH VALIDATION RESULTS")
    print(f"{'='*60}\n")

    # Determine which validations passed
    validation_tags = []
    for result in all_results:
        if result.get("success", False):
            # Convert validation name to tag format
            name = result["validation"]
            tag = name.lower().replace(" ", "_").replace("-", "_")
            validation_tags.append(tag)

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
    has_failures = any(not r.get("success", False) for r in all_results)

    if has_failures:
        # Prompt user to run validation fix workflow
        print("\n" + "=" * 60)
        print("VALIDATION FAILURES DETECTED")
        print("=" * 60)
        print("\nYou can automatically fix validation errors by submitting")
        print("failed YAMLs back to OpenAI for correction.")
        print("\nThis will:")
        print("  1. Create fix batch requests from validation failures")
        print("  2. Upload to OpenAI API")
        print("  3. Monitor until completion")
        print("  4. Unpack fixed YAMLs (overwrites originals)")
        print("  5. Prompt you to re-run validation")
        print("\nNote: Original files are backed up in git history.")

        response = input("\nRun validation fix workflow? [y/N]: ")

        if response.lower() == "y":
            print("\nLaunching validation fix workflow...")
            print("=" * 60 + "\n")

            # Import and run validation fix workflow directly
            from qsp_llm_workflows.core.workflow_orchestrator import WorkflowOrchestrator

            orchestrator = WorkflowOrchestrator()
            result = orchestrator.run_validation_fix_workflow(
                workflow_type=args.workflow_type,
                use_batch_api=False,  # Use immediate mode by default
            )

            sys.exit(0 if result else 1)
        else:
            print("\nTo manually run validation fix later:")
            print(f"  python scripts/run_validation_fix.py {args.workflow_type} --immediate")
            sys.exit(1)
    else:
        print("\n✓ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
