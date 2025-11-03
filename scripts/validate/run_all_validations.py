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

Usage:
    python scripts/validate/run_all_validations.py test_statistics
    python scripts/validate/run_all_validations.py parameter_estimates
"""
import argparse
import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_validation(script_name: str, args: list, description: str) -> dict:
    """
    Run a validation script and return results.

    Args:
        script_name: Name of validation script
        args: List of command-line arguments
        description: Human-readable description

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")

    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = ['python', script_path] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        success = result.returncode == 0

        return {
            'validation': description,
            'script': script_name,
            'success': success,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            'validation': description,
            'script': script_name,
            'success': False,
            'error': 'Timeout (>10 min)'
        }
    except Exception as e:
        return {
            'validation': description,
            'script': script_name,
            'success': False,
            'error': str(e)
        }


def load_validation_summary(json_path: str) -> dict:
    """Load summary from validation JSON output."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if 'summary' in data:
                return data['summary']
            return data
    except:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run all validation checks on metadata files",
        epilog="""
Examples:
    python scripts/validate/run_all_validations.py test_statistics
    python scripts/validate/run_all_validations.py parameter_estimates
        """
    )
    parser.add_argument(
        "workflow_type",
        choices=["parameter_estimates", "test_statistics"],
        help="Type of workflow to validate"
    )

    args = parser.parse_args()

    # Determine paths based on workflow type
    if args.workflow_type == "test_statistics":
        data_dir = Path("../qsp-metadata-storage/to-review/test_statistics")
        template = Path("templates/test_statistic_template.yaml")
    else:  # parameter_estimates
        data_dir = Path("../qsp-metadata-storage/to-review/parameter_estimates")
        template = Path("templates/parameter_metadata_template.yaml")

    output_dir = Path("output/validation_results")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths exist
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    if not template.exists():
        print(f"Error: Template not found: {template}")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print("# CORE AUTOMATED VALIDATION SUITE")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Data directory: {data_dir}")
    print(f"# Template: {template}")
    print(f"# Output directory: {output_dir}")
    print(f"{'#'*60}")

    all_results = []

    # 1. Template compliance
    result = run_validation(
        'check_schema_compliance.py',
        [str(data_dir), str(template), str(output_dir / 'schema_compliance.json')],
        "Template Compliance Validation"
    )
    all_results.append(result)
    print(result['stdout'])
    if not result['success'] and result.get('stderr'):
        print("ERRORS:", file=sys.stderr)
        print(result['stderr'], file=sys.stderr)

    # 2. Code execution
    result = run_validation(
        'test_code_execution.py',
        [str(data_dir), str(output_dir / 'code_execution.json')],
        "Code Execution Testing"
    )
    all_results.append(result)
    print(result['stdout'])
    if not result['success'] and result.get('stderr'):
        print("ERRORS:", file=sys.stderr)
        print(result['stderr'], file=sys.stderr)

    # 3. Text snippet validation
    result = run_validation(
        'check_text_snippets.py',
        [str(data_dir), str(output_dir / 'text_snippets.json')],
        "Text Snippet Validation"
    )
    all_results.append(result)
    print(result['stdout'])
    if not result['success'] and result.get('stderr'):
        print("ERRORS:", file=sys.stderr)
        print(result['stderr'], file=sys.stderr)

    # 4. Source reference validation
    result = run_validation(
        'check_source_references.py',
        [str(data_dir), str(output_dir / 'source_references.json')],
        "Source Reference Validation"
    )
    all_results.append(result)
    print(result['stdout'])
    if not result['success'] and result.get('stderr'):
        print("ERRORS:", file=sys.stderr)
        print(result['stderr'], file=sys.stderr)

    # 5. DOI resolution
    result = run_validation(
        'check_doi_validity.py',
        [str(data_dir), str(output_dir / 'doi_validity.json')],
        "DOI Resolution Validation"
    )
    all_results.append(result)
    print(result['stdout'])
    if not result['success'] and result.get('stderr'):
        print("ERRORS:", file=sys.stderr)
        print(result['stderr'], file=sys.stderr)

    # 6. Value consistency
    result = run_validation(
        'check_value_consistency.py',
        [str(data_dir), str(output_dir / 'value_consistency.json')],
        "Value Consistency Checking"
    )
    all_results.append(result)
    print(result['stdout'])
    if not result['success'] and result.get('stderr'):
        print("ERRORS:", file=sys.stderr)
        print(result['stderr'], file=sys.stderr)

    # 7. Manual snippet source verification (interactive - run directly)
    print(f"\n{'='*60}")
    print("Running: Manual Snippet Source Verification")
    print(f"{'='*60}")

    script_path = os.path.join(os.path.dirname(__file__), 'check_snippet_sources_manual_verify.py')
    cmd = ['python', script_path, str(data_dir), str(output_dir / 'snippet_sources.json')]

    try:
        # Run without capturing output so user can interact
        result = subprocess.run(cmd, timeout=600)  # 10 min timeout

        success = result.returncode == 0
        all_results.append({
            'validation': 'Manual Snippet Source Verification',
            'script': 'check_snippet_sources_manual_verify.py',
            'success': success,
            'returncode': result.returncode
        })

    except subprocess.TimeoutExpired:
        print("ERROR: Manual verification timed out (>10 min)", file=sys.stderr)
        all_results.append({
            'validation': 'Manual Snippet Source Verification',
            'script': 'check_snippet_sources_manual_verify.py',
            'success': False,
            'error': 'Timeout'
        })
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        all_results.append({
            'validation': 'Manual Snippet Source Verification',
            'script': 'check_snippet_sources_manual_verify.py',
            'success': False,
            'error': str(e)
        })

    # Generate master summary
    print(f"\n{'='*60}")
    print("MASTER VALIDATION SUMMARY")
    print(f"{'='*60}")

    master_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': str(data_dir),
        'template': str(template),
        'validations': []
    }

    for result in all_results:
        validation_name = result['validation']
        success = result.get('success', False)
        status = "✓ PASSED" if success else "✗ FAILED"

        print(f"\n{validation_name}: {status}")

        # Try to load detailed summary
        summary_data = None
        if success and 'script' in result:
            script_name = result['script'].replace('.py', '')
            summary_path = output_dir / f"{script_name.replace('check_', '').replace('test_', '')}_summary.json"
            summary_data = load_validation_summary(str(summary_path))

        validation_entry = {
            'name': validation_name,
            'success': success,
            'summary': summary_data
        }

        if summary_data:
            if 'total' in summary_data:
                print(f"  Total: {summary_data['total']}")
                print(f"  Passed: {summary_data['passed']} ({summary_data.get('pass_rate', 0)*100:.1f}%)")
                print(f"  Failed: {summary_data['failed']}")
            elif 'n_comparisons' in summary_data:
                print(f"  Comparisons: {summary_data['n_comparisons']}")
                if 'pearson_r' in summary_data:
                    print(f"  Correlation: {summary_data['pearson_r']:.3f}")

        master_summary['validations'].append(validation_entry)

    # Save master summary
    summary_path = output_dir / 'master_validation_summary.json'
    with open(summary_path, 'w') as f:
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
        if result.get('success', False):
            # Convert validation name to tag format
            name = result['validation']
            tag = name.lower().replace(' ', '_').replace('-', '_')
            validation_tags.append(tag)

    if validation_tags:
        print(f"Validation tags to add: {', '.join(validation_tags)}")

        # Tag all files in the data directory
        tag_script = Path(__file__).parent / 'tag_validation_results.py'
        try:
            tag_result = subprocess.run(
                ['python', str(tag_script), str(data_dir), *validation_tags],
                capture_output=True,
                text=True,
                timeout=120
            )

            if tag_result.returncode == 0:
                print(tag_result.stdout)
            else:
                print(f"⚠ Warning: Could not tag files")
                if tag_result.stderr:
                    print(f"  Error: {tag_result.stderr}")
        except Exception as e:
            print(f"⚠ Warning: Could not tag files: {e}")
    else:
        print("No validations passed - skipping tagging")

    print()

    # Check if any validations failed
    has_failures = any(not r.get('success', False) for r in all_results)

    if has_failures:
        # Prompt user to run validation fix workflow
        print("\n" + "="*60)
        print("VALIDATION FAILURES DETECTED")
        print("="*60)
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

        if response.lower() == 'y':
            print("\nLaunching validation fix workflow...")
            print("="*60 + "\n")

            # Run validation fix workflow
            script_dir = Path(__file__).parent.parent
            fix_script = script_dir / "run_validation_fix.py"

            cmd = [
                sys.executable,
                str(fix_script),
                args.workflow_type,
                "--immediate"  # Use immediate by default for faster feedback
            ]

            result = subprocess.run(cmd)
            sys.exit(result.returncode)
        else:
            print("\nTo manually run validation fix later:")
            print(f"  python scripts/run_validation_fix.py {args.workflow_type} --immediate")
            sys.exit(1)
    else:
        print("\n✓ All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
