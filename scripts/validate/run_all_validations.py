#!/usr/bin/env python3
"""
Master validation runner - executes all core validation checks.

Runs:
1. Template compliance validation
2. Code execution testing
3. Text snippet validation
4. Source reference validation
5. DOI resolution validation

Usage:
    python scripts/validate/run_all_validations.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        templates/parameter_metadata_template.yaml \\
        output/validation_results/
"""
import argparse
import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime

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
    parser = argparse.ArgumentParser(description="Run all core validation checks")
    parser.add_argument(
        "data_dir",
        help="Directory with parameter YAML files (e.g., ../qsp-metadata-storage/parameter_estimates)"
    )
    parser.add_argument(
        "template",
        help="Path to template YAML (e.g., templates/parameter_metadata_template_v2.yaml)"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for validation results"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print("# CORE AUTOMATED VALIDATION SUITE")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Data directory: {args.data_dir}")
    print(f"# Template: {args.template}")
    print(f"# Output directory: {args.output_dir}")
    print(f"{'#'*60}")

    all_results = []

    # 1. Template compliance
    result = run_validation(
        'check_schema_compliance.py',
        [args.data_dir, args.template, str(output_dir / 'schema_compliance.json')],
        "Template Compliance Validation"
    )
    all_results.append(result)
    print(result['stdout'])

    # 2. Code execution
    result = run_validation(
        'test_code_execution.py',
        [args.data_dir, str(output_dir / 'code_execution.json')],
        "Code Execution Testing"
    )
    all_results.append(result)
    print(result['stdout'])

    # 3. Text snippet validation
    result = run_validation(
        'check_text_snippets.py',
        [args.data_dir, str(output_dir / 'text_snippets.json')],
        "Text Snippet Validation"
    )
    all_results.append(result)
    print(result['stdout'])

    # 4. Source reference validation
    result = run_validation(
        'check_source_references.py',
        [args.data_dir, str(output_dir / 'source_references.json')],
        "Source Reference Validation"
    )
    all_results.append(result)
    print(result['stdout'])

    # 5. DOI resolution
    result = run_validation(
        'check_doi_validity.py',
        [args.data_dir, str(output_dir / 'doi_validity.json')],
        "DOI Resolution Validation"
    )
    all_results.append(result)
    print(result['stdout'])

    # Generate master summary
    print(f"\n{'='*60}")
    print("MASTER VALIDATION SUMMARY")
    print(f"{'='*60}")

    master_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'template': args.template,
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

    # Exit with error code if any validations failed
    if any(not r.get('success', False) for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
