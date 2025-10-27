"""
Utility functions for validation scripts.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


def load_yaml_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load a YAML file and return its contents.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary of YAML contents, or None if file cannot be loaded
    """
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        return None


def load_yaml_directory(directory: str, pattern: str = "*.yaml") -> List[Dict[str, Any]]:
    """
    Load all YAML files from a directory.

    Args:
        directory: Path to directory containing YAML files
        pattern: Glob pattern for matching files (default: *.yaml)

    Returns:
        List of dictionaries containing YAML contents and metadata
    """
    results = []
    dir_path = Path(directory)

    for filepath in dir_path.glob(pattern):
        data = load_yaml_file(str(filepath))
        if data is not None:
            results.append({
                'filepath': str(filepath),
                'filename': filepath.name,
                'data': data
            })

    return results


def extract_parameter_name_from_filename(filename: str) -> str:
    """
    Extract parameter name from standard filename format.

    Format: {param_name}_{author_year}_{cancer_type}_{hash}.yaml

    Args:
        filename: YAML filename

    Returns:
        Parameter name
    """
    # Remove .yaml extension
    base = filename.replace('.yaml', '')
    # Split on underscore and take first part
    parts = base.split('_')
    return parts[0] if parts else ''


def parse_numeric_value(value: Any) -> Optional[float]:
    """
    Parse a numeric value from various formats.

    Args:
        value: Value to parse (can be float, int, string, etc.)

    Returns:
        Float value, or None if cannot be parsed
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None

    return None


class ValidationReport:
    """
    Container for validation results with summary statistics.
    """

    def __init__(self, name: str):
        self.name = name
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, item: str, details: str = ""):
        """Add a passing result."""
        self.passed.append({'item': item, 'details': details})

    def add_fail(self, item: str, reason: str):
        """Add a failing result."""
        self.failed.append({'item': item, 'reason': reason})

    def add_warning(self, item: str, message: str):
        """Add a warning."""
        self.warnings.append({'item': item, 'message': message})

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total = len(self.passed) + len(self.failed)
        return {
            'name': self.name,
            'total': total,
            'passed': len(self.passed),
            'failed': len(self.failed),
            'warnings': len(self.warnings),
            'pass_rate': len(self.passed) / total if total > 0 else 0.0
        }

    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"Validation Report: {summary['name']}")
        print(f"{'='*60}")
        print(f"Total:    {summary['total']}")
        print(f"Passed:   {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
        print(f"Failed:   {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")

        if self.failed:
            print(f"\nFailed Items:")
            for item in self.failed[:10]:  # Show first 10
                print(f"  - {item['item']}: {item['reason']}")
            if len(self.failed) > 10:
                print(f"  ... and {len(self.failed) - 10} more")

        if self.warnings:
            print(f"\nWarnings:")
            for item in self.warnings[:10]:  # Show first 10
                print(f"  - {item['item']}: {item['message']}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

    def save_to_json(self, output_path: str):
        """Save full report to JSON file."""
        report = {
            'summary': self.get_summary(),
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
