#!/usr/bin/env python3
"""
Check value consistency across related extractions.

Compares new extractions against:
1. Legacy database values (files with '_legacy' suffix)
2. Other derivations with same context_hash (same model context, different studies)

Reports discrepancies as warnings (does not fail validation).

Works for both parameter estimates and test statistics.

Usage:
    python scripts/validate/check_value_consistency.py \\
        ../qsp-metadata-storage/parameter_estimates \\
        output/value_consistency.json
"""
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import (
    load_yaml_directory,
    parse_numeric_value,
    ValidationReport
)


class ValueConsistencyChecker:
    """
    Check value consistency across related extractions.
    Works for both parameters and test statistics.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.legacy_dir = self._get_legacy_dir()
        self.all_files = []
        self.legacy_values = defaultdict(list)  # param_name -> list of values
        self.context_groups = defaultdict(list)  # (param_name, context_hash) -> list of values

    def _get_legacy_dir(self) -> Path:
        """
        Determine legacy directory path based on data directory.

        Convention:
        - parameter_estimates -> parameter_estimates_legacy
        - test_statistics -> test_statistics_legacy
        - quick_estimates -> quick_estimates_legacy
        """
        dir_name = self.data_dir.name
        parent = self.data_dir.parent

        legacy_name = f"{dir_name}_legacy"
        legacy_path = parent / legacy_name

        return legacy_path if legacy_path.exists() else None

    def load_all_files(self):
        """Load all YAML files and organize by parameter/context."""
        print(f"Loading files from {self.data_dir}...")
        self.all_files = load_yaml_directory(str(self.data_dir))

        # Also load legacy files if legacy directory exists
        legacy_files = []
        if self.legacy_dir:
            print(f"Loading legacy files from {self.legacy_dir}...")
            legacy_files = load_yaml_directory(str(self.legacy_dir))

        all_files_combined = self.all_files + legacy_files

        for file_info in all_files_combined:
            filename = file_info['filename']
            filepath = file_info['filepath']
            data = file_info['data']

            # Extract parameter/test_statistic identifier
            identifier = self._get_identifier(data, filename)
            if not identifier:
                continue

            # Extract mean value
            mean_value = self._extract_mean(data)
            if mean_value is None:
                continue

            # Extract context_hash
            context_hash = data.get('context_hash', 'unknown')

            # Determine if this is a legacy file (either by directory or _legacy suffix)
            is_legacy = (self.legacy_dir and str(self.legacy_dir) in filepath) or '_legacy' in filename

            # Store in appropriate collection
            if is_legacy:
                self.legacy_values[identifier].append({
                    'value': mean_value,
                    'filename': filename,
                    'context_hash': context_hash
                })
            else:
                self.context_groups[(identifier, context_hash)].append({
                    'value': mean_value,
                    'filename': filename
                })

        print(f"  Found {len(self.all_files)} regular files")
        if legacy_files:
            print(f"  Found {len(legacy_files)} legacy files")
        print(f"  {len(self.legacy_values)} identifiers with legacy values")
        print(f"  {len(self.context_groups)} unique (identifier, context_hash) groups")

    def _get_identifier(self, data: dict, filename: str) -> str:
        """
        Get identifier from YAML (parameter_name or test_statistic_id).

        Returns:
            Identifier string or None
        """
        # Try parameter_name first
        if 'parameter_name' in data:
            return data['parameter_name']

        # Try test_statistic_id
        if 'test_statistic_id' in data:
            return data['test_statistic_id']

        return None

    def _extract_mean(self, data: dict) -> float:
        """
        Extract mean value from either parameter_estimates or test_statistic_estimates.

        Returns:
            Mean value or None
        """
        # Try parameter_estimates
        if 'parameter_estimates' in data:
            estimates = data['parameter_estimates']
            if isinstance(estimates, dict) and 'mean' in estimates:
                return parse_numeric_value(estimates['mean'])

        # Try test_statistic_estimates
        if 'test_statistic_estimates' in data:
            estimates = data['test_statistic_estimates']
            if isinstance(estimates, dict) and 'mean' in estimates:
                return parse_numeric_value(estimates['mean'])

        return None

    def compare_to_legacy(self, identifier: str, value: float, filename: str) -> tuple:
        """
        Compare value to legacy database values.

        Returns:
            (has_comparison, warnings) tuple
        """
        if identifier not in self.legacy_values:
            return (False, [])

        legacy_entries = self.legacy_values[identifier]
        warnings = []

        for legacy_entry in legacy_entries:
            legacy_value = legacy_entry['value']
            legacy_file = legacy_entry['filename']

            # Calculate percent difference
            pct_diff = abs(value - legacy_value) / abs(legacy_value) * 100 if legacy_value != 0 else float('inf')

            # Report if difference is notable (>20%)
            if pct_diff > 20:
                warnings.append(
                    f"Large difference from legacy ({pct_diff:.1f}%): "
                    f"new={value:.3e}, legacy={legacy_value:.3e} ({legacy_file})"
                )

        return (True, warnings)

    def compare_to_same_context(self, identifier: str, context_hash: str,
                                value: float, filename: str) -> tuple:
        """
        Compare value to other derivations with same context_hash.

        Returns:
            (has_comparison, warnings) tuple
        """
        key = (identifier, context_hash)

        if key not in self.context_groups:
            return (False, [])

        same_context_entries = self.context_groups[key]

        # Exclude self
        other_entries = [e for e in same_context_entries if e['filename'] != filename]

        if not other_entries:
            return (False, [])

        warnings = []
        other_values = [e['value'] for e in other_entries]

        # Calculate statistics of other values
        mean_others = np.mean(other_values)
        std_others = np.std(other_values, ddof=1) if len(other_values) > 1 else 0
        min_others = np.min(other_values)
        max_others = np.max(other_values)

        # Check if new value is within range
        if value < min_others or value > max_others:
            pct_diff_from_mean = abs(value - mean_others) / abs(mean_others) * 100 if mean_others != 0 else float('inf')
            warnings.append(
                f"Outside range of same-context derivations ({pct_diff_from_mean:.1f}% from mean): "
                f"new={value:.3e}, range=[{min_others:.3e}, {max_others:.3e}], "
                f"n={len(other_values)}"
            )

        # Check if far from mean (>50% difference)
        elif mean_others != 0:
            pct_diff_from_mean = abs(value - mean_others) / abs(mean_others) * 100
            if pct_diff_from_mean > 50:
                warnings.append(
                    f"Large difference from same-context mean ({pct_diff_from_mean:.1f}%): "
                    f"new={value:.3e}, mean={mean_others:.3e}, n={len(other_values)}"
                )

        return (True, warnings)

    def validate_file(self, file_info: dict) -> tuple:
        """
        Check value consistency for a single file.

        Returns:
            (is_valid, warnings) tuple
        """
        data = file_info['data']
        filename = file_info['filename']
        filepath = file_info['filepath']

        # Skip legacy files (they are the reference, not being validated)
        is_legacy = (self.legacy_dir and str(self.legacy_dir) in filepath) or '_legacy' in filename
        if is_legacy:
            return (True, [])

        # Get identifier and value
        identifier = self._get_identifier(data, filename)
        if not identifier:
            return (True, [])

        value = self._extract_mean(data)
        if value is None:
            return (True, [])

        context_hash = data.get('context_hash', 'unknown')

        all_warnings = []

        # Compare to legacy
        has_legacy, legacy_warnings = self.compare_to_legacy(identifier, value, filename)
        if has_legacy and legacy_warnings:
            all_warnings.extend(legacy_warnings)

        # Compare to same context
        has_context, context_warnings = self.compare_to_same_context(
            identifier, context_hash, value, filename
        )
        if has_context and context_warnings:
            all_warnings.extend(context_warnings)

        # If no comparisons available, note it
        if not has_legacy and not has_context:
            all_warnings.append("No comparison values available (first extraction for this context)")

        # Always pass validation (warnings only)
        return (True, all_warnings)

    def validate_directory(self) -> ValidationReport:
        """Check value consistency across all files."""
        # First load all files to build comparison database
        self.load_all_files()

        report = ValidationReport("Value Consistency")

        print(f"\nChecking value consistency...")

        for file_info in self.all_files:
            filename = file_info['filename']

            is_valid, warnings = self.validate_file(file_info)

            if warnings:
                # Report warnings but still mark as pass
                warning_msg = "; ".join(warnings)
                report.add_pass(filename, f"⚠ {warning_msg}")
            else:
                report.add_pass(filename, "Values consistent with existing data")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Check value consistency across related extractions"
    )
    parser.add_argument("data_dir", help="Directory with YAML files to validate")
    parser.add_argument("output", help="Output JSON file for validation report")

    args = parser.parse_args()

    # Run validation
    validator = ValueConsistencyChecker(args.data_dir)
    report = validator.validate_directory()

    # Print summary
    report.print_summary()

    # Save results
    report.save_to_json(args.output)
    print(f"\nValue consistency report saved to {args.output}")


if __name__ == "__main__":
    main()
