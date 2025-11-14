#!/usr/bin/env python3
"""
Manual snippet source verification.

Generates a report of sources with DOI links and snippets for manual verification.
Waits for user to verify snippets in papers, then writes validation report.

Usage:
    python scripts/validate/check_snippet_sources_manual_verify.py \
        ../qsp-metadata-storage/parameter_estimates \
        output/snippet_sources.json
"""
import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


def collect_sources(data: dict) -> dict:
    """Collect all defined sources from YAML."""
    sources = {}

    # Collect from primary_data_sources
    if 'primary_data_sources' in data:
        pds = data['primary_data_sources']
        if isinstance(pds, list):
            for source in pds:
                if isinstance(source, dict) and 'source_tag' in source:
                    sources[source['source_tag']] = source
        elif isinstance(pds, dict):
            for tag, source in pds.items():
                sources[tag] = source if isinstance(source, dict) else {'source_tag': tag}

    # Collect from secondary_data_sources
    if 'secondary_data_sources' in data:
        sds = data['secondary_data_sources']
        if isinstance(sds, list):
            for source in sds:
                if isinstance(source, dict) and 'source_tag' in source:
                    sources[source['source_tag']] = source
        elif isinstance(sds, dict):
            for tag, source in sds.items():
                sources[tag] = source if isinstance(source, dict) else {'source_tag': tag}

    # Collect from methodological_sources
    if 'methodological_sources' in data:
        ms = data['methodological_sources']
        if isinstance(ms, list):
            for source in ms:
                if isinstance(source, dict) and 'source_tag' in source:
                    sources[source['source_tag']] = source
        elif isinstance(ms, dict):
            for tag, source in ms.items():
                sources[tag] = source if isinstance(source, dict) else {'source_tag': tag}

    return sources


def extract_inputs_from_yaml(data: dict) -> list:
    """Extract inputs from either parameter_estimates or test_statistic_estimates."""
    # Try parameter_estimates first
    if 'parameter_estimates' in data and isinstance(data['parameter_estimates'], dict):
        if 'inputs' in data['parameter_estimates']:
            return data['parameter_estimates']['inputs']

    # Try test_statistic_estimates
    if 'test_statistic_estimates' in data and isinstance(data['test_statistic_estimates'], dict):
        if 'inputs' in data['test_statistic_estimates']:
            return data['test_statistic_estimates']['inputs']

    return []


def get_doi_from_source(source: dict) -> str:
    """Extract DOI from source definition."""
    # Primary sources use 'doi'
    if 'doi' in source:
        return source['doi']

    # Secondary/methodological use 'doi_or_url'
    if 'doi_or_url' in source:
        value = source['doi_or_url']
        if value and (value.startswith('10.') or 'doi.org' in value.lower()):
            return value

    return None


def collect_verification_data(data_dir: str) -> dict:
    """
    Collect all sources and snippets that need verification.

    Returns:
        Dict mapping source_tag to {doi, snippets, inputs}
    """
    files = load_yaml_directory(data_dir)

    source_data = defaultdict(lambda: {
        'doi': None,
        'snippets': set(),
        'inputs': []
    })

    for file_info in files:
        data = file_info['data']
        filename = file_info['filename']

        sources = collect_sources(data)
        inputs = extract_inputs_from_yaml(data)

        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            input_name = inp.get('name', 'unnamed')
            source_ref = inp.get('source_ref')

            if not source_ref or source_ref not in sources:
                continue

            source = sources[source_ref]
            doi = get_doi_from_source(source)

            if not doi:
                continue

            value_snippet = inp.get('value_snippet')
            units_snippet = inp.get('units_snippet')

            if value_snippet:
                source_data[source_ref]['snippets'].add(value_snippet)
            if units_snippet:
                source_data[source_ref]['snippets'].add(units_snippet)

            source_data[source_ref]['doi'] = doi
            source_data[source_ref]['inputs'].append({
                'name': input_name,
                'filename': filename,
                'value_snippet': value_snippet,
                'units_snippet': units_snippet
            })

    return source_data


def print_verification_report(source_data: dict):
    """Print verification report to console."""
    print("\n" + "="*80)
    print("MANUAL SNIPPET SOURCE VERIFICATION")
    print("="*80)
    print("\nInstructions:")
    print("  1. For each source below, click the DOI link to open the paper")
    print("  2. Use Ctrl+F (Cmd+F on Mac) to search for each snippet")
    print("  3. Verify that each snippet appears verbatim in the paper")
    print("\n" + "="*80 + "\n")

    for source_tag in sorted(source_data.keys()):
        info = source_data[source_tag]
        doi = info['doi']
        snippets = sorted(info['snippets'])

        # Get unique filenames
        filenames = sorted(set(inp['filename'] for inp in info['inputs']))

        print(f"Source: {source_tag}")
        print(f"DOI: https://doi.org/{doi}")
        print(f"YAML file(s): {', '.join(filenames)}")
        print(f"\nSnippets to verify ({len(snippets)} unique):")
        for i, snippet in enumerate(snippets, 1):
            # Truncate long snippets for display
            display_snippet = snippet if len(snippet) <= 100 else snippet[:97] + "..."
            print(f"  {i}. \"{display_snippet}\"")
        print(f"\nUsed in {len(info['inputs'])} input(s)")
        print("-" * 80 + "\n")

    print(f"\nTotal sources to verify: {len(source_data)}")
    print(f"Total unique snippets: {sum(len(info['snippets']) for info in source_data.values())}")
    print("="*80 + "\n")


def get_user_verification() -> bool:
    """Prompt user to verify snippets and return result."""
    print("Please verify all snippets in the papers listed above.")
    print()

    while True:
        response = input("Have all snippets been verified? [y/n]: ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")


class SnippetSourceManualVerifier:
    """Interactive manual verification of snippets in source papers."""

    def __init__(self, data_dir: str):
        """Initialize verifier with data directory."""
        self.data_dir = data_dir

    def verify_interactive(self) -> dict:
        """
        Interactively verify snippets with user.

        Returns:
            dict: Verification results with user_verified boolean
        """
        # Collect data
        source_data = collect_verification_data(self.data_dir)

        if not source_data:
            print("No sources with snippets found.")
            return {'user_verified': False}

        # Print report
        print_verification_report(source_data)

        # Get user verification
        user_verified = get_user_verification()

        return {'user_verified': user_verified}
