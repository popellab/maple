#!/usr/bin/env python3
"""
Generate manual verification report for text snippets.

Lists each source with its DOI link and all unique snippets to verify.
Makes it easy to manually click DOI and search for snippets in the paper.

Usage:
    python scripts/validate/check_snippet_sources_manual.py \
        ../qsp-metadata-storage/parameter_estimates \
        output/snippet_verification_manual.txt
"""
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.dirname(__file__))

from validation_utils import load_yaml_directory


def collect_sources(data: dict) -> dict:
    """
    Collect all defined sources from YAML.

    Returns:
        Dict mapping source_tag to source definition
    """
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
    """
    Extract inputs from either parameter_estimates or test_statistic_estimates.

    Returns:
        List of input dicts
    """
    # Try parameter_estimates first (params)
    if 'parameter_estimates' in data and isinstance(data['parameter_estimates'], dict):
        if 'inputs' in data['parameter_estimates']:
            return data['parameter_estimates']['inputs']

    # Try test_statistic_estimates (test stats)
    if 'test_statistic_estimates' in data and isinstance(data['test_statistic_estimates'], dict):
        if 'inputs' in data['test_statistic_estimates']:
            return data['test_statistic_estimates']['inputs']

    return []


def get_doi_from_source(source: dict) -> str:
    """
    Extract DOI from source definition.

    Args:
        source: Source dict

    Returns:
        DOI string or None
    """
    # Primary sources use 'doi'
    if 'doi' in source:
        return source['doi']

    # Secondary/methodological use 'doi_or_url'
    if 'doi_or_url' in source:
        value = source['doi_or_url']
        # Check if it's a DOI (not a URL)
        if value and (value.startswith('10.') or 'doi.org' in value.lower()):
            return value

    return None


def generate_manual_report(data_dir: str) -> dict:
    """
    Generate manual verification report.

    Returns:
        Dict mapping source_tag to {doi, snippets, inputs}
    """
    files = load_yaml_directory(data_dir)

    # Collect snippets by source
    source_snippets = defaultdict(lambda: {
        'doi': None,
        'snippets': set(),
        'inputs': []
    })

    for file_info in files:
        data = file_info['data']
        filename = file_info['filename']

        # Collect sources
        sources = collect_sources(data)

        # Extract inputs
        inputs = extract_inputs_from_yaml(data)

        for inp in inputs:
            if not isinstance(inp, dict):
                continue

            input_name = inp.get('name', 'unnamed')
            source_ref = inp.get('source_ref')

            if not source_ref:
                continue

            if source_ref not in sources:
                continue

            # Get DOI
            source = sources[source_ref]
            doi = get_doi_from_source(source)

            if not doi:
                continue

            # Collect snippets
            value_snippet = inp.get('value_snippet')
            units_snippet = inp.get('units_snippet')

            if value_snippet:
                source_snippets[source_ref]['snippets'].add(value_snippet)
            if units_snippet:
                source_snippets[source_ref]['snippets'].add(units_snippet)

            # Store DOI and input info
            source_snippets[source_ref]['doi'] = doi
            source_snippets[source_ref]['inputs'].append({
                'name': input_name,
                'filename': filename,
                'value_snippet': value_snippet,
                'units_snippet': units_snippet
            })

    return source_snippets


def format_report(source_snippets: dict) -> str:
    """Format the manual verification report."""
    lines = []
    lines.append("=" * 80)
    lines.append("MANUAL SNIPPET VERIFICATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Instructions:")
    lines.append("  1. Click each DOI link to open the paper")
    lines.append("  2. Use Ctrl+F (Cmd+F on Mac) to search for each snippet")
    lines.append("  3. Verify that the snippet appears verbatim in the paper")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Sort by source tag
    for source_tag in sorted(source_snippets.keys()):
        info = source_snippets[source_tag]
        doi = info['doi']
        snippets = sorted(info['snippets'])
        inputs = info['inputs']

        lines.append(f"Source: {source_tag}")
        lines.append(f"DOI: https://doi.org/{doi}")
        lines.append("")
        lines.append("Snippets to verify:")
        for i, snippet in enumerate(snippets, 1):
            lines.append(f"  {i}. \"{snippet}\"")
        lines.append("")
        lines.append(f"Used in {len(inputs)} input(s):")
        for inp in inputs:
            lines.append(f"  - {inp['filename']}: {inp['name']}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    lines.append("")
    lines.append("=" * 80)
    lines.append(f"Total sources to verify: {len(source_snippets)}")
    lines.append(f"Total snippets: {sum(len(info['snippets']) for info in source_snippets.values())}")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate manual verification report for text snippets"
    )
    parser.add_argument("data_dir", help="Directory with YAML files")
    parser.add_argument("output", help="Output text file for manual verification report")

    args = parser.parse_args()

    source_snippets = generate_manual_report(args.data_dir)
    report_text = format_report(source_snippets)

    # Save to file
    with open(args.output, 'w') as f:
        f.write(report_text)

    # Print to console (will be captured by caller)
    print(report_text)


if __name__ == "__main__":
    main()
