#!/usr/bin/env python3
"""
Automated snippet source verification via Europe PMC.

Fetches full-text articles from Europe PMC and searches for snippets.
Falls back to manual verification for papers not available in PMC.

Usage:
    # Called as part of validation suite via run_all_validations.py
"""
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Optional

import requests

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


# Reuse collection functions from manual verifier
from qsp_llm_workflows.validate.check_snippet_sources_manual_verify import (
    collect_sources,
    extract_inputs_from_yaml,
    get_doi_from_source,
)


class AutomatedSnippetVerifier(Validator):
    """
    Automated snippet verification via Europe PMC full-text search.

    Fetches full-text XML from Europe PMC for papers with DOIs,
    parses to plain text, and fuzzy-searches for each snippet.

    Falls back to manual verification for papers not in PMC.
    """

    EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    EUROPE_PMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def __init__(
        self,
        data_dir: str,
        rate_limit: float = 0.5,
        fuzzy_threshold: float = 0.8,
        **kwargs,
    ):
        """
        Initialize automated verifier.

        Args:
            data_dir: Directory containing YAML files to validate
            rate_limit: Seconds between API requests (default 0.5)
            fuzzy_threshold: Similarity threshold for fuzzy matching (default 0.8)
            **kwargs: Additional configuration
        """
        super().__init__(data_dir, **kwargs)
        self.rate_limit = rate_limit
        self.fuzzy_threshold = fuzzy_threshold
        self.last_request_time = 0
        self._paper_text_cache: dict[str, Optional[str]] = {}

    @property
    def name(self) -> str:
        return "Automated Snippet Source Verification"

    def rate_limit_wait(self):
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def normalize_doi(self, doi: str) -> str:
        """Normalize DOI by removing URL prefixes."""
        if not doi:
            return ""
        doi_clean = doi.strip()
        doi_clean = doi_clean.replace("https://doi.org/", "")
        doi_clean = doi_clean.replace("http://doi.org/", "")
        return doi_clean

    def get_pmcid_from_doi(self, doi: str) -> Optional[str]:
        """
        Query Europe PMC to get PMCID from DOI.

        Only returns PMCID if full text is available (isOpenAccess or inPMC with full text).

        Args:
            doi: DOI string (e.g., "10.1056/NEJMoa1200690")

        Returns:
            PMCID (e.g., "PMC1234567") or None if not found or no full text
        """
        doi_clean = self.normalize_doi(doi)
        if not doi_clean:
            return None

        self.rate_limit_wait()

        try:
            params = {
                "query": f"DOI:{doi_clean}",
                "format": "json",
                "resultType": "core",
            }
            response = requests.get(
                self.EUROPE_PMC_SEARCH_URL,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            if not results:
                return None

            result = results[0]
            pmcid = result.get("pmcid")

            # Check if full text is actually available
            # isOpenAccess indicates OA full text; inPMC with hasFullText indicates PMC availability
            is_open_access = result.get("isOpenAccess") == "Y"
            in_pmc = result.get("inPMC") == "Y"

            # Only return PMCID if we expect full text to be available
            if pmcid and (is_open_access or in_pmc):
                return pmcid

            return None

        except (requests.RequestException, ValueError, KeyError):
            return None

    def fetch_full_text_xml(self, pmcid: str) -> Optional[str]:
        """
        Fetch full-text XML from Europe PMC.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            XML string or None if not available
        """
        if not pmcid:
            return None

        self.rate_limit_wait()

        try:
            url = f"{self.EUROPE_PMC_FULLTEXT_URL}/{pmcid}/fullTextXML"
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                return None

            return response.text

        except requests.RequestException:
            return None

    def extract_text_from_xml(self, xml_content: str) -> str:
        """
        Parse Europe PMC XML and extract searchable body text.

        Extracts text from <body> element and all descendants,
        strips XML tags, and normalizes whitespace.

        Args:
            xml_content: Raw XML string

        Returns:
            Plain text string suitable for searching
        """
        if not xml_content:
            return ""

        try:
            root = ET.fromstring(xml_content)

            # Find body element (JATS XML uses <body>)
            # Try different possible locations
            body = root.find(".//body")
            if body is None:
                # Try without namespace
                body = root.find("body")
            if body is None:
                # Fall back to extracting all text from document
                body = root

            # Extract all text content recursively
            def get_text(element) -> str:
                texts = []
                if element.text:
                    texts.append(element.text)
                for child in element:
                    texts.append(get_text(child))
                    if child.tail:
                        texts.append(child.tail)
                return " ".join(texts)

            text = get_text(body)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        except ET.ParseError:
            return ""

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching: lowercase, collapse whitespace."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.lower().strip())

    def fuzzy_find_snippet(self, snippet: str, full_text: str) -> tuple[bool, float]:
        """
        Search for snippet in full text using fuzzy matching.

        Strategy:
        1. Normalize both strings
        2. Try exact substring match first (fast path)
        3. If not found, use sliding window with SequenceMatcher

        Args:
            snippet: Text snippet to find
            full_text: Full paper text to search in

        Returns:
            (found, best_similarity_score) tuple
        """
        if not snippet or not full_text:
            return (False, 0.0)

        snippet_norm = self.normalize_text(snippet)
        text_norm = self.normalize_text(full_text)

        # Fast path: exact substring match
        if snippet_norm in text_norm:
            return (True, 1.0)

        # Sliding window fuzzy match
        snippet_len = len(snippet_norm)
        if snippet_len > len(text_norm):
            return (False, 0.0)

        best_score = 0.0
        # Use a larger window to account for minor variations
        window_size = int(snippet_len * 1.2)

        # Sample positions to avoid O(n*m) complexity on long documents
        # Check every 10 characters for documents > 10000 chars
        step = 1 if len(text_norm) < 10000 else max(1, snippet_len // 3)

        for i in range(0, len(text_norm) - snippet_len + 1, step):
            window = text_norm[i : i + window_size]
            score = SequenceMatcher(None, snippet_norm, window).ratio()
            if score > best_score:
                best_score = score
                if score >= self.fuzzy_threshold:
                    return (True, score)

        return (best_score >= self.fuzzy_threshold, best_score)

    def get_paper_text(self, doi: str) -> tuple[Optional[str], Optional[str]]:
        """
        Get full text for a paper by DOI.

        Returns cached result if available.

        Args:
            doi: DOI string

        Returns:
            (full_text, pmcid) tuple. full_text is None if not in PMC.
        """
        doi_clean = self.normalize_doi(doi)
        if doi_clean in self._paper_text_cache:
            cached = self._paper_text_cache[doi_clean]
            return (cached, doi_clean if cached else None)

        # Try to get from Europe PMC
        pmcid = self.get_pmcid_from_doi(doi_clean)
        if not pmcid:
            self._paper_text_cache[doi_clean] = None
            return (None, None)

        xml_content = self.fetch_full_text_xml(pmcid)
        if not xml_content:
            self._paper_text_cache[doi_clean] = None
            return (None, pmcid)

        full_text = self.extract_text_from_xml(xml_content)
        self._paper_text_cache[doi_clean] = full_text
        return (full_text, pmcid)

    def collect_verification_data(self) -> dict:
        """
        Collect all sources and snippets that need verification.

        Returns:
            Dict mapping source_tag to {doi, snippets, inputs}
        """
        files = load_yaml_directory(self.data_dir)

        source_data = defaultdict(lambda: {"doi": None, "snippets": set(), "inputs": []})

        for file_info in files:
            data = file_info["data"]
            filename = file_info["filename"]

            sources = collect_sources(data)
            inputs = extract_inputs_from_yaml(data)

            for inp in inputs:
                if not isinstance(inp, dict):
                    continue

                input_name = inp.get("name", "unnamed")
                source_ref = inp.get("source_ref")

                if not source_ref or source_ref not in sources:
                    continue

                source = sources[source_ref]
                doi = get_doi_from_source(source)

                if not doi:
                    continue

                value_snippet = inp.get("value_snippet")
                units_snippet = inp.get("units_snippet")

                if value_snippet:
                    source_data[source_ref]["snippets"].add(value_snippet)
                if units_snippet:
                    source_data[source_ref]["snippets"].add(units_snippet)

                source_data[source_ref]["doi"] = doi
                source_data[source_ref]["inputs"].append(
                    {
                        "name": input_name,
                        "filename": filename,
                        "value_snippet": value_snippet,
                        "units_snippet": units_snippet,
                    }
                )

        return source_data

    def print_manual_verification_prompt(self, manual_sources: dict) -> None:
        """Print sources requiring manual verification."""
        print("\n" + "=" * 70)
        print("MANUAL VERIFICATION REQUIRED")
        print("The following papers are not available in Europe PMC.")
        print("=" * 70 + "\n")

        for source_tag, info in sorted(manual_sources.items()):
            doi = info["doi"]
            snippets = sorted(info["snippets"])
            filenames = sorted(set(inp["filename"] for inp in info["inputs"]))

            print(f"Source: {source_tag}")
            print(f"DOI: https://doi.org/{self.normalize_doi(doi)}")
            print(f"YAML file(s): {', '.join(filenames)}")
            print(f"\nSnippets to verify ({len(snippets)}):")
            for i, snippet in enumerate(snippets, 1):
                display = snippet if len(snippet) <= 80 else snippet[:77] + "..."
                print(f'  {i}. "{display}"')
            print("-" * 70 + "\n")

        print(f"Total sources requiring manual verification: {len(manual_sources)}")
        print("=" * 70 + "\n")

    def get_manual_verification(self) -> bool:
        """Prompt user to verify remaining papers manually."""
        print("Please verify the snippets in the papers listed above.")
        print("Click DOI links and use Ctrl+F (Cmd+F) to search.\n")

        while True:
            response = input("Have all snippets been verified? [y/n]: ").lower()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")

    def validate(self) -> ValidationReport:
        """
        Run automated snippet verification.

        1. Collect all DOIs + snippets from YAML files
        2. For each DOI, try to fetch full text from Europe PMC
        3. Search for snippets in available papers
        4. Fall back to manual verification for papers not in PMC

        Returns:
            ValidationReport with results
        """
        report = ValidationReport(self.name)

        print(f"Verifying snippets in {self.data_dir}...")
        print("Using Europe PMC for full-text retrieval")
        print(f"Fuzzy match threshold: {self.fuzzy_threshold}")
        print()

        # Collect all sources and snippets
        source_data = self.collect_verification_data()

        if not source_data:
            print("No sources with snippets found.")
            report.add_pass("All sources", "No snippets to verify")
            return report

        # Track papers needing manual verification
        manual_sources: dict = {}
        automated_results: list[tuple[str, str, bool, float]] = []

        # Process each source
        total_sources = len(source_data)
        for idx, (source_tag, info) in enumerate(sorted(source_data.items()), 1):
            doi = info["doi"]
            snippets = info["snippets"]
            filenames = sorted(set(inp["filename"] for inp in info["inputs"]))

            print(f"[{idx}/{total_sources}] Checking {source_tag}...", end=" ")

            # Try to get full text
            full_text, pmcid = self.get_paper_text(doi)

            if not full_text:
                print("Not in PMC - queued for manual verification")
                manual_sources[source_tag] = info
                continue

            print(f"Found in PMC ({pmcid})")

            # Search for each snippet
            for snippet in snippets:
                found, score = self.fuzzy_find_snippet(snippet, full_text)
                item_name = f"{', '.join(filenames)} → {source_tag}"

                if found:
                    automated_results.append((item_name, snippet, True, score))
                else:
                    automated_results.append((item_name, snippet, False, score))

        print()

        # Add automated results to report
        for item_name, snippet, found, score in automated_results:
            snippet_display = snippet if len(snippet) <= 50 else snippet[:47] + "..."
            if found:
                report.add_pass(
                    f'{item_name} → "{snippet_display}"',
                    f"Found in PMC (score: {score:.2f})",
                )
            else:
                report.add_fail(
                    f'{item_name} → "{snippet_display}"',
                    f"Snippet not found in paper text (best score: {score:.2f})",
                )

        # Handle manual verification fallback
        if manual_sources:
            self.print_manual_verification_prompt(manual_sources)
            user_verified = self.get_manual_verification()

            for source_tag, info in manual_sources.items():
                filenames = sorted(set(inp["filename"] for inp in info["inputs"]))
                item_name = f"{', '.join(filenames)} → {source_tag}"

                if user_verified:
                    report.add_warning(
                        item_name,
                        "Paper not in Europe PMC, verified manually by user",
                    )
                else:
                    report.add_fail(
                        item_name,
                        "Paper not in Europe PMC, user did not verify",
                    )

        return report
