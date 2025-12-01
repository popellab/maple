#!/usr/bin/env python3
"""
Automated snippet source verification via NCBI PMC HTML scraping.

Fetches full-text articles from NCBI PubMed Central and searches for snippets.
For papers not in PMC, attempts verification against the abstract.
Falls back to manual verification only for papers with no available text.

Usage:
    # Called as part of validation suite via run_all_validations.py
"""
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

import requests
from bs4 import BeautifulSoup

from qsp_llm_workflows.validate.validator import Validator
from qsp_llm_workflows.core.validation_utils import load_yaml_directory, ValidationReport


# Reuse collection functions from manual verifier
from qsp_llm_workflows.validate.check_snippet_sources_manual_verify import (
    collect_sources,
    extract_inputs_from_yaml,
    get_doi_from_source,
)


@dataclass
class PaperInfo:
    """Information about a paper from Europe PMC."""

    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    is_open_access: bool = False
    in_pmc: bool = False
    abstract: Optional[str] = None
    title: Optional[str] = None


class AutomatedSnippetVerifier(Validator):
    """
    Automated snippet verification via NCBI PMC HTML scraping.

    Fetches full-text HTML from NCBI PubMed Central for papers with DOIs,
    extracts text with BeautifulSoup, and fuzzy-searches for each snippet.

    Falls back to manual verification for papers not in PMC.
    """

    EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    NCBI_PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles"

    # User agent required for NCBI PMC access
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

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

    def get_paper_info_from_doi(self, doi: str) -> Optional[PaperInfo]:
        """
        Query Europe PMC to get paper information from DOI.

        Returns PaperInfo with PMCID, open access status, and abstract.

        Args:
            doi: DOI string (e.g., "10.1056/NEJMoa1200690")

        Returns:
            PaperInfo object or None if not found
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

            # Strip HTML tags from abstract if present
            abstract = result.get("abstractText", "")
            if abstract:
                # Remove HTML tags like <h4>Background</h4>
                abstract = re.sub(r"<[^>]+>", " ", abstract)
                abstract = re.sub(r"\s+", " ", abstract).strip()

            return PaperInfo(
                pmcid=result.get("pmcid"),
                pmid=result.get("pmid"),
                is_open_access=result.get("isOpenAccess") == "Y",
                in_pmc=result.get("inPMC") == "Y",
                abstract=abstract if abstract else None,
                title=result.get("title"),
            )

        except (requests.RequestException, ValueError, KeyError):
            return None

    def fetch_pmc_html(self, pmcid: str) -> Optional[str]:
        """
        Fetch full-text HTML from NCBI PMC.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            HTML string or None if not available
        """
        if not pmcid:
            return None

        self.rate_limit_wait()

        try:
            url = f"{self.NCBI_PMC_ARTICLE_URL}/{pmcid}/"
            headers = {"User-Agent": self.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                return None

            return response.text

        except requests.RequestException:
            return None

    def extract_text_from_html(self, html_content: str) -> str:
        """
        Parse NCBI PMC HTML and extract searchable text.

        Uses BeautifulSoup to extract all text content from the page,
        stripping scripts and styles.

        Args:
            html_content: Raw HTML string

        Returns:
            Plain text string suitable for searching
        """
        if not html_content:
            return ""

        try:
            soup = BeautifulSoup(html_content, "lxml")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get text with space separator
            text = soup.get_text(separator=" ", strip=True)

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        except Exception:
            return ""

    def normalize_text(self, text: str) -> str:
        """Normalize text for matching: lowercase, collapse whitespace."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text.lower().strip())

    def normalize_snippet(self, snippet: str) -> str:
        """
        Normalize snippet by removing LaTeX formatting and table artifacts.

        Handles:
        - LaTeX superscripts: ^{+}, ^{-}, ^{2}, etc. -> +, -, 2
        - LaTeX subscripts: _{2}, _{d}, etc. -> 2, d
        - Table separators: |
        - Ellipsis placeholders: ...
        - Multiple whitespace/newlines

        Args:
            snippet: Raw snippet text

        Returns:
            Cleaned snippet suitable for matching
        """
        if not snippet:
            return ""

        text = snippet

        # Remove LaTeX super/subscripts: ^{+} -> +, _{2} -> 2
        text = re.sub(r"\^{([^}]*)}", r"\1", text)
        text = re.sub(r"_{([^}]*)}", r"\1", text)

        # Also handle simple ^ and _ without braces: ^+ -> +
        text = re.sub(r"\^(\S)", r"\1", text)
        text = re.sub(r"_(\S)", r"\1", text)

        # Remove table separators
        text = text.replace("|", " ")

        # Remove ellipsis placeholders
        text = text.replace("...", " ")

        # Collapse whitespace and newlines
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def fuzzy_find_snippet(
        self, snippet: str, full_text: str
    ) -> tuple[bool, float, str, Optional[str]]:
        """
        Search for snippet in full text using fuzzy matching.

        Strategy:
        1. Normalize snippet (remove LaTeX, table formatting)
        2. Normalize both strings (lowercase, collapse whitespace)
        3. Try exact substring match first (fast path)
        4. If not found, use sliding window with SequenceMatcher

        Args:
            snippet: Text snippet to find
            full_text: Full paper text to search in

        Returns:
            (found, best_similarity_score, normalized_snippet, best_match_text) tuple
            best_match_text is the text from the paper that best matched (for near-misses)
        """
        if not snippet or not full_text:
            return (False, 0.0, "", None)

        # First normalize snippet to remove LaTeX/table formatting
        snippet_cleaned = self.normalize_snippet(snippet)
        # Then apply standard text normalization (lowercase, whitespace)
        snippet_norm = self.normalize_text(snippet_cleaned)
        text_norm = self.normalize_text(full_text)

        # Fast path: exact substring match
        if snippet_norm in text_norm:
            return (True, 1.0, snippet_cleaned, None)

        # Sliding window fuzzy match
        snippet_len = len(snippet_norm)
        if snippet_len > len(text_norm):
            return (False, 0.0, snippet_cleaned, None)

        best_score = 0.0
        best_match_pos = 0
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
                best_match_pos = i
                if score >= self.fuzzy_threshold:
                    return (True, score, snippet_cleaned, None)

        # Extract best matching text for near-misses (score >= 0.5)
        best_match_text = None
        if best_score >= 0.5:
            # Get the matching window from the original (non-normalized) text
            # We need to approximate the position in the original text
            best_match_text = text_norm[best_match_pos : best_match_pos + window_size]

        return (best_score >= self.fuzzy_threshold, best_score, snippet_cleaned, best_match_text)

    def get_paper_text(self, doi: str) -> tuple[Optional[str], Optional[PaperInfo], str]:
        """
        Get text for a paper by DOI.

        Tries full text from PMC first, then falls back to abstract.
        Returns cached result if available.

        Args:
            doi: DOI string

        Returns:
            (text, paper_info, status) tuple.
            status is one of:
                - "full_text_open_access": Full text from open access PMC article
                - "full_text_restricted": Full text from restricted PMC article
                - "abstract_only": Only abstract available (not in PMC)
                - "cached": Using cached result
                - "html_fetch_failed": PMC article exists but couldn't fetch
                - "no_text_available": No text could be retrieved
        """
        doi_clean = self.normalize_doi(doi)

        # Check cache first
        cache_key = f"text_{doi_clean}"
        info_cache_key = f"info_{doi_clean}"
        if cache_key in self._paper_text_cache:
            cached_text = self._paper_text_cache[cache_key]
            cached_info = self._paper_text_cache.get(info_cache_key)
            if cached_text:
                return (cached_text, cached_info, "cached")
            # If cached as None, we already tried and failed
            return (None, cached_info, "no_text_available")

        # Get paper info from Europe PMC
        paper_info = self.get_paper_info_from_doi(doi_clean)
        if not paper_info:
            self._paper_text_cache[cache_key] = None
            return (None, None, "no_text_available")

        # Cache the paper info
        self._paper_text_cache[info_cache_key] = paper_info

        # Try to get full text from PMC if available
        if paper_info.in_pmc and paper_info.pmcid:
            html_content = self.fetch_pmc_html(paper_info.pmcid)
            if html_content:
                full_text = self.extract_text_from_html(html_content)
                if full_text:
                    self._paper_text_cache[cache_key] = full_text
                    status = (
                        "full_text_open_access"
                        if paper_info.is_open_access
                        else "full_text_restricted"
                    )
                    return (full_text, paper_info, status)

            # PMC fetch failed, but article should be there
            # Fall through to try abstract

        # Fall back to abstract
        if paper_info.abstract:
            self._paper_text_cache[cache_key] = paper_info.abstract
            return (paper_info.abstract, paper_info, "abstract_only")

        # No text available at all
        self._paper_text_cache[cache_key] = None
        return (None, paper_info, "no_text_available")

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
        print("The following papers are not available in PubMed Central.")
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
        2. For each DOI, try to fetch full text from NCBI PMC
        3. If not in PMC, try verification against abstract
        4. Fall back to manual verification only if no text available

        Returns:
            ValidationReport with results
        """
        report = ValidationReport(self.name)

        print(f"Verifying snippets in {self.data_dir}...")
        print("Using NCBI PubMed Central for full-text, Europe PMC for abstracts")
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
        automated_results: list[tuple[str, str, bool, float, str]] = []  # Added text_source

        # Process each source
        total_sources = len(source_data)
        for idx, (source_tag, info) in enumerate(sorted(source_data.items()), 1):
            doi = info["doi"]
            snippets = info["snippets"]
            filenames = sorted(set(inp["filename"] for inp in info["inputs"]))

            print(f"[{idx}/{total_sources}] Checking {source_tag}...")

            # Try to get text (full text or abstract)
            text, paper_info, status = self.get_paper_text(doi)

            if not text:
                if paper_info:
                    print("  ✗ No full text or abstract available - queued for manual verification")
                else:
                    print("  ✗ Paper not found in Europe PMC - queued for manual verification")
                manual_sources[source_tag] = info
                continue

            # Report what we got
            text_len = len(text)
            pmcid = paper_info.pmcid if paper_info else None

            if status == "full_text_open_access":
                print(f"  ✓ Full text from PMC ({pmcid}) - Open Access - {text_len:,} chars")
                text_source = "full_text"
            elif status == "full_text_restricted":
                print(f"  ✓ Full text from PMC ({pmcid}) - Restricted Access - {text_len:,} chars")
                text_source = "full_text"
            elif status == "abstract_only":
                oa_label = "Open Access" if paper_info.is_open_access else "Restricted"
                print(f"  ⚠ Abstract only ({oa_label}) - {text_len:,} chars - not in PMC")
                text_source = "abstract"
            elif status == "cached":
                print(f"  ✓ Using cached text - {text_len:,} chars")
                text_source = "cached"
            else:
                print(f"  ✓ Retrieved text - {text_len:,} chars")
                text_source = "unknown"

            # Search for each snippet and collect results
            snippet_results: list[tuple[str, str, bool, float, str, Optional[str]]] = []
            for snippet in snippets:
                found, score, normalized, best_match = self.fuzzy_find_snippet(snippet, text)
                snippet_results.append((snippet, normalized, found, score, best_match))

            # Sort: failures first (by score ascending), then successes
            snippet_results.sort(key=lambda x: (x[2], x[3]))  # (found, score)

            # ANSI color codes
            GREEN = "\033[92m"
            YELLOW = "\033[93m"
            RED = "\033[91m"
            RESET = "\033[0m"

            num_snippets = len(snippet_results)
            found_count = 0
            item_name = f"{', '.join(filenames)} → {source_tag}"

            for snippet, normalized, found, score, best_match in snippet_results:
                # Check if normalization changed the snippet
                was_normalized = normalized != snippet

                # Display truncated versions
                norm_display = normalized if len(normalized) <= 50 else normalized[:47] + "..."

                if found:
                    found_count += 1
                    automated_results.append((item_name, snippet, True, score, text_source))
                    print(f'{GREEN}    ✓ "{norm_display}" (score: {score:.2f}){RESET}')
                else:
                    automated_results.append((item_name, snippet, False, score, text_source))

                    # Color based on score: yellow for near-miss (0.6-0.8), red for failure
                    is_near_miss = 0.6 <= score < self.fuzzy_threshold
                    color = YELLOW if is_near_miss else RED

                    if was_normalized:
                        snippet_display = snippet if len(snippet) <= 40 else snippet[:37] + "..."
                        print(f'{color}    ✗ "{snippet_display}"{RESET}')
                        print(f'{color}      → normalized: "{norm_display}"{RESET}')
                        print(f"{color}      → best score: {score:.2f}{RESET}")
                    else:
                        print(f'{color}    ✗ "{norm_display}" (best score: {score:.2f}){RESET}')

                    # Show best matching text for near-misses
                    if is_near_miss and best_match:
                        match_display = (
                            best_match if len(best_match) <= 50 else best_match[:47] + "..."
                        )
                        print(f'{YELLOW}      → closest match: "{match_display}"{RESET}')

            print(f"  Summary: {found_count}/{num_snippets} snippets matched")

        print()

        # Add automated results to report
        for item_name, snippet, found, score, source_type in automated_results:
            snippet_display = snippet if len(snippet) <= 50 else snippet[:47] + "..."
            if found:
                if source_type == "abstract":
                    report.add_pass(
                        f'{item_name} → "{snippet_display}"',
                        f"Found in abstract (score: {score:.2f})",
                    )
                else:
                    report.add_pass(
                        f'{item_name} → "{snippet_display}"',
                        f"Found in full text (score: {score:.2f})",
                    )
            else:
                if source_type == "abstract":
                    report.add_fail(
                        f'{item_name} → "{snippet_display}"',
                        f"Not found in abstract (best score: {score:.2f}) - full text not available",
                    )
                else:
                    report.add_fail(
                        f'{item_name} → "{snippet_display}"',
                        f"Snippet not found in full text (best score: {score:.2f})",
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
                        "Paper not in PubMed Central, verified manually by user",
                    )
                else:
                    report.add_fail(
                        item_name,
                        "Paper not in PubMed Central, user did not verify",
                    )

        return report
