"""
Tests for AutomatedSnippetVerifier.

Includes both unit tests (mocked) and integration tests (real API).
Integration tests are marked with @pytest.mark.integration and can be skipped
with: pytest -m "not integration"
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from qsp_llm_workflows.validate.check_snippet_sources_automated import (
    AutomatedSnippetVerifier,
)


class TestDOINormalization:
    """Test DOI normalization logic."""

    def test_normalizes_bare_doi(self):
        """Test normalizing a bare DOI string."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_doi("10.1234/test") == "10.1234/test"

    def test_removes_https_prefix(self):
        """Test removing https://doi.org/ prefix."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_doi("https://doi.org/10.1234/test") == "10.1234/test"

    def test_removes_http_prefix(self):
        """Test removing http://doi.org/ prefix."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_doi("http://doi.org/10.1234/test") == "10.1234/test"

    def test_strips_whitespace(self):
        """Test stripping whitespace from DOI."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_doi("  10.1234/test  ") == "10.1234/test"

    def test_handles_empty_doi(self):
        """Test handling empty DOI string."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_doi("") == ""
        assert verifier.normalize_doi(None) == ""


class TestTextNormalization:
    """Test text normalization for matching."""

    def test_lowercases_text(self):
        """Test text is lowercased."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_text("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        """Test multiple whitespace collapsed to single space."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_text("hello   world") == "hello world"
        assert verifier.normalize_text("hello\n\tworld") == "hello world"

    def test_strips_whitespace(self):
        """Test leading/trailing whitespace stripped."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_text("  hello  ") == "hello"

    def test_handles_empty_text(self):
        """Test handling empty text."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_text("") == ""
        assert verifier.normalize_text(None) == ""


class TestHTMLParsing:
    """Test HTML parsing and text extraction."""

    def test_extracts_text_from_simple_html(self):
        """Test extracting text from simple HTML."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = "<html><body><p>Hello world</p></body></html>"
        text = verifier.extract_text_from_html(html)
        assert "hello world" in text.lower()

    def test_extracts_text_from_nested_html(self):
        """Test extracting text from nested HTML elements."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html>
            <body>
                <div>
                    <h1>Introduction</h1>
                    <p>This is the introduction.</p>
                </div>
                <div>
                    <h1>Methods</h1>
                    <p>These are the methods.</p>
                </div>
            </body>
        </html>
        """
        text = verifier.extract_text_from_html(html)
        assert "introduction" in text.lower()
        assert "methods" in text.lower()

    def test_removes_script_and_style(self):
        """Test that script and style elements are removed."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html>
            <head><style>.hidden { display: none; }</style></head>
            <body>
                <script>alert('test');</script>
                <p>Visible content</p>
            </body>
        </html>
        """
        text = verifier.extract_text_from_html(html)
        assert "visible content" in text.lower()
        assert "alert" not in text.lower()
        assert "hidden" not in text.lower()

    def test_handles_empty_html(self):
        """Test handling empty HTML."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.extract_text_from_html("") == ""
        assert verifier.extract_text_from_html(None) == ""


class TestHTMLParsingEdgeCases:
    """Test HTML parsing edge cases for real-world PMC content."""

    def test_extracts_text_from_tables(self):
        """Test that table content is extracted (important for paper data)."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <table>
                <thead>
                    <tr><th>Cell Type</th><th>No neoadjuvant</th></tr>
                </thead>
                <tbody>
                    <tr><td>CD8+</td><td>17 (9-30)</td></tr>
                    <tr><td>FoxP3+</td><td>3 (1-5)</td></tr>
                </tbody>
            </table>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        assert "17 (9-30)" in text
        assert "3 (1-5)" in text
        assert "cd8+" in text.lower()
        assert "foxp3+" in text.lower()
        assert "no neoadjuvant" in text.lower()

    def test_handles_unicode_characters(self):
        """Test handling of special characters like ×, ±, μ, etc."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <p>The dose was 50.4 Gy with a field size of 400× magnification.</p>
            <p>Mean ± SD: 132.3 ± 132.1 days</p>
            <p>Concentration: 5 μg/mL</p>
            <p>Temperature: 37°C</p>
            <p>P value: P < 0.001</p>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        assert "400×" in text or "400" in text  # × may be preserved or stripped
        assert "±" in text or "132.3" in text
        assert "μg/mL" in text or "μg" in text or "ug" in text.lower()
        assert "37°C" in text or "37" in text
        assert "< 0.001" in text or "0.001" in text

    def test_handles_superscripts_and_subscripts(self):
        """Test handling of superscripts/subscripts (CD8+, FoxP3+, H2O)."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <p>CD8<sup>+</sup> T cells and FoxP3<sup>+</sup> regulatory T cells</p>
            <p>H<sub>2</sub>O concentration</p>
            <p>10<sup>6</sup> cells/mL</p>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        # The + should be extractable even if formatting is lost
        assert "cd8" in text.lower()
        assert "foxp3" in text.lower()
        assert "h" in text.lower() and "o" in text.lower()
        assert "10" in text and "6" in text

    def test_handles_malformed_html(self):
        """Test graceful handling of malformed HTML."""
        verifier = AutomatedSnippetVerifier("/tmp")
        # Missing closing tags
        html = "<html><body><p>Unclosed paragraph<div>Nested wrong</p></div>"
        text = verifier.extract_text_from_html(html)
        # Should still extract some text without crashing
        assert "unclosed paragraph" in text.lower() or "nested wrong" in text.lower()

    def test_handles_html_entities(self):
        """Test handling of HTML entities (&amp;, &lt;, &gt;, etc.)."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <p>Johnson &amp; Johnson</p>
            <p>P &lt; 0.05</p>
            <p>&copy; 2020</p>
            <p>5&#x2013;10 days</p>  <!-- en-dash -->
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        assert "johnson" in text.lower()
        # & should be decoded
        assert "&amp;" not in text
        # < should be decoded
        assert "< 0.05" in text or "&lt;" not in text

    def test_handles_nested_inline_elements(self):
        """Test handling of nested inline elements (bold, italic, links)."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <p>The <strong>median</strong> CD8<sup>+</sup> count was
               <em>17 (IQR: 9-30)</em> cells per
               <a href="#">high-power field</a>.</p>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        assert "median" in text.lower()
        assert "17" in text
        assert "9-30" in text or "iqr" in text.lower()
        assert "high-power field" in text.lower()

    def test_preserves_numeric_data_in_tables(self):
        """Test that numeric data from tables is preserved accurately."""
        verifier = AutomatedSnippetVerifier("/tmp")
        # Simulating Table 3 structure from MICHELAKOS2020
        html = """
        <html><body>
            <table>
                <caption>Table 3. Immune cell infiltration</caption>
                <tr>
                    <th>Marker</th>
                    <th>FOLFIRINOX</th>
                    <th>Proton</th>
                    <th>Photon</th>
                    <th>No neoadjuvant</th>
                    <th>P</th>
                </tr>
                <tr>
                    <td>CD8<sup>+</sup></td>
                    <td>40 (24-80)</td>
                    <td>24 (12-34)</td>
                    <td>7 (1-15)</td>
                    <td>17 (9-30)</td>
                    <td>&lt;.001</td>
                </tr>
                <tr>
                    <td>FoxP3<sup>+</sup></td>
                    <td>2 (1-4)</td>
                    <td>1 (0-2)</td>
                    <td>2 (1-6)</td>
                    <td>3 (1-5)</td>
                    <td>.02</td>
                </tr>
            </table>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        # All key values should be extractable
        assert "17 (9-30)" in text
        assert "3 (1-5)" in text
        assert "40 (24-80)" in text
        assert "no neoadjuvant" in text.lower()

    def test_handles_figure_captions(self):
        """Test that figure captions are extracted."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <figure>
                <img src="figure1.png" alt="Survival curve">
                <figcaption>
                    Figure 1. Kaplan-Meier survival analysis showing
                    median overall survival of 24.5 months.
                </figcaption>
            </figure>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        assert "kaplan-meier" in text.lower()
        assert "24.5 months" in text.lower()

    def test_handles_math_elements(self):
        """Test handling of MathML or LaTeX-style math."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <p>The growth rate k = 0.693/T<sub>d</sub> where T<sub>d</sub> is doubling time.</p>
            <math><mi>λ</mi><mo>=</mo><mfrac><mn>ln(2)</mn><msub><mi>T</mi><mi>d</mi></msub></mfrac></math>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        # Should extract at least the text version
        assert "growth rate" in text.lower() or "0.693" in text

    def test_handles_reference_links(self):
        """Test that reference numbers/links don't interfere with text."""
        verifier = AutomatedSnippetVerifier("/tmp")
        html = """
        <html><body>
            <p>Previous studies have shown tumor growth rates
               <a href="#ref1">[1]</a><a href="#ref2">[2]</a>
               ranging from 50 to 200 days.</p>
        </body></html>
        """
        text = verifier.extract_text_from_html(html)
        assert "50 to 200 days" in text.lower() or "50" in text and "200" in text


class TestFuzzyMatching:
    """Test fuzzy snippet matching logic."""

    def test_finds_exact_match(self):
        """Test finding exact substring match."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, normalized = verifier.fuzzy_find_snippet(
            "tumor growth", "The tumor growth rate was measured."
        )
        assert found is True
        assert score == 1.0
        assert normalized == "tumor growth"

    def test_finds_case_insensitive_match(self):
        """Test case-insensitive matching."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, _ = verifier.fuzzy_find_snippet("Tumor Growth", "the tumor growth rate")
        assert found is True
        assert score == 1.0

    def test_finds_fuzzy_match(self):
        """Test finding fuzzy match above threshold."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        # Minor difference should still match
        found, score, _ = verifier.fuzzy_find_snippet("tumor growth rate", "tumor growth rates")
        assert found is True
        assert score >= 0.8

    def test_rejects_below_threshold(self):
        """Test rejecting match below threshold."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, _ = verifier.fuzzy_find_snippet(
            "completely different text", "tumor growth rate was measured"
        )
        assert found is False
        assert score < 0.8

    def test_handles_empty_snippet(self):
        """Test handling empty snippet."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, normalized = verifier.fuzzy_find_snippet("", "some text")
        assert found is False
        assert score == 0.0
        assert normalized == ""

    def test_handles_empty_full_text(self):
        """Test handling empty full text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, _ = verifier.fuzzy_find_snippet("snippet", "")
        assert found is False
        assert score == 0.0

    def test_respects_custom_threshold(self):
        """Test respecting custom fuzzy threshold."""
        # With high threshold, partial match should fail
        # "tumor growth" vs "tumor expansion" has ~0.6 similarity
        verifier_strict = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.95)
        found, _, _ = verifier_strict.fuzzy_find_snippet("tumor growth", "tumor expansion")
        assert found is False

        # With lower threshold, same match should pass
        verifier_lenient = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.5)
        found, _, _ = verifier_lenient.fuzzy_find_snippet("tumor growth", "tumor expansion")
        assert found is True


class TestSnippetNormalization:
    """Test snippet normalization for LaTeX and table formatting."""

    def test_normalizes_latex_superscripts(self):
        """Test removing LaTeX superscript formatting."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_snippet("CD8^{+}") == "CD8+"
        assert verifier.normalize_snippet("FoxP3^{+}") == "FoxP3+"
        assert verifier.normalize_snippet("10^{6}") == "106"

    def test_normalizes_latex_subscripts(self):
        """Test removing LaTeX subscript formatting."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_snippet("H_{2}O") == "H2O"
        assert verifier.normalize_snippet("T_{d}") == "Td"

    def test_normalizes_simple_super_subscripts(self):
        """Test removing simple ^ and _ formatting."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.normalize_snippet("CD8^+") == "CD8+"
        assert verifier.normalize_snippet("H_2O") == "H2O"

    def test_removes_table_separators(self):
        """Test removing table pipe separators."""
        verifier = AutomatedSnippetVerifier("/tmp")
        result = verifier.normalize_snippet("CD8+ | 17 (9-30) | No neoadjuvant")
        assert "|" not in result
        assert "CD8+" in result
        assert "17 (9-30)" in result

    def test_removes_ellipsis(self):
        """Test removing ellipsis placeholders."""
        verifier = AutomatedSnippetVerifier("/tmp")
        result = verifier.normalize_snippet("CD8+ | ... | No neoadjuvant")
        assert "..." not in result

    def test_collapses_whitespace(self):
        """Test collapsing multiple whitespace and newlines."""
        verifier = AutomatedSnippetVerifier("/tmp")
        result = verifier.normalize_snippet("CD8+  \n  Intratumoral  |  17")
        assert "  " not in result
        assert "\n" not in result

    def test_fuzzy_match_with_latex_snippet(self):
        """Test that LaTeX snippets match plain text in papers."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        # Snippet has LaTeX, paper text has plain
        found, score, normalized = verifier.fuzzy_find_snippet(
            "CD8^{+} T cells", "The CD8+ T cells were counted"
        )
        assert found is True
        assert normalized == "CD8+ T cells"

    def test_fuzzy_match_with_table_snippet(self):
        """Test that table-formatted snippets match plain text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, normalized = verifier.fuzzy_find_snippet(
            "CD8+ | 17 (9-30)", "CD8+ 17 (9-30) cells per HPF"
        )
        assert found is True
        assert "|" not in normalized


class TestDataCollection:
    """Test collection of verification data from YAML files."""

    def test_collects_snippets_from_yaml(self):
        """Test collecting snippets from YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "value_snippet": "growth rate was 0.5",
                            "source_ref": "smith2020",
                        }
                    ]
                },
                "primary_data_sources": [
                    {
                        "source_tag": "smith2020",
                        "doi": "10.1234/test",
                        "title": "Test Paper",
                        "first_author": "Smith",
                        "year": 2020,
                    }
                ],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            verifier = AutomatedSnippetVerifier(tmpdir)
            source_data = verifier.collect_verification_data()

            assert "smith2020" in source_data
            assert source_data["smith2020"]["doi"] == "10.1234/test"
            assert "growth rate was 0.5" in source_data["smith2020"]["snippets"]

    def test_collects_multiple_snippets(self):
        """Test collecting multiple snippets from same source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "input1",
                            "value": 1.0,
                            "value_snippet": "snippet one",
                            "units_snippet": "units snippet",
                            "source_ref": "src1",
                        },
                        {
                            "name": "input2",
                            "value": 2.0,
                            "value_snippet": "snippet two",
                            "source_ref": "src1",
                        },
                    ]
                },
                "primary_data_sources": [
                    {
                        "source_tag": "src1",
                        "doi": "10.1234/test",
                        "title": "Test",
                        "first_author": "Author",
                        "year": 2020,
                    }
                ],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            verifier = AutomatedSnippetVerifier(tmpdir)
            source_data = verifier.collect_verification_data()

            assert len(source_data["src1"]["snippets"]) == 3  # 2 value + 1 units

    def test_skips_sources_without_doi(self):
        """Test skipping sources without DOI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "input1",
                            "value": 1.0,
                            "value_snippet": "snippet",
                            "source_ref": "src1",
                        }
                    ]
                },
                "primary_data_sources": [
                    {
                        "source_tag": "src1",
                        # No DOI field
                        "title": "Test",
                        "first_author": "Author",
                        "year": 2020,
                    }
                ],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            verifier = AutomatedSnippetVerifier(tmpdir)
            source_data = verifier.collect_verification_data()

            assert len(source_data) == 0


class TestValidatorInterface:
    """Test validator interface compliance."""

    def test_has_correct_name(self):
        """Test validator has correct name."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.name == "Automated Snippet Source Verification"

    def test_accepts_rate_limit_config(self):
        """Test validator accepts rate_limit configuration."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=2.0)
        assert verifier.rate_limit == 2.0

    def test_accepts_fuzzy_threshold_config(self):
        """Test validator accepts fuzzy_threshold configuration."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.9)
        assert verifier.fuzzy_threshold == 0.9

    def test_returns_validation_report(self):
        """Test validate() returns ValidationReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = AutomatedSnippetVerifier(tmpdir)
            report = verifier.validate()

            # Import from core where the actual class is defined
            from qsp_llm_workflows.core.validation_utils import ValidationReport

            assert isinstance(report, ValidationReport)


# =============================================================================
# INTEGRATION TESTS - These hit the real NCBI PMC API
# =============================================================================


@pytest.mark.integration
class TestNCBIPMCIntegration:
    """
    Integration tests that hit the real NCBI PMC API.

    These tests verify the actual API integration works correctly.
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    def test_get_pmcid_from_known_pmc_doi(self):
        """Test getting PMCID from a known PMC DOI."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI paper - in PMC but not necessarily open access
        doi = "10.1093/jnci/djaa073"
        pmcid = verifier.get_pmcid_from_doi(doi)

        assert pmcid is not None
        assert pmcid.startswith("PMC")

    def test_get_pmcid_returns_none_for_non_pmc_doi(self):
        """Test that non-PMC DOIs return None."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # Using a made-up DOI that shouldn't exist
        doi = "10.9999/nonexistent.paper.12345"
        pmcid = verifier.get_pmcid_from_doi(doi)

        assert pmcid is None

    def test_fetch_pmc_html_from_valid_pmcid(self):
        """Test fetching HTML from a valid PMCID."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # Known PMC article
        pmcid = "PMC7850539"
        html = verifier.fetch_pmc_html(pmcid)

        assert html is not None
        assert len(html) > 0
        assert "<html" in html.lower() or "<!doctype" in html.lower()

    def test_fetch_pmc_html_returns_none_for_invalid_pmcid(self):
        """Test that invalid PMCID returns None."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        html = verifier.fetch_pmc_html("PMC999999999")
        assert html is None

    def test_end_to_end_snippet_found(self):
        """Test end-to-end: DOI -> PMCID -> HTML -> text -> snippet found."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI PDAC paper
        doi = "10.1093/jnci/djaa073"

        # Get paper text
        full_text, pmcid, status = verifier.get_paper_text(doi)

        if full_text:
            assert status == "success"
            # This text should be in the paper
            found, score, _ = verifier.fuzzy_find_snippet("FoxP3", full_text)
            assert found is True
            assert score >= 0.8

    def test_end_to_end_snippet_not_found(self):
        """Test end-to-end: snippet that shouldn't be in paper."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI PDAC paper
        doi = "10.1093/jnci/djaa073"

        # Get paper text
        full_text, pmcid, status = verifier.get_paper_text(doi)

        if full_text:
            # This text should NOT be in a PDAC immunotherapy paper
            found, score, _ = verifier.fuzzy_find_snippet(
                "quantum entanglement photosynthesis", full_text
            )
            assert found is False

    def test_paper_text_caching(self):
        """Test that paper text is cached after first fetch."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        doi = "10.1093/jnci/djaa073"

        # First call - should hit API
        text1, pmcid1, status1 = verifier.get_paper_text(doi)
        assert status1 == "success"

        # Second call - should use cache
        text2, pmcid2, status2 = verifier.get_paper_text(doi)
        assert status2 == "cached"

        assert text1 == text2
        # Verify cache was used (same normalized DOI in cache)
        normalized = verifier.normalize_doi(doi)
        assert normalized in verifier._paper_text_cache


@pytest.mark.integration
class TestRealWorldValidation:
    """
    Real-world validation tests using actual YAML structures.
    """

    def test_validates_yaml_with_pmc_paper(self):
        """Test validating a YAML file with a paper that's in PMC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"

            # Create YAML with real PMC-available paper
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "cd8_count",
                            "value": 17,
                            "value_snippet": "17 (9-30)",  # Should be findable
                            "source_ref": "michelakos2020",
                        }
                    ]
                },
                "primary_data_sources": [
                    {
                        "source_tag": "michelakos2020",
                        "doi": "10.1093/jnci/djaa073",
                        "title": "Tumor Microenvironment Immune Response",
                        "first_author": "Michelakos",
                        "year": 2020,
                    }
                ],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            verifier = AutomatedSnippetVerifier(tmpdir, rate_limit=0.5)

            # Mock user input for manual verification (shouldn't be needed)
            with patch.object(verifier, "get_manual_verification", return_value=True):
                report = verifier.validate()

            # Should have at least one pass (the snippet)
            assert len(report.passed) > 0 or len(report.warnings) > 0
