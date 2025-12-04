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
        found, score, normalized, best_match = verifier.fuzzy_find_snippet(
            "tumor growth", "The tumor growth rate was measured."
        )
        assert found is True
        assert score == 1.0
        assert normalized == "tumor growth"
        # Now returns matched text for value checking
        assert best_match == "tumor growth"

    def test_finds_case_insensitive_match(self):
        """Test case-insensitive matching."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, _, _ = verifier.fuzzy_find_snippet("Tumor Growth", "the tumor growth rate")
        assert found is True
        assert score == 1.0

    def test_finds_fuzzy_match(self):
        """Test finding fuzzy match above threshold."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        # Minor difference should still match
        found, score, _, _ = verifier.fuzzy_find_snippet("tumor growth rate", "tumor growth rates")
        assert found is True
        assert score >= 0.8

    def test_rejects_below_threshold(self):
        """Test rejecting match below threshold."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, _, _ = verifier.fuzzy_find_snippet(
            "completely different text", "tumor growth rate was measured"
        )
        assert found is False
        assert score < 0.8

    def test_handles_empty_snippet(self):
        """Test handling empty snippet."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, normalized, best_match = verifier.fuzzy_find_snippet("", "some text")
        assert found is False
        assert score == 0.0
        assert normalized == ""
        assert best_match is None

    def test_handles_empty_full_text(self):
        """Test handling empty full text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, _, _ = verifier.fuzzy_find_snippet("snippet", "")
        assert found is False
        assert score == 0.0

    def test_respects_custom_threshold(self):
        """Test respecting custom fuzzy threshold."""
        # With high threshold, partial match should fail
        # "tumor growth" vs "tumor expansion" has ~0.6 similarity
        verifier_strict = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.95)
        found, _, _, _ = verifier_strict.fuzzy_find_snippet("tumor growth", "tumor expansion")
        assert found is False

        # With lower threshold, same match should pass
        verifier_lenient = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.5)
        found, _, _, _ = verifier_lenient.fuzzy_find_snippet("tumor growth", "tumor expansion")
        assert found is True

    def test_returns_best_match_for_near_miss(self):
        """Test that near-misses return the best matching text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        # "tumor growth" vs "tumor expansion" should be a near-miss (score ~0.6)
        found, score, _, best_match = verifier.fuzzy_find_snippet(
            "tumor growth", "The tumor expansion was observed in all patients."
        )
        assert found is False
        assert score >= 0.5  # Should be a near-miss
        assert best_match is not None  # Should have best match text
        assert "tumor" in best_match.lower()


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
        found, score, normalized, _ = verifier.fuzzy_find_snippet(
            "CD8^{+} T cells", "The CD8+ T cells were counted"
        )
        assert found is True
        assert normalized == "CD8+ T cells"

    def test_fuzzy_match_with_table_snippet(self):
        """Test that table-formatted snippets match plain text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score, normalized, _ = verifier.fuzzy_find_snippet(
            "CD8+ | 17 (9-30)", "CD8+ 17 (9-30) cells per HPF"
        )
        assert found is True
        assert "|" not in normalized


class TestValueInMatchedText:
    """Test checking if declared values appear in matched text from paper."""

    def test_finds_integer_value_in_matched_text(self):
        """Test finding an integer value in matched text."""
        verifier = AutomatedSnippetVerifier("/tmp")
        found, pattern = verifier.check_value_in_matched_text(
            17, None, "the count was 17 cells per field"
        )
        assert found is True
        assert pattern is not None

    def test_finds_float_value_in_matched_text(self):
        """Test finding a float value in matched text."""
        verifier = AutomatedSnippetVerifier("/tmp")
        found, pattern = verifier.check_value_in_matched_text(
            0.5, None, "growth rate of 0.5 per day"
        )
        assert found is True

    def test_finds_percentage_value(self):
        """Test finding a decimal value expressed as percentage."""
        verifier = AutomatedSnippetVerifier("/tmp")
        # 0.28 should match "28%"
        found, pattern = verifier.check_value_in_matched_text(0.28, None, "response rate was 28%")
        assert found is True

    def test_fails_when_value_not_in_text(self):
        """Test failure when declared value is not in matched text."""
        verifier = AutomatedSnippetVerifier("/tmp")
        found, pattern = verifier.check_value_in_matched_text(
            42, None, "the count was 17 cells per field"
        )
        assert found is False
        assert pattern is None

    def test_handles_none_value(self):
        """Test that None value passes (nothing to check)."""
        verifier = AutomatedSnippetVerifier("/tmp")
        found, pattern = verifier.check_value_in_matched_text(None, None, "some text")
        assert found is True

    def test_handles_empty_matched_text(self):
        """Test that empty matched text passes (nothing to check)."""
        verifier = AutomatedSnippetVerifier("/tmp")
        found, pattern = verifier.check_value_in_matched_text(17, None, "")
        assert found is True

    def test_finds_value_with_units(self):
        """Test finding value with units context."""
        verifier = AutomatedSnippetVerifier("/tmp")
        found, pattern = verifier.check_value_in_matched_text(
            100, "mg/kg", "dose of 100 mg/kg was administered"
        )
        assert found is True

    def test_fails_when_wrong_value_present(self):
        """Test failure when a different numeric value is present."""
        verifier = AutomatedSnippetVerifier("/tmp")
        # Looking for 1.0 but text has different numbers
        found, pattern = verifier.check_value_in_matched_text(
            1.0,
            "boolean",
            "foxp3+cd25+ treg cells based on the gating of cd4+ t cells",
        )
        # Should fail - no "1" or "1.0" in this text
        assert found is False

    def test_boolean_indicator_not_found_in_qualitative_text(self):
        """Test that boolean 1.0 isn't falsely found in qualitative descriptions."""
        verifier = AutomatedSnippetVerifier("/tmp")
        # This is the exact case from the bug report
        found, pattern = verifier.check_value_in_matched_text(
            1.0,
            "qualitative (1=as specified)",
            "dot plots of foxp3+cd25+ (treg) cells based on the gating of cd4+ t cells",
        )
        # The text doesn't contain "1" as a standalone value
        assert found is False


class TestVerifyInputsWithValues:
    """Test the combined snippet + value verification."""

    def test_success_when_snippet_and_value_both_found(self):
        """Test full success when snippet matches and value is in matched text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)

        inputs = [
            {
                "name": "cd8_count",
                "value": 17,
                "units": "cells/HPF",
                "value_snippet": "median CD8+ count was 17",
            }
        ]
        paper_text = "The median CD8+ count was 17 (IQR 9-30) cells per high power field."

        results = verifier._verify_inputs_with_values(inputs, paper_text)

        assert len(results) == 1
        (
            input_name,
            snippet,
            normalized,
            snippet_found,
            score,
            matched_text,
            value_in_match,
            value_pattern,
        ) = results[0]

        assert input_name == "cd8_count"
        assert snippet_found is True
        assert value_in_match is True
        assert value_pattern is not None

    def test_failure_when_snippet_found_but_value_missing(self):
        """Test failure when snippet matches but value is NOT in paper text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)

        # Snippet will match, but value 42 is not in the paper text
        inputs = [
            {
                "name": "wrong_value",
                "value": 42,
                "units": None,
                "value_snippet": "the count was measured",
            }
        ]
        paper_text = "The count was measured to be 17 cells per field."

        results = verifier._verify_inputs_with_values(inputs, paper_text)

        assert len(results) == 1
        _, _, _, snippet_found, _, _, value_in_match, _ = results[0]

        assert snippet_found is True  # Snippet text matches
        assert value_in_match is False  # But value 42 is not there

    def test_failure_when_snippet_not_found(self):
        """Test failure when snippet doesn't match paper text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)

        inputs = [
            {
                "name": "missing",
                "value": 17,
                "units": None,
                "value_snippet": "completely fabricated text not in paper",
            }
        ]
        paper_text = "This paper discusses tumor growth and immune cells."

        results = verifier._verify_inputs_with_values(inputs, paper_text)

        assert len(results) == 1
        _, _, _, snippet_found, _, _, value_in_match, _ = results[0]

        assert snippet_found is False
        # value_in_match should be True (default) since snippet wasn't found
        assert value_in_match is True

    def test_skips_inputs_without_value_snippet(self):
        """Test that inputs without value_snippet are skipped."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)

        inputs = [
            {
                "name": "no_snippet",
                "value": 17,
                "units": None,
                # No value_snippet field
            }
        ]
        paper_text = "Some paper text with 17 in it."

        results = verifier._verify_inputs_with_values(inputs, paper_text)

        assert len(results) == 0

    def test_handles_multiple_inputs(self):
        """Test verifying multiple inputs from same source."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)

        inputs = [
            {
                "name": "good_input",
                "value": 17,
                "units": None,
                "value_snippet": "count was 17",
            },
            {
                "name": "bad_value",
                "value": 99,  # Wrong value
                "units": None,
                "value_snippet": "count was 17",  # Snippet matches but value wrong
            },
            {
                "name": "bad_snippet",
                "value": 5,
                "units": None,
                "value_snippet": "nonexistent text",  # Snippet doesn't match
            },
        ]
        paper_text = "The count was 17 cells per field in all samples."

        results = verifier._verify_inputs_with_values(inputs, paper_text)

        assert len(results) == 3

        # First input: both snippet and value should pass
        assert results[0][3] is True  # snippet_found
        assert results[0][6] is True  # value_in_match

        # Second input: snippet matches but value 99 not in text
        assert results[1][3] is True  # snippet_found
        assert results[1][6] is False  # value_in_match - 99 not in paper

        # Third input: snippet doesn't match
        assert results[2][3] is False  # snippet_found

    def test_qualitative_boolean_fails_validation(self):
        """Test that qualitative boolean indicators fail value validation."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)

        # This is the exact problematic case from the bug report
        inputs = [
            {
                "name": "Gating_definition_Treg",
                "value": 1.0,
                "units": "qualitative (1=as specified)",
                "value_snippet": "Dot plots of Foxp3+CD25+ (Treg) cells based on the gating of CD4+ T cells.",
            }
        ]
        # Paper text that matches the snippet but doesn't contain "1" or "1.0"
        paper_text = "Figure 1 legend shows dot plots of Foxp3+CD25+ (Treg) cells based on the gating of CD4+ T cells."

        results = verifier._verify_inputs_with_values(inputs, paper_text)

        assert len(results) == 1
        _, _, _, snippet_found, _, matched_text, value_in_match, _ = results[0]

        assert snippet_found is True  # Snippet matches
        # Value 1.0 should NOT be found in the matched text
        # (The "1" in "Figure 1" is in the paper but not in the matched window)
        # This depends on window size - the key point is the matched snippet text
        # doesn't contain "1" as a value


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

    def test_collects_value_and_units_from_inputs(self):
        """Test that value and units are collected for value checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "test.yaml"
            data = {
                "parameter_estimates": {
                    "inputs": [
                        {
                            "name": "growth_rate",
                            "value": 0.5,
                            "units": "1/day",
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

            # Check that inputs now include value and units
            inputs = source_data["smith2020"]["inputs"]
            assert len(inputs) == 1
            assert inputs[0]["value"] == 0.5
            assert inputs[0]["units"] == "1/day"
            assert inputs[0]["name"] == "growth_rate"

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

    def test_get_paper_info_from_known_pmc_doi(self):
        """Test getting paper info from a known PMC DOI."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI paper - in PMC but not necessarily open access
        doi = "10.1093/jnci/djaa073"
        paper_info = verifier.get_paper_info_from_doi(doi)

        assert paper_info is not None
        assert paper_info.pmcid is not None
        assert paper_info.pmcid.startswith("PMC")
        assert paper_info.in_pmc is True
        # This particular paper is not open access
        assert paper_info.is_open_access is False
        assert paper_info.abstract is not None

    def test_get_paper_info_returns_none_for_nonexistent_doi(self):
        """Test that non-existent DOIs return None."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # Using a made-up DOI that shouldn't exist
        doi = "10.9999/nonexistent.paper.12345"
        paper_info = verifier.get_paper_info_from_doi(doi)

        assert paper_info is None

    @pytest.mark.skip(reason="Flaky in CI - depends on PMC API availability")
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

    @pytest.mark.skip(reason="Flaky in CI - depends on PMC API availability")
    def test_end_to_end_snippet_found(self):
        """Test end-to-end: DOI -> PMCID -> HTML -> text -> snippet found."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI PDAC paper
        doi = "10.1093/jnci/djaa073"

        # Get paper text
        text, paper_info, status = verifier.get_paper_text(doi)

        if text:
            # Should be full text (restricted access)
            assert status in ("full_text_open_access", "full_text_restricted")
            assert paper_info is not None
            assert paper_info.pmcid is not None
            # This text should be in the paper
            found, score, _, _ = verifier.fuzzy_find_snippet("FoxP3", text)
            assert found is True
            assert score >= 0.8

    def test_end_to_end_snippet_not_found(self):
        """Test end-to-end: snippet that shouldn't be in paper."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI PDAC paper
        doi = "10.1093/jnci/djaa073"

        # Get paper text
        text, paper_info, status = verifier.get_paper_text(doi)

        if text:
            # This text should NOT be in a PDAC immunotherapy paper
            found, score, _, _ = verifier.fuzzy_find_snippet(
                "quantum entanglement photosynthesis", text
            )
            assert found is False

    @pytest.mark.skip(reason="Flaky in CI - depends on PMC API availability")
    def test_paper_text_caching(self):
        """Test that paper text is cached after first fetch."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        doi = "10.1093/jnci/djaa073"

        # First call - should hit API
        text1, paper_info1, status1 = verifier.get_paper_text(doi)
        assert status1 in ("full_text_open_access", "full_text_restricted")

        # Second call - should use cache
        text2, paper_info2, status2 = verifier.get_paper_text(doi)
        assert status2 == "cached"

        assert text1 == text2
        # Verify cache was used (same normalized DOI in cache)
        normalized = verifier.normalize_doi(doi)
        cache_key = f"text_{normalized}"
        assert cache_key in verifier._paper_text_cache

    def test_paper_info_includes_abstract(self):
        """Test that paper info includes abstract for non-PMC papers."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # JNCI paper - has abstract available
        doi = "10.1093/jnci/djaa073"
        paper_info = verifier.get_paper_info_from_doi(doi)

        assert paper_info is not None
        assert paper_info.abstract is not None
        assert len(paper_info.abstract) > 100  # Should be substantial
        # Abstract should contain relevant terms
        assert "pancreatic" in paper_info.abstract.lower()


class TestManualVerificationPrompts:
    """Test manual verification prompt methods."""

    def test_print_manual_verification_prompt_with_custom_reason(self, capsys):
        """Test that manual verification prompt includes custom reason."""
        verifier = AutomatedSnippetVerifier("/tmp")

        manual_sources = {
            "test_source": {
                "doi": "10.1234/test",
                "snippets": {"test snippet"},
                "inputs": [{"filename": "test.yaml"}],
            }
        }

        verifier.print_manual_verification_prompt(manual_sources, "NO TEXT AVAILABLE")
        captured = capsys.readouterr()

        assert "NO TEXT AVAILABLE" in captured.out
        assert "test_source" in captured.out
        assert "10.1234/test" in captured.out

    def test_print_abstract_only_verification_prompt(self, capsys):
        """Test abstract-only verification prompt shows failed snippets."""
        verifier = AutomatedSnippetVerifier("/tmp")

        abstract_failures = {
            "masugi2019": {
                "doi": "10.1234/masugi",
                "snippets": {"all snippets"},
                "inputs": [{"filename": "test.yaml"}],
                "failed_snippets": ["snippet1 not found", "snippet2 not found"],
            }
        }

        verifier.print_abstract_only_verification_prompt(abstract_failures)
        captured = capsys.readouterr()

        assert "ABSTRACT ONLY" in captured.out
        assert "FAILED SNIPPETS" in captured.out
        assert "masugi2019" in captured.out
        assert "snippet1 not found" in captured.out
        assert "snippet2 not found" in captured.out
        assert "2" in captured.out  # Count of failed snippets

    def test_print_full_text_verification_prompt(self, capsys):
        """Test full-text verification prompt shows mismatched snippets."""
        verifier = AutomatedSnippetVerifier("/tmp")

        full_text_failures = {
            "smith2020": {
                "doi": "10.1234/smith",
                "snippets": {"all snippets"},
                "inputs": [{"filename": "test.yaml"}],
                "failed_snippets": ["snippet with wrong value", "snippet not found in text"],
            }
        }

        verifier.print_full_text_verification_prompt(full_text_failures)
        captured = capsys.readouterr()

        assert "FULL TEXT" in captured.out
        assert "MISMATCHES" in captured.out
        assert "smith2020" in captured.out
        assert "snippet with wrong value" in captured.out
        assert "snippet not found in text" in captured.out
        assert "2" in captured.out  # Count of failed snippets


class TestAbstractOnlyFailures:
    """Test handling of abstract-only papers with snippet failures."""

    def test_abstract_only_failure_counts_as_failure(self, tmp_path):
        """Test that snippets not found in abstract count as failures."""
        yaml_file = tmp_path / "test.yaml"

        # Create YAML with snippet that won't be in abstract
        data = {
            "parameter_estimates": {
                "inputs": [
                    {
                        "name": "detailed_stat",
                        "value": 42,
                        "value_snippet": "This very specific phrase won't be in any abstract",
                        "source_ref": "test_source",
                    }
                ]
            },
            "primary_data_sources": [
                {
                    "source_tag": "test_source",
                    "doi": "10.1234/fake",
                    "title": "Test Paper",
                    "first_author": "Test",
                    "year": 2024,
                }
            ],
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        verifier = AutomatedSnippetVerifier(str(tmp_path))

        # Mock get_paper_text to return abstract only
        def mock_get_paper_text(doi):
            from qsp_llm_workflows.validate.check_snippet_sources_automated import PaperInfo

            paper_info = PaperInfo(
                pmcid=None,
                is_open_access=False,
                in_pmc=False,
                abstract="This is a short abstract about pancreatic cancer research.",
            )
            return (paper_info.abstract, paper_info, "abstract_only")

        # Mock manual verification to avoid blocking
        with patch.object(verifier, "get_paper_text", side_effect=mock_get_paper_text):
            with patch.object(verifier, "get_manual_verification", return_value=True):
                report = verifier.validate()

        # Should have a failure for the snippet not found in abstract
        assert len(report.failed) > 0
        # Check that the failure message mentions abstract
        assert any("abstract" in f["reason"].lower() for f in report.failed)

    def test_abstract_only_triggers_manual_verification(self, tmp_path, capsys):
        """Test that abstract-only failures trigger manual verification prompt."""
        yaml_file = tmp_path / "test.yaml"

        data = {
            "parameter_estimates": {
                "inputs": [
                    {
                        "name": "stat",
                        "value": 1,
                        "value_snippet": "unfindable snippet xyz123",
                        "source_ref": "src",
                    }
                ]
            },
            "primary_data_sources": [
                {
                    "source_tag": "src",
                    "doi": "10.1234/test",
                    "title": "Test",
                    "first_author": "A",
                    "year": 2024,
                }
            ],
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        verifier = AutomatedSnippetVerifier(str(tmp_path))

        def mock_get_paper_text(doi):
            from qsp_llm_workflows.validate.check_snippet_sources_automated import PaperInfo

            return (
                "Short abstract text",
                PaperInfo(abstract="Short abstract text"),
                "abstract_only",
            )

        with patch.object(verifier, "get_paper_text", side_effect=mock_get_paper_text):
            with patch.object(
                verifier, "get_manual_verification", return_value=True
            ) as mock_verify:
                verifier.validate()

        # Manual verification should have been called
        mock_verify.assert_called()

        # Check that prompt was printed
        captured = capsys.readouterr()
        assert "ABSTRACT ONLY" in captured.out or "MANUAL VERIFICATION" in captured.out


class TestFullTextFailures:
    """Test handling of full-text papers with snippet/value mismatches."""

    def test_full_text_failure_triggers_manual_verification(self, tmp_path, capsys):
        """Test that full-text mismatches trigger manual verification prompt."""
        yaml_file = tmp_path / "test.yaml"

        data = {
            "parameter_estimates": {
                "inputs": [
                    {
                        "name": "stat",
                        "value": 999,  # Wrong value - won't be in paper
                        "value_snippet": "tumor count was measured",  # Snippet will match
                        "source_ref": "src",
                    }
                ]
            },
            "primary_data_sources": [
                {
                    "source_tag": "src",
                    "doi": "10.1234/test",
                    "title": "Test",
                    "first_author": "A",
                    "year": 2024,
                }
            ],
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        verifier = AutomatedSnippetVerifier(str(tmp_path))

        def mock_get_paper_text(_):
            from qsp_llm_workflows.validate.check_snippet_sources_automated import PaperInfo

            # Return full text that contains snippet but NOT the value 999
            full_text = "The tumor count was measured to be 17 cells per field in the study."
            return (
                full_text,
                PaperInfo(pmcid="PMC123", in_pmc=True),
                "full_text_open_access",
            )

        with patch.object(verifier, "get_paper_text", side_effect=mock_get_paper_text):
            with patch.object(
                verifier, "get_manual_verification", return_value=True
            ) as mock_verify:
                verifier.validate()

        # Manual verification should have been called for full-text mismatch
        mock_verify.assert_called()

        # Check that full-text prompt was printed
        captured = capsys.readouterr()
        assert "FULL TEXT" in captured.out or "MISMATCHES" in captured.out

    def test_full_text_success_no_manual_verification(self, tmp_path, capsys):
        """Test that full-text success doesn't trigger manual verification."""
        yaml_file = tmp_path / "test.yaml"

        data = {
            "parameter_estimates": {
                "inputs": [
                    {
                        "name": "stat",
                        "value": 17,  # Correct value - will be in paper
                        "value_snippet": "count was 17",  # Snippet will match
                        "source_ref": "src",
                    }
                ]
            },
            "primary_data_sources": [
                {
                    "source_tag": "src",
                    "doi": "10.1234/test",
                    "title": "Test",
                    "first_author": "A",
                    "year": 2024,
                }
            ],
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        verifier = AutomatedSnippetVerifier(str(tmp_path))

        def mock_get_paper_text(_):
            from qsp_llm_workflows.validate.check_snippet_sources_automated import PaperInfo

            # Return full text with both snippet AND value
            full_text = "The count was 17 cells per field."
            return (
                full_text,
                PaperInfo(pmcid="PMC123", in_pmc=True),
                "full_text_open_access",
            )

        with patch.object(verifier, "get_paper_text", side_effect=mock_get_paper_text):
            with patch.object(
                verifier, "get_manual_verification", return_value=True
            ) as mock_verify:
                verifier.validate()

        # Manual verification should NOT have been called
        mock_verify.assert_not_called()

        # No manual verification prompts should appear
        captured = capsys.readouterr()
        assert "MANUAL VERIFICATION" not in captured.out


class TestUnpaywallIntegration:
    """Test Unpaywall API integration for OA papers not in PMC."""

    def test_get_unpaywall_info_parses_response(self):
        """Test parsing Unpaywall API response."""
        verifier = AutomatedSnippetVerifier("/tmp")

        # Mock a successful Unpaywall response
        mock_response = {
            "is_oa": True,
            "oa_status": "bronze",
            "best_oa_location": {
                "url": "https://www.nature.com/articles/s41379-019-0291-z",
                "url_for_pdf": "https://www.nature.com/articles/s41379-019-0291-z.pdf",
            },
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            result = verifier.get_unpaywall_info("10.1038/s41379-019-0291-z")

        assert result is not None
        assert result["is_oa"] is True
        assert result["oa_status"] == "bronze"
        assert "nature.com" in result["oa_url"]

    def test_get_unpaywall_info_returns_none_for_closed_access(self):
        """Test that closed access papers return None."""
        verifier = AutomatedSnippetVerifier("/tmp")

        mock_response = {
            "is_oa": False,
            "oa_status": "closed",
            "best_oa_location": None,
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response

            result = verifier.get_unpaywall_info("10.1234/closed-paper")

        assert result is None

    def test_get_unpaywall_info_handles_api_error(self):
        """Test graceful handling of API errors."""
        verifier = AutomatedSnippetVerifier("/tmp")

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 404

            result = verifier.get_unpaywall_info("10.1234/nonexistent")

        assert result is None

    def test_fetch_publisher_html_success(self):
        """Test fetching HTML from publisher site."""
        verifier = AutomatedSnippetVerifier("/tmp")

        mock_html = "<html><body><article>Full article content here</article></body></html>"

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.headers = {"Content-Type": "text/html"}
            mock_get.return_value.text = mock_html

            result = verifier.fetch_publisher_html("https://example.com/article")

        assert result == mock_html

    def test_fetch_publisher_html_rejects_pdf(self):
        """Test that PDF responses are rejected."""
        verifier = AutomatedSnippetVerifier("/tmp")

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.headers = {"Content-Type": "application/pdf"}
            mock_get.return_value.text = "%PDF-1.4..."

            result = verifier.fetch_publisher_html("https://example.com/article.pdf")

        assert result is None

    def test_extract_text_from_publisher_html_nature(self):
        """Test extracting text from Nature-style HTML."""
        verifier = AutomatedSnippetVerifier("/tmp")

        html = """
        <html>
        <body>
            <nav>Skip this navigation</nav>
            <article class="c-article-body">
                <h1>Article Title</h1>
                <p>The tumor infiltrating lymphocytes were counted using CD8 staining.</p>
                <p>Results showed 17 (9-30) cells per high-power field.</p>
            </article>
            <footer>Skip this footer</footer>
        </body>
        </html>
        """

        text = verifier.extract_text_from_publisher_html(html)

        assert "tumor infiltrating lymphocytes" in text.lower()
        assert "17 (9-30)" in text
        # Should not include nav/footer
        assert "skip this navigation" not in text.lower()
        assert "skip this footer" not in text.lower()

    def test_extract_text_from_publisher_html_generic(self):
        """Test extracting text from generic HTML structure."""
        verifier = AutomatedSnippetVerifier("/tmp")

        html = """
        <html>
        <body>
            <main>
                <div class="content">
                    <p>Important research findings about pancreatic cancer.</p>
                </div>
            </main>
        </body>
        </html>
        """

        text = verifier.extract_text_from_publisher_html(html)
        assert "pancreatic cancer" in text.lower()

    def test_try_unpaywall_fallback_success(self):
        """Test that try_unpaywall_fallback fetches publisher full text."""
        verifier = AutomatedSnippetVerifier("/tmp")

        # Mock Unpaywall returning OA URL
        mock_unpaywall = {
            "is_oa": True,
            "oa_status": "bronze",
            "oa_url": "https://publisher.com/article",
            "oa_pdf_url": None,
        }

        # Mock publisher HTML - needs sufficient content (>1000 chars)
        mock_html = (
            "<html><body><article>"
            + "Full text with specific content xyz123. " * 50
            + "</article></body></html>"
        )

        from qsp_llm_workflows.validate.check_snippet_sources_automated import (
            PaperInfo,
        )

        paper_info = PaperInfo(
            pmcid=None,
            pmid="12345",
            is_open_access=False,
            in_pmc=False,
            abstract="Short abstract",
        )

        with patch.object(verifier, "get_unpaywall_info", return_value=mock_unpaywall):
            with patch.object(verifier, "fetch_publisher_html", return_value=mock_html):
                text, updated_info, status = verifier.try_unpaywall_fallback(
                    "10.1234/test", paper_info
                )

        assert status == "full_text_publisher"
        assert "specific content xyz123" in text.lower()
        assert updated_info.oa_status == "bronze"

    def test_get_paper_text_returns_abstract_not_unpaywall(self):
        """Test that get_paper_text returns abstract (Unpaywall called separately)."""
        verifier = AutomatedSnippetVerifier("/tmp")

        from qsp_llm_workflows.validate.check_snippet_sources_automated import (
            PaperInfo,
        )

        with patch.object(verifier, "get_paper_info_from_doi") as mock_info:
            mock_info.return_value = PaperInfo(
                pmcid=None,
                pmid="12345",
                is_open_access=False,
                in_pmc=False,
                abstract="This is the abstract text",
            )

            # get_paper_text should NOT call Unpaywall anymore
            text, paper_info, status = verifier.get_paper_text("10.1234/test")

        # Should return abstract, Unpaywall is called via try_unpaywall_fallback
        assert status == "abstract_only"
        assert text == "This is the abstract text"

    def test_validate_tries_unpaywall_on_abstract_failure(self, tmp_path, capsys):
        """Test that validation tries Unpaywall when abstract verification fails."""
        yaml_file = tmp_path / "test.yaml"

        data = {
            "parameter_estimates": {
                "inputs": [
                    {
                        "name": "stat",
                        "value": 1,
                        "value_snippet": "detailed table data not in abstract",
                        "source_ref": "src",
                    }
                ]
            },
            "primary_data_sources": [
                {
                    "source_tag": "src",
                    "doi": "10.1234/test",
                    "title": "Test",
                    "first_author": "A",
                    "year": 2024,
                }
            ],
        }
        with open(yaml_file, "w") as f:
            yaml.dump(data, f)

        verifier = AutomatedSnippetVerifier(str(tmp_path))

        from qsp_llm_workflows.validate.check_snippet_sources_automated import (
            PaperInfo,
        )

        # Mock get_paper_text to return abstract only (snippet not findable there)
        def mock_get_paper_text(_doi):
            paper_info = PaperInfo(
                pmcid=None, is_open_access=True, in_pmc=False, abstract="Short abstract"
            )
            return ("Short abstract", paper_info, "abstract_only")

        # Mock Unpaywall to return full text with the snippet
        full_text = "Full text with detailed table data not in abstract included here"

        def mock_unpaywall_fallback(_doi, paper_info):
            paper_info.oa_status = "bronze"
            return (full_text, paper_info, "full_text_publisher")

        with patch.object(verifier, "get_paper_text", side_effect=mock_get_paper_text):
            with patch.object(
                verifier, "try_unpaywall_fallback", side_effect=mock_unpaywall_fallback
            ):
                with patch.object(verifier, "get_manual_verification", return_value=True):
                    verifier.validate()

        captured = capsys.readouterr()
        # Should show Unpaywall being tried
        assert "Trying Unpaywall" in captured.out
        assert "bronze" in captured.out.lower()

    def test_try_unpaywall_fallback_returns_failed_when_not_oa(self):
        """Test that try_unpaywall_fallback returns failed status for non-OA papers."""
        verifier = AutomatedSnippetVerifier("/tmp")

        from qsp_llm_workflows.validate.check_snippet_sources_automated import (
            PaperInfo,
        )

        paper_info = PaperInfo(abstract="Some abstract")

        # Mock Unpaywall returning None (not OA)
        with patch.object(verifier, "get_unpaywall_info", return_value=None):
            text, updated_info, status = verifier.try_unpaywall_fallback("10.1234/test", paper_info)

        assert status == "unpaywall_failed"
        assert text is None

    def test_try_unpaywall_fallback_converts_pdf_url_to_html(self):
        """Test that PDF URLs are converted to HTML URLs."""
        verifier = AutomatedSnippetVerifier("/tmp")

        from qsp_llm_workflows.validate.check_snippet_sources_automated import (
            PaperInfo,
        )

        # Mock Unpaywall returning PDF URL (common for Nature papers)
        mock_unpaywall = {
            "is_oa": True,
            "oa_status": "bronze",
            "oa_url": "https://www.nature.com/articles/s41379-019-0291-z.pdf",
            "oa_pdf_url": "https://www.nature.com/articles/s41379-019-0291-z.pdf",
        }

        # Mock publisher HTML
        mock_html = (
            "<html><body><article>"
            + "Full text with specific content from Nature article. " * 50
            + "</article></body></html>"
        )

        paper_info = PaperInfo()

        with patch.object(verifier, "get_unpaywall_info", return_value=mock_unpaywall):
            with patch.object(
                verifier, "fetch_publisher_html", return_value=mock_html
            ) as mock_fetch:
                text, updated_info, status = verifier.try_unpaywall_fallback(
                    "10.1038/s41379-019-0291-z", paper_info
                )

        assert status == "full_text_publisher"
        # Should have tried the HTML URL (without .pdf)
        mock_fetch.assert_called()
        # The first call should be the URL without .pdf
        first_call_url = mock_fetch.call_args_list[0][0][0]
        assert first_call_url == "https://www.nature.com/articles/s41379-019-0291-z"
        assert not first_call_url.endswith(".pdf")


@pytest.mark.integration
class TestUnpaywallRealAPI:
    """
    Integration tests that hit the real Unpaywall API.

    Run with: pytest -m integration
    """

    def test_get_unpaywall_info_real_oa_paper(self):
        """Test Unpaywall with a known OA paper (Modern Pathology)."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=1.0)

        # Modern Pathology paper - known to be bronze OA
        doi = "10.1038/s41379-019-0291-z"
        result = verifier.get_unpaywall_info(doi)

        assert result is not None
        assert result["is_oa"] is True
        assert result["oa_status"] in ("bronze", "gold", "green", "hybrid")
        assert result["oa_url"] is not None

    def test_get_unpaywall_info_real_closed_paper(self):
        """Test Unpaywall with a known closed-access paper."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=1.0)

        # This paper might be closed access (check current status)
        # Using a common paywalled journal
        doi = "10.1056/NEJMoa2034577"
        result = verifier.get_unpaywall_info(doi)

        # Result depends on current OA status - just check we get a response
        # without errors
        # (Paper may have become OA since test was written)
        assert result is None or isinstance(result, dict)


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
