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


class TestXMLParsing:
    """Test XML parsing and text extraction."""

    def test_extracts_text_from_simple_xml(self):
        """Test extracting text from simple XML."""
        verifier = AutomatedSnippetVerifier("/tmp")
        xml = "<article><body><p>Hello world</p></body></article>"
        text = verifier.extract_text_from_xml(xml)
        assert "hello world" in text.lower()

    def test_extracts_text_from_nested_xml(self):
        """Test extracting text from nested XML elements."""
        verifier = AutomatedSnippetVerifier("/tmp")
        xml = """
        <article>
            <body>
                <sec><title>Introduction</title>
                    <p>This is the introduction.</p>
                </sec>
                <sec><title>Methods</title>
                    <p>These are the methods.</p>
                </sec>
            </body>
        </article>
        """
        text = verifier.extract_text_from_xml(xml)
        assert "introduction" in text.lower()
        assert "methods" in text.lower()

    def test_handles_empty_xml(self):
        """Test handling empty XML."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.extract_text_from_xml("") == ""
        assert verifier.extract_text_from_xml(None) == ""

    def test_handles_invalid_xml(self):
        """Test handling invalid XML gracefully."""
        verifier = AutomatedSnippetVerifier("/tmp")
        assert verifier.extract_text_from_xml("<invalid>xml") == ""


class TestFuzzyMatching:
    """Test fuzzy snippet matching logic."""

    def test_finds_exact_match(self):
        """Test finding exact substring match."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score = verifier.fuzzy_find_snippet(
            "tumor growth", "The tumor growth rate was measured."
        )
        assert found is True
        assert score == 1.0

    def test_finds_case_insensitive_match(self):
        """Test case-insensitive matching."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score = verifier.fuzzy_find_snippet("Tumor Growth", "the tumor growth rate")
        assert found is True
        assert score == 1.0

    def test_finds_fuzzy_match(self):
        """Test finding fuzzy match above threshold."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        # Minor difference should still match
        found, score = verifier.fuzzy_find_snippet("tumor growth rate", "tumor growth rates")
        assert found is True
        assert score >= 0.8

    def test_rejects_below_threshold(self):
        """Test rejecting match below threshold."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score = verifier.fuzzy_find_snippet(
            "completely different text", "tumor growth rate was measured"
        )
        assert found is False
        assert score < 0.8

    def test_handles_empty_snippet(self):
        """Test handling empty snippet."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score = verifier.fuzzy_find_snippet("", "some text")
        assert found is False
        assert score == 0.0

    def test_handles_empty_full_text(self):
        """Test handling empty full text."""
        verifier = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.8)
        found, score = verifier.fuzzy_find_snippet("snippet", "")
        assert found is False
        assert score == 0.0

    def test_respects_custom_threshold(self):
        """Test respecting custom fuzzy threshold."""
        # With high threshold, partial match should fail
        # "tumor growth" vs "tumor expansion" has ~0.6 similarity
        verifier_strict = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.95)
        found, _ = verifier_strict.fuzzy_find_snippet("tumor growth", "tumor expansion")
        assert found is False

        # With lower threshold, same match should pass
        verifier_lenient = AutomatedSnippetVerifier("/tmp", fuzzy_threshold=0.5)
        found, _ = verifier_lenient.fuzzy_find_snippet("tumor growth", "tumor expansion")
        assert found is True


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
# INTEGRATION TESTS - These hit the real Europe PMC API
# =============================================================================


@pytest.mark.integration
class TestEuropePMCIntegration:
    """
    Integration tests that hit the real Europe PMC API.

    These tests verify the actual API integration works correctly.
    Run with: pytest -m integration
    Skip with: pytest -m "not integration"
    """

    def test_get_pmcid_from_known_oa_doi(self):
        """Test getting PMCID from a known open-access DOI."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # PLoS ONE paper - guaranteed open access
        doi = "10.1371/journal.pone.0035625"
        pmcid = verifier.get_pmcid_from_doi(doi)

        assert pmcid is not None
        assert pmcid.startswith("PMC")

    def test_get_pmcid_returns_none_for_non_pmc_doi(self):
        """Test that non-PMC DOIs return None."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # Nature paper that may not be in PMC
        # Using a made-up DOI that shouldn't exist
        doi = "10.9999/nonexistent.paper.12345"
        pmcid = verifier.get_pmcid_from_doi(doi)

        assert pmcid is None

    def test_fetch_full_text_xml_from_valid_pmcid(self):
        """Test fetching full text XML from a valid PMCID."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # First get a PMCID from a known OA paper
        doi = "10.1371/journal.pone.0035625"
        pmcid = verifier.get_pmcid_from_doi(doi)

        if pmcid:
            xml = verifier.fetch_full_text_xml(pmcid)
            assert xml is not None
            assert len(xml) > 0
            assert "<article" in xml or "<?xml" in xml

    def test_fetch_full_text_xml_returns_none_for_invalid_pmcid(self):
        """Test that invalid PMCID returns None."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        xml = verifier.fetch_full_text_xml("PMC999999999")
        assert xml is None

    def test_end_to_end_snippet_found(self):
        """Test end-to-end: DOI -> PMCID -> XML -> text -> snippet found."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # PLoS ONE paper about Cambrian fossils
        doi = "10.1371/journal.pone.0035625"

        # Get paper text
        full_text, pmcid = verifier.get_paper_text(doi)

        if full_text:
            # This text should be in the paper
            found, score = verifier.fuzzy_find_snippet("Cambrian", full_text)
            assert found is True
            assert score >= 0.8

    def test_end_to_end_snippet_not_found(self):
        """Test end-to-end: snippet that shouldn't be in paper."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        # PLoS ONE paper about Cambrian fossils
        doi = "10.1371/journal.pone.0035625"

        # Get paper text
        full_text, pmcid = verifier.get_paper_text(doi)

        if full_text:
            # This text should NOT be in a paleontology paper
            found, score = verifier.fuzzy_find_snippet(
                "checkpoint inhibitor immunotherapy", full_text
            )
            assert found is False

    def test_paper_text_caching(self):
        """Test that paper text is cached after first fetch."""
        verifier = AutomatedSnippetVerifier("/tmp", rate_limit=0.5)

        doi = "10.1371/journal.pone.0035625"

        # First call - should hit API
        text1, pmcid1 = verifier.get_paper_text(doi)

        # Second call - should use cache
        text2, pmcid2 = verifier.get_paper_text(doi)

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
                            "name": "fossil_age",
                            "value": 500,
                            "value_snippet": "Cambrian",  # Should be findable
                            "source_ref": "stein2012",
                        }
                    ]
                },
                "primary_data_sources": [
                    {
                        "source_tag": "stein2012",
                        "doi": "10.1371/journal.pone.0035625",
                        "title": "A New Arthropod",
                        "first_author": "Stein",
                        "year": 2012,
                    }
                ],
            }
            with open(yaml_file, "w") as f:
                yaml.dump(data, f)

            verifier = AutomatedSnippetVerifier(tmpdir, rate_limit=0.5)

            # Mock user input for manual verification (shouldn't be needed)
            with patch.object(verifier, "get_manual_verification", return_value=True):
                report = verifier.validate()

            # Should have at least one pass (the Cambrian snippet)
            assert len(report.passed) > 0 or len(report.warnings) > 0
