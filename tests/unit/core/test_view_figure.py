#!/usr/bin/env python3
"""
Tests for the view_figure tool.

Tests HTML parsing, figure extraction, and fuzzy label matching.
"""


from maple.core.tools.view_figure import (
    _normalize_label,
    _label_matches,
    extract_figures_from_html,
    find_figure,
)


# ============================================================================
# Sample HTML fixtures
# ============================================================================

PMC_FIGURE_HTML = """
<html>
<body>
<div class="fig" id="fig2">
  <label>Fig. 2A</label>
  <div class="caption">
    <p>CD8+ T cell infiltration across patient cohort.</p>
  </div>
  <img src="/pmc/articles/PMC12345/bin/nihms-fig2a.jpg"
       alt="Fig. 2A CD8 infiltration" />
</div>
<div class="fig" id="fig3">
  <label>Figure 3</label>
  <div class="caption">
    <p>Survival analysis by treatment arm.</p>
  </div>
  <img src="/pmc/articles/PMC12345/bin/nihms-fig3.jpg"
       alt="Figure 3 Survival" />
</div>
</body>
</html>
"""

FIGURE_TAG_HTML = """
<html>
<body>
<figure id="f1">
  <img src="/images/fig1.png" alt="Figure 1" />
  <figcaption>Figure 1. Tumor growth curves for all patients.</figcaption>
</figure>
<figure id="f2">
  <img src="/images/fig2.png" alt="Figure 2" />
  <figcaption>Figure 2. Immune cell densities in resected tumors.</figcaption>
</figure>
</body>
</html>
"""

NO_FIGURES_HTML = """
<html>
<body>
<h1>Abstract</h1>
<p>No figures on this page.</p>
</body>
</html>
"""


# ============================================================================
# Label normalization tests
# ============================================================================


class TestLabelNormalization:
    """Tests for figure label normalization and matching."""

    def test_normalize_figure_to_fig(self):
        assert _normalize_label("Figure 2A") == "fig 2a"

    def test_normalize_fig_dot(self):
        assert _normalize_label("Fig. 2A") == "fig 2a"

    def test_normalize_fig_no_dot(self):
        assert _normalize_label("Fig 2A") == "fig 2a"

    def test_normalize_case_insensitive(self):
        assert _normalize_label("FIGURE 2A") == "fig 2a"

    def test_normalize_extra_spaces(self):
        assert _normalize_label("Figure  2A") == "fig 2a"

    def test_label_matches_figure_vs_fig(self):
        assert _label_matches("Fig. 2A CD8 infiltration", "Figure 2A")

    def test_label_matches_exact(self):
        assert _label_matches("Figure 2A", "Figure 2A")

    def test_label_matches_case_insensitive(self):
        assert _label_matches("fig. 2a", "Figure 2A")

    def test_label_no_match(self):
        assert not _label_matches("Figure 3B", "Figure 2A")


# ============================================================================
# HTML parsing tests
# ============================================================================


class TestFigureExtraction:
    """Tests for HTML figure extraction."""

    def test_pmc_div_figures(self):
        """Extract figures from PMC-style <div class='fig'> elements."""
        figures = extract_figures_from_html(
            PMC_FIGURE_HTML, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/"
        )
        assert len(figures) == 2
        assert "fig2a" in figures[0]["img_url"].lower() or "fig2a" in figures[0]["img_url"]
        assert figures[0]["img_url"].startswith("https://")

    def test_standard_figure_tags(self):
        """Extract figures from standard <figure> tags."""
        figures = extract_figures_from_html(FIGURE_TAG_HTML, "https://example.com/paper/")
        assert len(figures) == 2
        assert "fig1.png" in figures[0]["img_url"]

    def test_no_figures(self):
        """Return empty list when no figures found."""
        figures = extract_figures_from_html(NO_FIGURES_HTML, "https://example.com/")
        assert figures == []

    def test_relative_url_resolution(self):
        """Relative image URLs are resolved against base URL."""
        figures = extract_figures_from_html(
            PMC_FIGURE_HTML, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/"
        )
        for fig in figures:
            assert fig["img_url"].startswith("https://")


# ============================================================================
# Figure matching tests
# ============================================================================


class TestFigureMatching:
    """Tests for find_figure matching logic."""

    def test_find_exact_match(self):
        """Find figure by exact label match."""
        figures = extract_figures_from_html(
            PMC_FIGURE_HTML, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/"
        )
        match = find_figure(figures, "Fig. 2A")
        assert match is not None
        assert "fig2a" in match["img_url"].lower() or "fig2a" in match["img_url"]

    def test_find_fuzzy_match(self):
        """Find figure with different label format (Figure vs Fig.)."""
        figures = extract_figures_from_html(
            PMC_FIGURE_HTML, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/"
        )
        match = find_figure(figures, "Figure 2A")
        assert match is not None

    def test_find_by_number_fallback(self):
        """Fall back to matching just the number part."""
        figures = extract_figures_from_html(FIGURE_TAG_HTML, "https://example.com/paper/")
        match = find_figure(figures, "Supplementary Figure 2")
        # Should match Figure 2 via number fallback
        assert match is not None
        assert "fig2.png" in match["img_url"]

    def test_no_match_returns_none(self):
        """Return None when no figure matches."""
        figures = extract_figures_from_html(
            PMC_FIGURE_HTML, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/"
        )
        match = find_figure(figures, "Figure 99")
        assert match is None
