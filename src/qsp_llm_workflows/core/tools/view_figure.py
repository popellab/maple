#!/usr/bin/env python3
"""
Custom pydantic-ai tool for viewing figures from scientific papers.

Fetches a paper's HTML page, locates the specified figure by label,
and returns the image so the model can read numeric values from plots.
"""

import re
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin

import httpx
from pydantic_ai import ImageUrl, ToolReturn


class _FigureExtractor(HTMLParser):
    """Extract figure elements from HTML, targeting PMC and common publisher layouts."""

    def __init__(self) -> None:
        super().__init__()
        self.figures: list[dict[str, str]] = []
        self._current_figure: Optional[dict[str, str]] = None
        self._container_tag: Optional[str] = None  # "figure" or "div"
        self._depth = 0  # nesting depth for container tag type
        self._text_buffer: list[str] = []

    def _is_figure_container(self, tag: str, attr_dict: dict) -> bool:
        """Check if this element is a figure container."""
        if tag == "figure":
            return True
        if tag == "div" and any(
            cls in (attr_dict.get("class") or "")
            for cls in ("fig", "fig-group", "figure", "image-container")
        ):
            return True
        return False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attr_dict = dict(attrs)

        # Track nesting of same tag type inside figure container
        if self._current_figure is not None and tag == self._container_tag:
            self._depth += 1

        # Detect figure containers: <figure>, <div class="fig">, etc.
        if self._current_figure is None and self._is_figure_container(tag, attr_dict):
            self._current_figure = {"label": "", "caption": "", "img_url": ""}
            self._container_tag = tag
            self._depth = 1
            self._text_buffer = []

        # Capture image URL inside a figure
        if self._current_figure is not None and tag == "img":
            src = attr_dict.get("src") or attr_dict.get("data-src") or ""
            if src:
                self._current_figure["img_url"] = src
            alt = attr_dict.get("alt") or ""
            if alt and not self._current_figure["label"]:
                self._current_figure["label"] = alt

    def handle_endtag(self, tag: str) -> None:
        if self._current_figure is not None and tag == self._container_tag:
            self._depth -= 1
            if self._depth <= 0:
                # Finalize figure — outermost container closed
                captured = " ".join(self._text_buffer).strip()
                if captured and not self._current_figure["label"]:
                    self._current_figure["label"] = captured
                elif captured:
                    self._current_figure["caption"] = captured
                if self._current_figure["img_url"]:
                    self.figures.append(self._current_figure)
                self._current_figure = None
                self._container_tag = None
                self._depth = 0
                self._text_buffer = []

    def handle_data(self, data: str) -> None:
        if self._current_figure is not None:
            stripped = data.strip()
            if stripped:
                self._text_buffer.append(stripped)


def _normalize_label(label: str) -> str:
    """Normalize figure label for fuzzy matching.

    Handles variations like "Figure 2A", "Fig. 2A", "Fig 2a", "FIGURE 2A".
    """
    s = label.lower().strip()
    s = re.sub(r"\bfigure\b", "fig", s)
    s = re.sub(r"\bfig\.\s*", "fig ", s)
    s = re.sub(r"\bfig\s+", "fig ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _label_matches(figure_text: str, query_label: str) -> bool:
    """Check if figure_text contains or matches query_label (fuzzy)."""
    norm_text = _normalize_label(figure_text)
    norm_query = _normalize_label(query_label)
    return norm_query in norm_text


def extract_figures_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    """Parse HTML and extract figure metadata.

    Args:
        html: Raw HTML content
        base_url: Base URL for resolving relative image paths

    Returns:
        List of dicts with keys: label, caption, img_url
    """
    parser = _FigureExtractor()
    parser.feed(html)

    # Resolve relative URLs
    for fig in parser.figures:
        if fig["img_url"] and not fig["img_url"].startswith(("http://", "https://")):
            fig["img_url"] = urljoin(base_url, fig["img_url"])

    return parser.figures


def find_figure(figures: list[dict[str, str]], figure_label: str) -> Optional[dict[str, str]]:
    """Find the best matching figure for a given label.

    Args:
        figures: List of extracted figure dicts
        figure_label: Query label (e.g., "Figure 2A")

    Returns:
        Best matching figure dict, or None if no match
    """
    # Exact/fuzzy match on label or caption
    for fig in figures:
        if _label_matches(fig["label"], figure_label):
            return fig
        if _label_matches(fig["caption"], figure_label):
            return fig

    # Fallback: match just the number part
    number_match = re.search(r"(\d+[a-zA-Z]?)", figure_label)
    if number_match:
        num_part = number_match.group(1).lower()
        for fig in figures:
            fig_num = re.search(r"(\d+[a-zA-Z]?)", fig["label"])
            if fig_num and fig_num.group(1).lower() == num_part:
                return fig

    return None


async def view_figure(paper_url: str, figure_label: str) -> ToolReturn:
    """View a figure image from a scientific paper.

    Fetches the paper page and extracts the specified figure image so you
    can read numeric values from plots, bar charts, scatter plots, and other
    data visualizations.

    Args:
        paper_url: URL of the paper (PMC, publisher site, etc.)
        figure_label: Figure identifier, e.g. 'Figure 2A', 'Fig. 3', 'Supplementary Figure S1'
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; QSP-LLM-Workflows/1.0; " "mailto:research@example.com)"
            )
        }
        resp = await client.get(paper_url, headers=headers)
        resp.raise_for_status()

    html = resp.text
    figures = extract_figures_from_html(html, paper_url)

    if not figures:
        return ToolReturn(
            return_value=(
                f"No figures found on page {paper_url}. "
                "The page may use JavaScript rendering or a non-standard layout. "
                "Try a PMC URL (e.g., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC...)."
            )
        )

    match = find_figure(figures, figure_label)

    if match:
        caption_preview = match["caption"][:300] if match["caption"] else "(no caption)"
        return ToolReturn(
            return_value=(
                f"Found {figure_label}: label='{match['label']}', " f"caption='{caption_preview}'"
            ),
            content=[ImageUrl(url=match["img_url"])],
        )

    # No exact match — list available figures
    available = [
        f"  - '{fig['label']}'" + (f" ({fig['caption'][:80]}...)" if fig["caption"] else "")
        for fig in figures[:10]
    ]
    return ToolReturn(
        return_value=(
            f"Could not find '{figure_label}' on {paper_url}.\n"
            f"Available figures ({len(figures)} total):\n" + "\n".join(available)
        )
    )
