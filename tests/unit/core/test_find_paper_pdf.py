"""Regression tests for ``snippet_validator.find_paper_pdf``.

The fallback path matches filename text against ``<Author><Year>`` source
tags. Author surnames in published papers commonly contain non-ASCII
characters (e.g. ``Canè``, ``Müller``, ``Brügger``); the matcher must
fold both sides through NFKD + ASCII so the substring lookup succeeds.
"""

from pathlib import Path

from maple.core.calibration.snippet_validator import find_paper_pdf


def _make_paper(papers_dir: Path, name: str) -> Path:
    p = papers_dir / name
    p.write_bytes(b"")  # empty file is fine — find_paper_pdf only checks names
    return p


def test_finds_pdf_in_subdirectory(tmp_path: Path) -> None:
    papers_dir = tmp_path
    sub = papers_dir / "Ganusov2014"
    sub.mkdir()
    pdf = sub / "ganusov-2014-T-cell-dynamics.pdf"
    pdf.write_bytes(b"")
    assert find_paper_pdf("Ganusov2014", papers_dir) == pdf


def test_finds_pdf_with_accented_surname(tmp_path: Path) -> None:
    """``Cane2023`` source tag should match a PDF named with the actual
    accented surname ``Canè et al. - 2023 - ...pdf``."""
    pdf = _make_paper(
        tmp_path,
        "Canè et al. - 2023 - Neutralization of NET-associated human ARG1.pdf",
    )
    assert find_paper_pdf("Cane2023", tmp_path) == pdf


def test_finds_pdf_with_umlaut(tmp_path: Path) -> None:
    pdf = _make_paper(tmp_path, "Müller et al. - 2020 - Some title.pdf")
    assert find_paper_pdf("Muller2020", tmp_path) == pdf


def test_finds_pdf_with_hyphenated_surname(tmp_path: Path) -> None:
    """Hyphens in the surname should not block the match."""
    pdf = _make_paper(tmp_path, "Vukmanovic-Stejic et al. - 2008 - Title.pdf")
    assert find_paper_pdf("VukmanovicStejic2008", tmp_path) == pdf


def test_finds_pdf_with_nonbreaking_space(tmp_path: Path) -> None:
    """Non-breaking space (\\xa0) between words must collapse to match."""
    pdf = _make_paper(tmp_path, "den\xa0Braber et al. - 2012 - Title.pdf")
    assert find_paper_pdf("denBraber2012", tmp_path) == pdf


def test_returns_none_when_no_match(tmp_path: Path) -> None:
    _make_paper(tmp_path, "OtherAuthor et al. - 2020 - Unrelated.pdf")
    assert find_paper_pdf("Ganusov2014", tmp_path) is None


def test_returns_none_for_year_mismatch(tmp_path: Path) -> None:
    """Author matches but year differs -> no match."""
    _make_paper(tmp_path, "Canè et al. - 2021 - Different year.pdf")
    assert find_paper_pdf("Cane2023", tmp_path) is None
