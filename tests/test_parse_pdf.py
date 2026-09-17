from __future__ import annotations

from pathlib import Path

import pymupdf

from research_navigator.parse.models import SectionKind
from research_navigator.parse.pdf import parse_pdf


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Abstract", fontsize=14)
    page.insert_text(
        (72, 120),
        "We study retrieval augmented generation for learners.",
        fontsize=10,
    )
    page.insert_text((72, 200), "1 Introduction", fontsize=14)
    page.insert_text(
        (72, 240),
        "We introduce a system that grounds answers in sources.",
        fontsize=10,
    )
    page.insert_text((72, 320), "References", fontsize=14)
    page.insert_text(
        (72, 360),
        "[1] Lewis et al. 2020. Retrieval-Augmented Generation.",
        fontsize=10,
    )
    doc.save(str(path))
    doc.close()


def test_pdf_section_kinds(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path)
    doc = parse_pdf("arxiv-x", pdf_path, "Sample Paper")

    kinds = {s.kind for s in doc.sections}
    assert SectionKind.abstract in kinds
    assert SectionKind.references in kinds
    assert doc.title == "Sample Paper"
    assert all(s.kind is not SectionKind.references for s in doc.retrievable_sections)
