"""arXiv PDF parsing via PyMuPDF: column-aware, font-size heading detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf

from research_navigator.common.types import ContentType
from research_navigator.parse.models import Block, BlockType, ParsedDocument, Section, SectionKind

_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+\S")
_ABSTRACT_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"^\s*(references|bibliography)\b", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table)\s*\d", re.IGNORECASE)

_BOLD_FLAG = 1 << 4


def _iter_text_blocks(page: pymupdf.Page) -> tuple[list[dict[str, Any]], int]:
    """Return (text blocks with geometry+font, count of skipped image blocks)."""
    data = page.get_text("dict")
    blocks: list[dict[str, Any]] = []
    images = 0
    for b in data["blocks"]:
        if b.get("type") != 0:  # 1 == image
            images += 1
            continue
        lines_text: list[str] = []
        max_size = 0.0
        bold = False
        for line in b["lines"]:
            spans = line["spans"]
            lines_text.append("".join(s["text"] for s in spans))
            for s in spans:
                max_size = max(max_size, float(s["size"]))
                if int(s["flags"]) & _BOLD_FLAG:
                    bold = True
        text = " ".join(t.strip() for t in lines_text if t.strip()).strip()
        if not text:
            continue
        x0, y0, x1, _ = b["bbox"]
        blocks.append(
            {"text": text, "size": max_size, "bold": bold, "y0": y0, "xcenter": (x0 + x1) / 2}
        )
    return blocks, images


def _order_columns(blocks: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    if not blocks:
        return []
    mid = page_width / 2
    left = [b for b in blocks if b["xcenter"] < mid]
    right = [b for b in blocks if b["xcenter"] >= mid]
    two_col = len(left) >= 0.2 * len(blocks) and len(right) >= 0.2 * len(blocks)
    if two_col:
        return sorted(left, key=lambda b: b["y0"]) + sorted(right, key=lambda b: b["y0"])
    return sorted(blocks, key=lambda b: b["y0"])


def _modal_body_size(blocks: list[dict[str, Any]]) -> float:
    weight: dict[float, int] = {}
    for b in blocks:
        key = round(float(b["size"]) * 2) / 2
        weight[key] = weight.get(key, 0) + len(b["text"])
    return max(weight, key=lambda k: weight[k]) if weight else 10.0


def _is_heading(text: str, size: float, body_size: float) -> bool:
    if _ABSTRACT_RE.match(text) or _REFERENCES_RE.match(text) or _SECTION_RE.match(text):
        return True
    words = text.split()
    return len(words) <= 12 and size >= body_size * 1.12 and not text.endswith(".")


def _heading_level(text: str) -> int:
    m = _SECTION_RE.match(text)
    if m:
        return min(2 + m.group(1).count("."), 6)
    return 2


def _classify_kind(text: str) -> SectionKind:
    if _ABSTRACT_RE.match(text):
        return SectionKind.abstract
    if _REFERENCES_RE.match(text):
        return SectionKind.references
    return SectionKind.body


def parse_pdf(doc_id: str, path: Path, title: str) -> ParsedDocument:
    if not path.exists():
        raise FileNotFoundError(path)

    ordered: list[dict[str, Any]] = []
    skipped_images = 0
    with pymupdf.open(path) as doc:
        for page in doc:
            page_blocks, images = _iter_text_blocks(page)
            skipped_images += images
            ordered.extend(_order_columns(page_blocks, page.rect.width))

    body_size = _modal_body_size(ordered)
    sections: list[Section] = []
    current = Section(index=0, title="", level=1, kind=SectionKind.body)
    seen_heading = False

    for b in ordered:
        text = b["text"]
        if _CAPTION_RE.match(text):
            current.blocks.append(Block(type=BlockType.caption, text=text))
            continue
        if _is_heading(text, float(b["size"]), body_size):
            seen_heading = True
            if current.blocks or current.title:
                sections.append(current)
            current = Section(
                index=len(sections),
                title=text,
                level=_heading_level(text),
                kind=_classify_kind(text),
            )
        else:
            current.blocks.append(Block(type=BlockType.paragraph, text=text))

    if current.blocks or current.title:
        sections.append(current)
    for n, s in enumerate(sections):
        s.index = n

    warnings: list[str] = []
    if skipped_images:
        warnings.append(f"skipped {skipped_images} image blocks (figures/graphics)")
    if not seen_heading:
        warnings.append("no headings detected; single-body fallback")
    if not any(s.kind is SectionKind.references for s in sections):
        warnings.append("no references section detected")

    return ParsedDocument(
        doc_id=doc_id,
        content_type=ContentType.arxiv_paper,
        title=title,
        sections=sections,
        warnings=warnings,
    )
