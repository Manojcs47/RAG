"""Deterministic citation formatting. Pure functions -> unit-testable (M5)."""

from __future__ import annotations

import re

from .models import Citation, Source

_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")


def format_authors(authors: list[str], *, etal_threshold: int = 3) -> str:
    """'First et al.' for >= threshold authors; 'A and B' for two; 'A' for one."""
    names = [a for a in authors if a]
    if not names:
        return ""
    if len(names) >= etal_threshold:
        return f"{names[0]} et al."
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return names[0]


def extract_arxiv_id(*candidates: str | None) -> str | None:
    """Pull an arXiv id (e.g. 2305.14314) from a doc_id or source_url."""
    for c in candidates:
        if not c:
            continue
        m = _ARXIV_ID.search(c)
        if m:
            return m.group(1)
    return None


def source_label(
    content_type: str,
    *,
    doc_id: str,
    source_url: str | None,
    labels: dict[str, str],
) -> str:
    """Human-facing source tag. arXiv papers -> 'arXiv:<id>'; others via map."""
    if content_type == "arxiv_paper":
        arxiv_id = extract_arxiv_id(doc_id, source_url)
        return f"arXiv:{arxiv_id}" if arxiv_id else "arXiv"
    return labels.get(content_type, content_type or "source")


def to_citation(
    source: Source,
    index: int,
    *,
    etal_threshold: int,
    labels: dict[str, str],
) -> Citation:
    return Citation(
        index=index,
        doc_id=source.doc_id,
        title=source.title,
        authors=format_authors(source.authors, etal_threshold=etal_threshold),
        year=source.year,
        source=source_label(
            source.content_type,
            doc_id=source.doc_id,
            source_url=source.source_url,
            labels=labels,
        ),
        section=source.section_title,
        url=source.source_url,
    )
