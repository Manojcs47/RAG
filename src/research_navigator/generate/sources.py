"""Turn ranked chunks into numbered Sources, collapsing same-doc chunks into a
single citation entry (M2 citation dedup)."""

from __future__ import annotations

from research_navigator.retrieve.models import RetrievedChunk

from .models import Source
from .settings import GenerateSettings


def build_sources(
    chunks: tuple[RetrievedChunk, ...] | list[RetrievedChunk],
    settings: GenerateSettings,
) -> list[Source]:
    """Group chunks by doc_id -> one Source each.

    - representative section = the doc's highest-scoring chunk's section
    - context text = that doc's top chunks concatenated (char-budgeted)
    - Sources ordered by best chunk score; numbered 1..N (capped at max_sources)
    """
    groups: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        doc_id = c.doc_id or f"point:{c.point_id}"
        groups.setdefault(doc_id, []).append(c)

    ranked_docs = sorted(
        groups.items(),
        key=lambda kv: max(ch.score for ch in kv[1]),
        reverse=True,
    )

    sources: list[Source] = []
    for i, (doc_id, group) in enumerate(ranked_docs[: settings.max_sources], start=1):
        group_sorted = sorted(group, key=lambda ch: ch.score, reverse=True)
        rep = group_sorted[0]
        texts = [ch.text for ch in group_sorted[: settings.max_chunks_per_doc] if ch.text]
        combined = "\n\n".join(texts)[: settings.source_char_budget]
        payload = rep.payload
        authors_raw = payload.get("authors")
        authors = [str(a) for a in authors_raw] if isinstance(authors_raw, list) else []
        source_url = payload.get("source_url")
        sources.append(
            Source(
                index=i,
                doc_id=doc_id,
                title=rep.title
                or (payload.get("title") if isinstance(payload.get("title"), str) else None)
                or doc_id,
                authors=authors,
                year=rep.year,
                content_type=rep.content_type or "",
                section_title=rep.section_title,
                source_url=source_url if isinstance(source_url, str) else None,
                text=combined,
                score=rep.score,
            )
        )
    return sources
