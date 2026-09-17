"""The agent's tool layer: a corpus-metadata index (read from the same
``corpus/manifest.json`` the ingest/retrieve layers use) plus pure date-math.

This is the ">= 1 tool call" the M3 rubric asks for. Two route nodes exercise it:
``recent_developments`` (date-math + a corpus window count) and ``find_papers``
(a manifest query that answers *deterministically*, with zero LLM involvement and
therefore zero hallucination risk). ``paper_deep_dive`` uses it to resolve the
canonical paper metadata before generating.

The index depends only on the manifest rows, so it is trivially fakeable in tests
and stays correct if the corpus is swapped (tags/content-types are read, not
hard-coded — "corpus is a variable, not a constant").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def recency_cutoff(now_year: int, window_years: int) -> int:
    """Pure date-math: the earliest year that still counts as 'recent'.

    e.g. now=2026, window=2 -> 2024 (2024, 2025, 2026 are 'recent')."""
    return now_year - (window_years - 1)


@dataclass(frozen=True)
class DocMeta:
    """One document's manifest metadata (a superset of what find_papers displays)."""

    doc_id: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    content_type: str
    tags: tuple[str, ...]
    is_foundational: bool
    primary_category: str | None
    source_url: str | None

    @staticmethod
    def from_row(row: dict[str, Any]) -> DocMeta:
        def _s(key: str) -> str | None:
            v = row.get(key)
            return v if isinstance(v, str) else None

        year = row.get("year")
        authors = row.get("authors") or []
        tags = row.get("tags") or []
        return DocMeta(
            doc_id=_s("doc_id") or "",
            title=_s("title") or "(untitled)",
            authors=tuple(a for a in authors if isinstance(a, str)),
            year=year if isinstance(year, int) else None,
            content_type=_s("content_type") or "unknown",
            tags=tuple(t for t in tags if isinstance(t, str)),
            is_foundational=bool(row.get("is_foundational", False)),
            primary_category=_s("primary_category"),
            source_url=_s("source_url") or _s("local_path"),
        )


@dataclass(frozen=True)
class CorpusIndex:
    """In-memory, read-only view over the manifest. Constructed via ``from_manifest``
    (production) or ``from_rows`` (tests)."""

    docs: tuple[DocMeta, ...] = field(default_factory=tuple)

    # -- construction --------------------------------------------------------
    @staticmethod
    def from_rows(rows: list[dict[str, Any]]) -> CorpusIndex:
        return CorpusIndex(docs=tuple(DocMeta.from_row(r) for r in rows))

    @staticmethod
    def from_manifest(manifest_path: Path) -> CorpusIndex:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = data["documents"] if isinstance(data, dict) else data
        return CorpusIndex.from_rows(rows)

    # -- queries -------------------------------------------------------------
    @property
    def all_tags(self) -> frozenset[str]:
        return frozenset(t for d in self.docs for t in d.tags)

    @property
    def content_types(self) -> frozenset[str]:
        return frozenset(d.content_type for d in self.docs)

    def latest_year(self) -> int | None:
        years = [d.year for d in self.docs if d.year is not None]
        return max(years) if years else None

    def stats(self) -> dict[str, Any]:
        by_ct: dict[str, int] = {}
        by_year: dict[int, int] = {}
        n_found = 0
        for d in self.docs:
            by_ct[d.content_type] = by_ct.get(d.content_type, 0) + 1
            if d.year is not None:
                by_year[d.year] = by_year.get(d.year, 0) + 1
            n_found += int(d.is_foundational)
        return {
            "total": len(self.docs),
            "by_content_type": dict(sorted(by_ct.items())),
            "by_year": {y: by_year[y] for y in sorted(by_year)},
            "latest_year": self.latest_year(),
            "n_foundational": n_found,
            "n_tags": len(self.all_tags),
        }

    def find(
        self,
        *,
        text: str | None = None,
        tags: tuple[str, ...] = (),
        content_types: tuple[str, ...] = (),
        year_gte: int | None = None,
        year_lte: int | None = None,
        is_foundational: bool | None = None,
        limit: int | None = None,
    ) -> list[DocMeta]:
        """AND across constraints; tags/content_types match ANY within their field.
        Results ordered by year (newest first), then title. ``text`` is a
        case-insensitive substring match over title + tags."""
        needle = text.lower().strip() if text else None
        want_tags = {t.lower() for t in tags}
        want_ct = {c.lower() for c in content_types}

        out: list[DocMeta] = []
        for d in self.docs:
            if needle:
                hay = (d.title + " " + " ".join(d.tags)).lower()
                if needle not in hay:
                    continue
            if want_tags and not (want_tags & {t.lower() for t in d.tags}):
                continue
            if want_ct and d.content_type.lower() not in want_ct:
                continue
            if year_gte is not None and (d.year is None or d.year < year_gte):
                continue
            if year_lte is not None and (d.year is None or d.year > year_lte):
                continue
            if is_foundational is not None and d.is_foundational != is_foundational:
                continue
            out.append(d)

        out.sort(key=lambda m: (-(m.year or 0), m.title.lower()))
        return out[:limit] if limit is not None else out

    def match_tags(self, text: str) -> tuple[str, ...]:
        """Return corpus tags whose name appears in ``text`` (whitespace/underscore
        insensitive). Lets find_papers infer filters from the corpus's own vocabulary
        rather than a hard-coded list."""
        low = text.lower()
        hits = []
        for tag in sorted(self.all_tags):
            variants = {tag.lower(), tag.lower().replace("_", " "), tag.lower().replace("_", "")}
            if any(v and v in low for v in variants):
                hits.append(tag)
        return tuple(hits)
