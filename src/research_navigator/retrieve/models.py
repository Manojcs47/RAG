"""Data models for the M2 retrieval pipeline. Pure data, no I/O, no qdrant."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QueryIntent(StrEnum):
    """Coarse, deterministic retrieval hint.

    This is *not* the M3 LangGraph router (which owns the six official routes);
    it only biases M2 retrieval — e.g. ``RECENT`` triggers a chronological
    re-sort. M3 will supersede it with the real classifier.
    """

    CONCEPT = "concept"
    PAPER_SPECIFIC = "paper_specific"
    COMPARE = "compare"
    RECENT = "recent"
    FIND_PAPERS = "find_papers"


@dataclass(frozen=True)
class InferredFilters:
    """Metadata constraints inferred from the query. Empty == no filtering."""

    year_gte: int | None = None
    year_lte: int | None = None
    tags: tuple[str, ...] = ()
    content_types: tuple[str, ...] = ()
    primary_categories: tuple[str, ...] = ()
    is_foundational: bool | None = None
    doc_ids: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not any(
            (
                self.year_gte is not None,
                self.year_lte is not None,
                self.tags,
                self.content_types,
                self.primary_categories,
                self.is_foundational is not None,
                self.doc_ids,
            )
        )


@dataclass(frozen=True)
class QueryAnalysis:
    """Output of query understanding."""

    raw: str
    normalized: str
    intent: QueryIntent
    filters: InferredFilters
    # human-readable trace of *why* each filter fired (for logs / debugging / ADR)
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked chunk with its full carried-through payload."""

    point_id: int | str
    score: float  # fused rank score (RRF/DBSF) — ordering only, not cosine
    payload: dict[str, Any] = field(default_factory=dict)

    def _str(self, key: str) -> str | None:
        v = self.payload.get(key)
        return v if isinstance(v, str) else None

    @property
    def doc_id(self) -> str | None:
        return self._str("doc_id")

    @property
    def title(self) -> str | None:
        return self._str("title")

    @property
    def section_title(self) -> str | None:
        return self._str("section_title")

    @property
    def content_type(self) -> str | None:
        return self._str("content_type")

    @property
    def year(self) -> int | None:
        v = self.payload.get("year")
        return v if isinstance(v, int) else None

    @property
    def text(self) -> str:
        return self._str("text") or ""


@dataclass(frozen=True)
class RetrievalResult:
    """Everything downstream (S6 generation) needs to answer or refuse."""

    analysis: QueryAnalysis
    chunks: tuple[RetrievedChunk, ...]
    confidence: float  # max dense cosine over candidates (refusal signal)
    refused: bool
    # snapshot of the config that produced this result (eval reproducibility)
    fusion: str
    dense_only: bool
    top_k: int
    filtered: bool
