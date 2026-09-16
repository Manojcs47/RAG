"""M2 retrieval orchestrator: query understanding -> hybrid fused retrieval with
server-side metadata filtering -> cosine-based refusal gate."""

from __future__ import annotations

import structlog
from qdrant_client import models

from ..ingest.embedder import QueryVectors
from ..ingest.store import Hit
from .filters import build_qdrant_filter
from .models import QueryAnalysis, QueryIntent, RetrievalResult, RetrievedChunk
from .ports import HybridSearcher, QueryEmbedder
from .query_understanding import analyze
from .settings import RetrieveSettings
from .vocab import QueryCatalog

log = structlog.get_logger(__name__)


class Retriever:
    """Stateless (per-query) retriever. All qdrant access is delegated to the
    searcher (store), so this layer stays free of qdrant-client calls."""

    def __init__(
        self,
        *,
        embedder: QueryEmbedder,
        searcher: HybridSearcher,
        catalog: QueryCatalog,
        settings: RetrieveSettings,
    ) -> None:
        self._embedder = embedder
        self._searcher = searcher
        self._catalog = catalog
        self._settings = settings

    def analyze(self, query: str, *, now_year: int | None = None) -> QueryAnalysis:
        return analyze(query, catalog=self._catalog, settings=self._settings, now_year=now_year)

    def retrieve(self, query: str, *, now_year: int | None = None) -> RetrievalResult:
        s = self._settings
        analysis = self.analyze(query, now_year=now_year)
        qvec = self._embedder.embed_query(query)
        qfilter = build_qdrant_filter(analysis.filters)
        filtered = qfilter is not None

        log.info(
            "retrieve.analyzed",
            query=query,
            intent=analysis.intent.value,
            reasons=list(analysis.reasons),
            filtered=filtered,
            fusion=s.fusion,
            dense_only=s.dense_only,
        )

        hits = self._search(qvec, qfilter)

        # Filter-relaxation fallback: an over-eager inferred filter shouldn't
        # cause a false refusal. If a filtered search returns nothing, retry
        # once unfiltered and record that we did (no silent failure).
        if filtered and not hits:
            log.warning("retrieve.filter_relaxed", reason="filtered result empty")
            hits = self._search(qvec, None)
            qfilter, filtered = None, False

        # Confidence = max dense cosine over the (final) candidate space.
        probe = self._searcher.dense_query(dense=qvec.dense, query_filter=qfilter, limit=s.top_k)
        confidence = max((h.score for h in probe), default=0.0)
        refused = confidence < s.refusal_threshold or not hits

        chunks = [RetrievedChunk(point_id=h.id, score=h.score, payload=h.payload) for h in hits]
        # RECENT queries: chronological digest (newest first), stable on ties.
        if analysis.intent is QueryIntent.RECENT:
            chunks.sort(key=lambda c: c.year or 0, reverse=True)

        log.info(
            "retrieve.done",
            n_chunks=len(chunks),
            confidence=round(confidence, 4),
            refused=refused,
        )
        return RetrievalResult(
            analysis=analysis,
            chunks=tuple(chunks),
            confidence=confidence,
            refused=refused,
            fusion=s.fusion,
            dense_only=s.dense_only,
            top_k=s.top_k,
            filtered=filtered,
        )

    def _search(self, qvec: QueryVectors, qfilter: models.Filter | None) -> list[Hit]:
        s = self._settings
        return self._searcher.hybrid_query(
            dense=qvec.dense,
            sparse=None if s.dense_only else qvec.sparse,
            query_filter=qfilter,
            limit=s.top_k,
            prefetch_limit=s.prefetch_limit,
            fusion=s.fusion,
        )
