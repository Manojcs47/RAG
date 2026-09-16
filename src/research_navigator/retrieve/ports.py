"""Structural interfaces the retriever needs. RnStore (after the S5 patch) and
the real Embedder satisfy these, and tests can supply fakes."""

from __future__ import annotations

from typing import Protocol

from qdrant_client import models

from ..ingest.embedder import QueryVectors, SparseVec
from ..ingest.store import Hit


class QueryEmbedder(Protocol):
    @property
    def dense_dim(self) -> int: ...

    def embed_query(self, text: str) -> QueryVectors: ...


class HybridSearcher(Protocol):
    def hybrid_query(
        self,
        *,
        dense: list[float],
        sparse: SparseVec | None,
        query_filter: models.Filter | None,
        limit: int,
        prefetch_limit: int,
        fusion: str,
    ) -> list[Hit]: ...

    def dense_query(
        self,
        *,
        dense: list[float],
        query_filter: models.Filter | None,
        limit: int,
    ) -> list[Hit]: ...
