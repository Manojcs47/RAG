"""M2 retrieval: query understanding + hybrid fused retrieval with server-side
metadata filtering and a cosine-based refusal gate."""

from __future__ import annotations

from .factory import build_retriever, load_catalog
from .models import (
    InferredFilters,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult,
    RetrievedChunk,
)
from .query_understanding import analyze
from .retriever import Retriever
from .settings import RetrieveSettings
from .vocab import QueryCatalog

__all__ = [
    "InferredFilters",
    "QueryAnalysis",
    "QueryCatalog",
    "QueryIntent",
    "RetrievalResult",
    "RetrieveSettings",
    "RetrievedChunk",
    "Retriever",
    "analyze",
    "build_retriever",
    "load_catalog",
]
