"""Construct a live Qdrant-backed :class:`VectorStore` from settings.

Kept separate from :mod:`store` so unit tests can build a ``VectorStore`` around an
in-memory client without importing any real-connection code, and so the CLI stays thin.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from research_navigator.config import Settings
from research_navigator.ingest.store import VectorStore


def build_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        api_key=settings.qdrant_api_key,
    )


def build_store(settings: Settings, client: QdrantClient | None = None) -> VectorStore:
    return VectorStore(
        client or build_client(settings),
        settings.collection_name,
        scroll_page_size=settings.ingest_scroll_page_size,
    )
