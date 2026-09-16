"""Construct a live Qdrant-backed :class:`RnStore` from settings.

Kept separate from :mod:`store` so unit tests can build an ``RnStore`` around an
in-memory client without importing any real-connection code, and so the CLI stays thin.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

from research_navigator.config import QdrantSettings, Settings
from research_navigator.ingest.store import RnStore


def build_client(qdrant: QdrantSettings) -> QdrantClient:
    return QdrantClient(
        host=qdrant.host,
        port=qdrant.port,
        grpc_port=qdrant.grpc_port,
        prefer_grpc=qdrant.prefer_grpc,
        timeout=int(qdrant.timeout),
    )


def build_store(client: QdrantClient, settings: Settings, dense_dim: int) -> RnStore:
    return RnStore(
        client,
        settings.qdrant.collection,
        dense_dim,
        batch_size=settings.ingest.upsert_batch_size,
        scroll_page_size=settings.ingest.scroll_page_size,
    )
