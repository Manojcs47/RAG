"""Ingestion tunables (collection name, Qdrant connection, batch sizes).

Nested under ``Settings.ingest`` so everything is env-overridable, e.g.
``RN_INGEST__COLLECTION`` or ``RN_INGEST__QDRANT_URL`` -- no hardcoding in logic.
"""

from __future__ import annotations

from pydantic import BaseModel


class IngestSettings(BaseModel):
    collection: str = "research_navigator"

    # Server connection. Unit tests bypass this by injecting a ``QdrantClient(":memory:")``
    # directly; these defaults target the pinned local Docker server (v1.18.2).
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    prefer_grpc: bool = False

    # Batching. upsert/delete are chunked so a large corpus does not build one giant request.
    upsert_batch_size: int = 128
    scroll_page_size: int = 256
