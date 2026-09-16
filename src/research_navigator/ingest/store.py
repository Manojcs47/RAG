"""Thin wrapper around the Qdrant client.

Every raw client call lives here so the pipeline can be unit-tested against an in-memory
client and the rest of the codebase never imports ``qdrant_client`` directly. Methods are
deliberately small and side-effect-explicit; the *decision* of what to write lives in the
pipeline, which only calls ``upsert``/``delete`` when there is genuinely something to change
(this is what yields "re-ingest unchanged = 0 writes").
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import structlog
from qdrant_client import QdrantClient, models

from research_navigator.ingest import schema

log = structlog.get_logger(__name__)


class VectorStore:
    def __init__(
        self, client: QdrantClient, collection: str, *, scroll_page_size: int = 256
    ) -> None:
        self._client = client
        self._collection = collection
        self._scroll_page_size = scroll_page_size

    @property
    def collection(self) -> str:
        return self._collection

    def exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def create(self, dense_dim: int) -> None:
        """Create the collection with named dense+sparse vectors and payload indexes."""
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=schema.dense_vectors_config(dense_dim),
            sparse_vectors_config=schema.sparse_vectors_config(),
        )
        for field_name, field_schema in schema.PAYLOAD_INDEXES:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        log.info(
            "collection_created",
            collection=self._collection,
            dense_dim=dense_dim,
            indexes=[f for f, _ in schema.PAYLOAD_INDEXES],
        )

    def drop(self) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
            log.info("collection_dropped", collection=self._collection)

    def recreate(self, dense_dim: int) -> None:
        self.drop()
        self.create(dense_dim)

    def existing_ids_for_doc(self, doc_id: str) -> set[str]:
        """All point ids currently stored for a document (paginated scroll, ids only)."""
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        ids: set[str] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=flt,
                with_payload=False,
                with_vectors=False,
                limit=self._scroll_page_size,
                offset=offset,
            )
            ids.update(str(p.id) for p in points)
            if offset is None:
                break
        return ids

    def upsert(self, points: Sequence[models.PointStruct], *, batch_size: int = 128) -> int:
        written = 0
        for start in range(0, len(points), batch_size):
            batch = list(points[start : start + batch_size])
            self._client.upsert(collection_name=self._collection, points=batch)
            written += len(batch)
        return written

    def delete(self, ids: Iterable[str], *, batch_size: int = 128) -> int:
        id_list = list(ids)
        deleted = 0
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start : start + batch_size]
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=list(batch)),
            )
            deleted += len(batch)
        return deleted

    def count(self, count_filter: models.Filter | None = None) -> int:
        return int(
            self._client.count(
                collection_name=self._collection, count_filter=count_filter, exact=True
            ).count
        )

    def count_where(self, field: str, value: str | int | bool) -> int:
        flt = models.Filter(
            must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))]
        )
        return self.count(flt)

    def vector_names(self) -> tuple[list[str], list[str]]:
        """(dense names, sparse names) declared on the collection -- used by ``validate``."""
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors
        dense_names = list(vectors.keys()) if isinstance(vectors, dict) else []
        sparse_names = list(sparse.keys()) if isinstance(sparse, dict) else []
        return dense_names, sparse_names
