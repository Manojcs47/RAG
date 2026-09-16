"""Thin Qdrant wrapper. This is the ONLY module that talks to qdrant-client;
every other layer receives plain data (``Hit``), never qdrant types."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from .embedder import SparseVec
from .schema import DENSE_VECTOR, PAYLOAD_INDEXES, SPARSE_VECTOR

PointId = int | str

_FUSIONS: dict[str, models.Fusion] = {
    "rrf": models.Fusion.RRF,
    "dbsf": models.Fusion.DBSF,
}


def _as_point_id(pid: object) -> PointId:
    return pid if isinstance(pid, int | str) else str(pid)


@dataclass(frozen=True)
class Hit:
    """A retrieval hit, decoupled from qdrant's ScoredPoint."""

    id: PointId
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class PointRecord:
    """An embedded chunk ready to upsert."""

    id: PointId
    dense: list[float]
    sparse: SparseVec
    payload: dict[str, Any]


def _batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class RnStore:
    """Owns the collection lifecycle and all read/write access."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense_dim: int,
        *,
        batch_size: int = 128,
        scroll_page_size: int = 256,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dense_dim = dense_dim
        self._batch = batch_size
        self._page = scroll_page_size

    # --- introspection -------------------------------------------------------
    @property
    def collection(self) -> str:
        return self._collection

    def vector_names(self) -> tuple[str, str]:
        return (DENSE_VECTOR, SPARSE_VECTOR)

    def exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    def count_where(self, query_filter: models.Filter) -> int:
        return int(
            self._client.count(self._collection, count_filter=query_filter, exact=True).count
        )

    # --- lifecycle -----------------------------------------------------------
    def create(self) -> None:
        """Create the hybrid collection (dense + IDF sparse) and payload indexes."""
        self._client.create_collection(
            self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self._dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # IDF must be set at creation; bm25 emits raw TF, IDF is server-side
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field_name, field_schema in PAYLOAD_INDEXES:
            self._client.create_payload_index(
                self._collection,
                field_name=field_name,
                field_schema=field_schema,
            )

    def drop(self) -> None:
        if self.exists():
            self._client.delete_collection(self._collection)

    def recreate(self) -> None:
        self.drop()
        self.create()

    # --- diff support (idempotent upsert lives in pipeline) ------------------
    def existing_ids_for_doc(self, doc_id: str) -> set[PointId]:
        """All point ids currently stored for a document (paginated scroll)."""
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        ids: set[PointId] = set()
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                self._collection,
                scroll_filter=flt,
                limit=self._page,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(_as_point_id(p.id) for p in points)
            if offset is None:
                break
        return ids

    # --- writes --------------------------------------------------------------
    def upsert(self, records: Iterable[PointRecord]) -> int:
        recs = list(records)
        structs = [
            models.PointStruct(
                id=r.id,
                vector={
                    DENSE_VECTOR: r.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=r.sparse.indices, values=r.sparse.values
                    ),
                },
                payload=r.payload,
            )
            for r in recs
        ]
        for batch in _batched(structs, self._batch):
            self._client.upsert(self._collection, points=batch)
        return len(structs)

    def delete(self, ids: Iterable[PointId]) -> int:
        id_list = list(ids)
        for batch in _batched(id_list, self._batch):
            self._client.delete(
                self._collection,
                points_selector=models.PointIdsList(points=batch),
            )
        return len(id_list)

    # --- retrieval (M2) ------------------------------------------------------
    def hybrid_query(
        self,
        *,
        dense: list[float],
        sparse: SparseVec | None,
        query_filter: models.Filter | None,
        limit: int,
        prefetch_limit: int,
        fusion: str,
    ) -> list[Hit]:
        """Fused dense+sparse retrieval; filter applied inside each prefetch
        branch (server-side). ``sparse=None`` => dense-only prefetch."""
        prefetch = [
            models.Prefetch(
                query=dense, using=DENSE_VECTOR, filter=query_filter, limit=prefetch_limit
            )
        ]
        if sparse is not None:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE_VECTOR,
                    filter=query_filter,
                    limit=prefetch_limit,
                )
            )
        key = fusion.lower()
        if key not in _FUSIONS:
            raise ValueError(f"unknown fusion strategy: {fusion!r}")
        points = self._client.query_points(
            self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=_FUSIONS[key]),
            limit=limit,
            with_payload=True,
        ).points
        return [
            Hit(id=_as_point_id(p.id), score=p.score, payload=dict(p.payload or {})) for p in points
        ]

    def dense_query(
        self,
        *,
        dense: list[float],
        query_filter: models.Filter | None,
        limit: int,
    ) -> list[Hit]:
        """Dense-only query returning cosine scores (the refusal signal)."""
        points = self._client.query_points(
            self._collection,
            query=dense,
            using=DENSE_VECTOR,
            query_filter=query_filter,
            limit=limit,
            with_payload=False,
        ).points
        return [Hit(id=_as_point_id(p.id), score=p.score, payload={}) for p in points]
