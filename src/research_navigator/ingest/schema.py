"""Qdrant collection schema: named vector spaces and the payload index plan.

One point carries two named vectors -- a dense ``bge-small`` vector and a sparse ``bm25``
vector -- plus the full chunk payload. Keeping both vectors on the same point (rather than
in two collections) means one id, one payload, and one ingest path that cannot drift.

The sparse space is created with ``Modifier.IDF`` so Qdrant computes inverse document
frequency from live collection statistics at query time. This **must** be set at creation:
it cannot be added to an existing collection without recreating it.
"""

from __future__ import annotations

from qdrant_client import models

# Named vector spaces (referenced by these names in upserts and in the S5 Query API).
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Payload fields indexed for fast server-side filtering (S5 metadata filters).
# Field name -> Qdrant payload schema type. ``tags``/``secondary_categories`` are keyword
# indexes over list fields (membership queries). ``doc_id`` is indexed because the
# idempotent-upsert diff filters existing points by doc_id on every ingest.
PAYLOAD_INDEXES: tuple[tuple[str, models.PayloadSchemaType], ...] = (
    ("doc_id", models.PayloadSchemaType.KEYWORD),
    ("content_type", models.PayloadSchemaType.KEYWORD),
    ("primary_category", models.PayloadSchemaType.KEYWORD),
    ("tags", models.PayloadSchemaType.KEYWORD),
    ("is_foundational", models.PayloadSchemaType.BOOL),
    ("year", models.PayloadSchemaType.INTEGER),
)


def dense_vectors_config(dense_dim: int) -> dict[str, models.VectorParams]:
    """Dense space config; cosine matches bge-small's normalized embeddings."""
    return {DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)}


def sparse_vectors_config() -> dict[str, models.SparseVectorParams]:
    """Sparse space config with server-side IDF weighting for BM25."""
    return {SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)}
