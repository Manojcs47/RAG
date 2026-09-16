"""Deterministic Qdrant point identity.

A Qdrant point id must be an unsigned int or a UUID. We derive a UUIDv5 from the
Session 3 ``chunk_uid`` -- ``f"{doc_id}:{chunk_index}:{content_hash[:16]}"`` -- under a
fixed project namespace, so the same chunk always maps to the same point id on any
machine, in any run. Because ``chunk_uid`` embeds the content hash, editing a chunk's
text changes its point id: the edited chunk becomes a *new* point and its old id turns
stale, which is exactly what makes Session 4's diff-based upsert idempotent.
"""

from __future__ import annotations

import uuid

from research_navigator.chunk.hashing import chunk_uid

# Fixed namespace -> reproducible ids across machines. Never change this value for an
# existing collection: doing so would remap every point id and force a full re-ingest.
RN_NAMESPACE: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "ai-research-navigator.qdrant")


def point_id(doc_id: str, chunk_index: int, content_hash_hex: str) -> str:
    """Stable Qdrant point id (UUIDv5 string) for a chunk."""
    return str(uuid.uuid5(RN_NAMESPACE, chunk_uid(doc_id, chunk_index, content_hash_hex)))
