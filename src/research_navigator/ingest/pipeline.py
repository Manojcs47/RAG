"""Ingestion orchestration (M1c-M1f).

The core idea is a per-document *diff*:

* Every chunk's point id is ``uuid5`` over ``(doc_id, chunk_index, content_hash)``, so the
  id already encodes the content. Recomputing ids for the current chunk cache gives the
  *desired* id set.
* The *existing* id set is read back from Qdrant (scroll filtered by ``doc_id``).
* We embed + upsert only ``desired - existing`` (new or content-changed chunks) and delete
  only ``existing - desired`` (stale chunks). Chunks in both sets are untouched.

An unchanged document therefore triggers **zero** embeds and **zero** writes; a content
edit shows up as one add (new id) plus one delete (old id). Embeddings -- the expensive
step -- are computed only for chunks actually being written.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import structlog
from qdrant_client import models

from research_navigator.chunk.models import Chunk
from research_navigator.common.manifest import Manifest
from research_navigator.common.types import ContentType
from research_navigator.config import Settings
from research_navigator.ingest import schema
from research_navigator.ingest.embedder import Embedder
from research_navigator.ingest.ids import point_id
from research_navigator.ingest.models import (
    CollectionStats,
    DocIngestResult,
    IngestReport,
    ValidationIssue,
    ValidationReport,
)
from research_navigator.ingest.store import VectorStore

log = structlog.get_logger(__name__)


def load_chunks(settings: Settings, doc_id: str) -> list[Chunk] | None:
    """Load the cached chunks for a document, or ``None`` if the cache is absent."""
    path = settings.chunk_dir / f"{doc_id}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in raw]


def _desired_index(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    """Map each chunk to its deterministic point id."""
    return {point_id(c.doc_id, c.chunk_index, c.content_hash): c for c in chunks}


def _points_for(
    ids: Sequence[str], desired: dict[str, Chunk], embedder: Embedder
) -> list[models.PointStruct]:
    """Embed and build point structs for exactly the given ids (order preserved)."""
    chunks = [desired[i] for i in ids]
    texts = [c.text for c in chunks]
    dense = embedder.embed_documents(texts)
    sparse = embedder.embed_sparse(texts)
    points: list[models.PointStruct] = []
    for pid, chunk, dvec, svec in zip(ids, chunks, dense, sparse, strict=True):
        points.append(
            models.PointStruct(
                id=pid,
                vector={
                    schema.DENSE_VECTOR: dvec,
                    schema.SPARSE_VECTOR: models.SparseVector(
                        indices=svec.indices, values=svec.values
                    ),
                },
                payload=chunk.payload(),
            )
        )
    return points


def ingest_doc(
    store: VectorStore,
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    doc_id: str,
    batch_size: int = 128,
) -> DocIngestResult:
    """Idempotently sync one document's chunks into the collection."""
    desired = _desired_index(chunks)
    existing = store.existing_ids_for_doc(doc_id)

    to_add = sorted(desired.keys() - existing)  # sorted -> deterministic batching
    to_delete = sorted(existing - desired.keys())
    unchanged = len(desired.keys() & existing)

    if to_add:
        points = _points_for(to_add, desired, embedder)
        store.upsert(points, batch_size=batch_size)
    if to_delete:
        store.delete(to_delete, batch_size=batch_size)

    log.info(
        "ingest_doc_done",
        doc_id=doc_id,
        added=len(to_add),
        deleted=len(to_delete),
        unchanged=unchanged,
    )
    return DocIngestResult(
        doc_id=doc_id, added=len(to_add), deleted=len(to_delete), unchanged=unchanged
    )


def ingest_corpus(
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    manifest: Manifest,
    *,
    doc_ids: Iterable[str] | None = None,
) -> IngestReport:
    """Ensure the collection exists, then idempotently ingest each requested document."""
    if not store.exists():
        store.create(embedder.dense_dim)

    wanted = set(doc_ids) if doc_ids is not None else None
    results: list[DocIngestResult] = []
    for entry in manifest.documents:
        if wanted is not None and entry.doc_id not in wanted:
            continue
        chunks = load_chunks(settings, entry.doc_id)
        if chunks is None:
            log.warning("ingest_chunks_missing", doc_id=entry.doc_id)
            continue
        results.append(
            ingest_doc(
                store,
                embedder,
                chunks,
                doc_id=entry.doc_id,
                batch_size=settings.ingest_upsert_batch_size,
            )
        )

    report = IngestReport(collection=store.collection, docs=results)
    log.info(
        "ingest_corpus_done",
        collection=store.collection,
        documents=len(results),
        added=report.added,
        deleted=report.deleted,
        unchanged=report.unchanged,
    )
    return report


def collection_stats(store: VectorStore, manifest: Manifest) -> CollectionStats:
    """Point counts overall and grouped by content_type / year / foundational flag."""
    by_content_type = {ct.value: store.count_where("content_type", ct.value) for ct in ContentType}
    years = sorted({e.year for e in manifest.documents})
    by_year = {y: store.count_where("year", y) for y in years}
    return CollectionStats(
        collection=store.collection,
        total_points=store.count(),
        by_content_type=by_content_type,
        by_year=by_year,
        foundational=store.count_where("is_foundational", True),
    )


def validate_corpus(settings: Settings, store: VectorStore, manifest: Manifest) -> ValidationReport:
    """Check the collection matches the chunk cache: no missing, stale, or orphan points."""
    issues: list[ValidationIssue] = []
    expected = 0

    if not store.exists():
        return ValidationReport(
            collection=store.collection,
            ok=False,
            expected_points=0,
            actual_points=0,
            issues=[ValidationIssue(doc_id=None, kind="missing_collection", detail="absent")],
        )

    for entry in manifest.documents:
        chunks = load_chunks(settings, entry.doc_id)
        if chunks is None:
            continue
        desired = set(_desired_index(chunks).keys())
        expected += len(desired)
        existing = store.existing_ids_for_doc(entry.doc_id)
        missing = desired - existing
        stale = existing - desired
        if missing:
            issues.append(
                ValidationIssue(
                    doc_id=entry.doc_id, kind="missing_points", detail=f"{len(missing)} not indexed"
                )
            )
        if stale:
            issues.append(
                ValidationIssue(
                    doc_id=entry.doc_id, kind="stale_points", detail=f"{len(stale)} orphaned"
                )
            )

    actual = store.count()
    if actual != expected:
        issues.append(
            ValidationIssue(
                doc_id=None,
                kind="count_mismatch",
                detail=f"expected {expected}, found {actual}",
            )
        )

    return ValidationReport(
        collection=store.collection,
        ok=not issues,
        expected_points=expected,
        actual_points=actual,
        issues=issues,
    )
