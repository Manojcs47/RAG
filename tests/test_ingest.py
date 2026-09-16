"""Session 4 ingestion tests.

Hermetic: an in-memory ``QdrantClient(":memory:")`` plus a deterministic fake embedder,
so no Docker server, no model download, and no network are required. This exercises the
full ingest logic (schema, deterministic ids, diff-based idempotent upsert, stats,
validate) end to end.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from research_navigator.chunk.hashing import content_hash
from research_navigator.chunk.models import Chunk
from research_navigator.common.manifest import Manifest, ManifestEntry
from research_navigator.common.types import ContentType
from research_navigator.config import Settings
from research_navigator.ingest import schema
from research_navigator.ingest.embedder import SparseVec
from research_navigator.ingest.ids import point_id
from research_navigator.ingest.pipeline import (
    collection_stats,
    ingest_corpus,
    load_chunks,
    validate_corpus,
)
from research_navigator.ingest.store import VectorStore

# --------------------------------------------------------------------------- fakes


class FakeEmbedder:
    """Deterministic, offline stand-in for FastEmbed."""

    dense_dim = 8
    dense_model = "fake-dense"
    sparse_model = "fake-sparse"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._dense(t) for t in texts]

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseVec]:
        return [self._sparse(t) for t in texts]

    def _dense(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dense_dim]]

    def _sparse(self, text: str) -> SparseVec:
        weights: dict[int, float] = {}
        for tok in text.lower().split():
            idx = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16) % 100_000
            weights[idx] = weights.get(idx, 0.0) + 1.0
        # Qdrant rejects an empty sparse vector; give blank text a harmless singleton.
        if not weights:
            weights[0] = 1.0
        return SparseVec(indices=list(weights.keys()), values=list(weights.values()))


# --------------------------------------------------------------------------- helpers


def make_chunk(doc_id: str, idx: int, text: str, **overrides: object) -> Chunk:
    base: dict[str, object] = {
        "doc_id": doc_id,
        "content_type": ContentType.arxiv_paper,
        "title": f"Title {doc_id}",
        "authors": ["Author"],
        "year": 2017,
        "primary_category": "cs.CL",
        "tags": ["transformers", "attention"],
        "is_foundational": True,
        "source_url": f"https://arxiv.org/abs/{doc_id}",
        "local_path": f"corpus/{doc_id}.pdf",
        "section_index": 0,
        "section_title": "Body",
        "section_kind": "body",
        "chunk_index": idx,
        "text": text,
        "token_count": len(text.split()),
        "content_hash": content_hash(text),
        "is_abstract": False,
    }
    base.update(overrides)
    return Chunk.model_validate(base)


def write_cache(settings: Settings, doc_id: str, chunks: list[Chunk]) -> None:
    settings.chunk_dir.mkdir(parents=True, exist_ok=True)
    payload = [c.payload() for c in chunks]
    (settings.chunk_dir / f"{doc_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(chunk_dir=tmp_path / "chunks")
    s.collection_name = "test_rn"
    return s


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(QdrantClient(":memory:"), "test_rn")


@pytest.fixture
def corpus(settings: Settings) -> Manifest:
    """Two docs written to the chunk cache; return a matching manifest."""
    d1 = [
        make_chunk("arxiv-1", 0, "attention is all you need", is_abstract=True),
        make_chunk("arxiv-1", 1, "the transformer uses self attention over tokens"),
    ]
    d2 = [
        make_chunk(
            "hf-1",
            0,
            "tokenizers split text into subwords",
            content_type=ContentType.course_chapter,
            year=2023,
            is_foundational=False,
            primary_category="course",
        ),
    ]
    write_cache(settings, "arxiv-1", d1)
    write_cache(settings, "hf-1", d2)
    return Manifest(
        documents=[
            ManifestEntry(
                doc_id="arxiv-1",
                content_type=ContentType.arxiv_paper,
                title="T1",
                year=2017,
                primary_category="cs.CL",
                is_foundational=True,
                source_url="u",
                local_path="p",
            ),
            ManifestEntry(
                doc_id="hf-1",
                content_type=ContentType.course_chapter,
                title="T2",
                year=2023,
                primary_category="course",
                is_foundational=False,
                source_url="u",
                local_path="p",
            ),
        ]
    )


# --------------------------------------------------------------------------- tests


def test_point_id_is_deterministic_uuid_sensitive_to_hash() -> None:
    a = point_id("doc", 3, "deadbeefdeadbeef")
    b = point_id("doc", 3, "deadbeefdeadbeef")
    c = point_id("doc", 3, "0000beefdeadbeef")
    assert a == b  # reproducible
    assert a != c  # content hash flows into the id
    uuid.UUID(a)  # valid UUID string (accepted by Qdrant)


def test_ensure_collection_named_vectors(store: VectorStore) -> None:
    store.create(dense_dim=8)
    assert store.exists()
    dense_names, sparse_names = store.vector_names()
    assert dense_names == [schema.DENSE_VECTOR]
    assert sparse_names == [schema.SPARSE_VECTOR]


def test_ingest_is_idempotent_zero_writes_on_reingest(
    settings: Settings, store: VectorStore, corpus: Manifest
) -> None:
    embedder = FakeEmbedder()
    first = ingest_corpus(settings, store, embedder, corpus)
    assert first.added == 3  # 2 + 1 chunks
    assert first.deleted == 0
    assert store.count() == 3

    second = ingest_corpus(settings, store, embedder, corpus)
    assert second.added == 0  # <-- re-ingest unchanged = 0 writes
    assert second.deleted == 0
    assert second.unchanged == 3
    assert second.writes == 0
    assert store.count() == 3


def test_editing_chunk_text_replaces_point(
    settings: Settings, store: VectorStore, corpus: Manifest
) -> None:
    embedder = FakeEmbedder()
    ingest_corpus(settings, store, embedder, corpus)

    # Edit one chunk's text -> new content_hash -> new id; old id becomes stale.
    edited = [
        make_chunk("arxiv-1", 0, "attention is all you need", is_abstract=True),
        make_chunk("arxiv-1", 1, "COMPLETELY REWRITTEN body about cross attention"),
    ]
    write_cache(settings, "arxiv-1", edited)

    report = ingest_corpus(settings, store, embedder, corpus, doc_ids=["arxiv-1"])
    doc_result = next(d for d in report.docs if d.doc_id == "arxiv-1")
    assert doc_result.added == 1
    assert doc_result.deleted == 1
    assert store.count() == 3  # net point count unchanged


def test_removing_chunks_deletes_stale_points(
    settings: Settings, store: VectorStore, corpus: Manifest
) -> None:
    embedder = FakeEmbedder()
    ingest_corpus(settings, store, embedder, corpus)

    write_cache(settings, "arxiv-1", [make_chunk("arxiv-1", 0, "attention is all you need")])
    report = ingest_corpus(settings, store, embedder, corpus, doc_ids=["arxiv-1"])
    doc_result = next(d for d in report.docs if d.doc_id == "arxiv-1")
    assert doc_result.deleted == 1
    assert store.count() == 2


def test_full_manifest_payload_carried_to_points(
    settings: Settings, store: VectorStore, corpus: Manifest
) -> None:
    ingest_corpus(settings, store, FakeEmbedder(), corpus)
    chunks = load_chunks(settings, "arxiv-1")
    assert chunks is not None
    pid = point_id("arxiv-1", chunks[0].chunk_index, chunks[0].content_hash)
    got = store._client.retrieve("test_rn", ids=[pid], with_payload=True)
    payload = got[0].payload
    assert payload is not None
    for field in ("doc_id", "content_type", "title", "year", "tags", "is_foundational"):
        assert field in payload
    assert payload["tags"] == ["transformers", "attention"]


def test_stats_counts_are_correct(settings: Settings, store: VectorStore, corpus: Manifest) -> None:
    ingest_corpus(settings, store, FakeEmbedder(), corpus)
    stats = collection_stats(store, corpus)
    assert stats.total_points == 3
    assert stats.by_content_type["arxiv_paper"] == 2
    assert stats.by_content_type["course_chapter"] == 1
    assert stats.by_year[2017] == 2
    assert stats.by_year[2023] == 1
    assert stats.foundational == 2


def test_validate_ok_then_detects_drift(
    settings: Settings, store: VectorStore, corpus: Manifest
) -> None:
    ingest_corpus(settings, store, FakeEmbedder(), corpus)
    clean = validate_corpus(settings, store, corpus)
    assert clean.ok
    assert clean.expected_points == clean.actual_points == 3

    # Introduce drift by deleting a document's points behind the pipeline's back.
    stale_ids = store.existing_ids_for_doc("hf-1")
    store.delete(stale_ids)
    drifted = validate_corpus(settings, store, corpus)
    assert not drifted.ok
    kinds = {i.kind for i in drifted.issues}
    assert "missing_points" in kinds
    assert "count_mismatch" in kinds


def test_validate_missing_collection(settings: Settings, corpus: Manifest) -> None:
    empty = VectorStore(QdrantClient(":memory:"), "does_not_exist")
    report = validate_corpus(settings, empty, corpus)
    assert not report.ok
    assert report.issues[0].kind == "missing_collection"
