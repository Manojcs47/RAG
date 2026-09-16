from __future__ import annotations

import math

import pytest
from qdrant_client import QdrantClient, models

from research_navigator.ingest.embedder import QueryVectors, SparseVec
from research_navigator.ingest.store import PointRecord, RnStore
from research_navigator.retrieve import (
    QueryCatalog,
    QueryIntent,
    Retriever,
    RetrieveSettings,
    analyze,
)
from research_navigator.retrieve.filters import build_qdrant_filter


class FakeEmbedder:
    """Deterministic, dependency-free embedder for hermetic tests."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dense_dim(self) -> int:
        return self._dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        cleaned = "".join(c if c.isalnum() else " " for c in text.lower())
        return [t for t in cleaned.split() if t]

    def _encode(self, text: str) -> QueryVectors:
        vec = [0.0] * self._dim
        tf: dict[int, float] = {}
        for tok in self._tokens(text):
            vec[hash(tok) % self._dim] += 1.0
            idx = (hash(tok) % 90_000) + 1
            tf[idx] = tf.get(idx, 0.0) + 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        dense = [x / norm for x in vec]
        items = sorted(tf.items())
        sparse = SparseVec(indices=[i for i, _ in items], values=[v for _, v in items])
        return QueryVectors(dense=dense, sparse=sparse)

    def embed_query(self, text: str) -> QueryVectors:
        return self._encode(text)

    def embed_passages(self, texts: list[str]) -> list[QueryVectors]:
        return [self._encode(t) for t in texts]


NOW = 2025

CATALOG = QueryCatalog.from_manifest_rows(
    [
        {"content_type": "arxiv_paper", "tags": ["attention", "RAG", "long_context"]},
        {"content_type": "course_chapter", "tags": ["agents"]},
        {"content_type": "survey_blog", "tags": ["RLHF"]},
        {"content_type": "lab_blog_post", "tags": ["reasoning"]},
    ]
)


def _analyze(q: str, **overrides: object):
    settings = RetrieveSettings(**overrides)  # type: ignore[arg-type]
    return analyze(q, catalog=CATALOG, settings=settings, now_year=NOW)


# --------------------------- filter inference ------------------------------


def test_recency_cue_sets_year_gte() -> None:
    a = _analyze("recent work on attention", recency_window_years=2)
    assert a.filters.year_gte == NOW - 2
    assert a.intent is QueryIntent.RECENT


def test_explicit_year_wins_over_window() -> None:
    a = _analyze("attention papers since 2021")
    assert a.filters.year_gte == 2021


def test_last_n_years() -> None:
    a = _analyze("long context advances in the last 3 years")
    assert a.filters.year_gte == NOW - 3
    assert "long_context" in a.filters.tags


def test_last_n_months_rounds_up_to_years() -> None:
    a = _analyze("developments in the last 18 months")
    assert a.filters.year_gte == NOW - 2  # ceil(18/12) = 2


def test_tag_inference_is_canonical_and_multiword() -> None:
    a = _analyze("how does long context relate to rag")
    assert set(a.filters.tags) == {"long_context", "RAG"}  # canonical casing


def test_content_type_inference_intersects_corpus() -> None:
    a = _analyze("recommend papers and course chapters on agents")
    assert set(a.filters.content_types) == {"arxiv_paper", "course_chapter"}
    assert a.intent is QueryIntent.FIND_PAPERS


def test_foundational_cue() -> None:
    a = _analyze("what is the seminal paper on attention")
    assert a.filters.is_foundational is True


def test_compare_intent() -> None:
    a = _analyze("compare RAG versus long context")
    assert a.intent is QueryIntent.COMPARE


def test_plain_query_infers_nothing() -> None:
    a = _analyze("explain how transformers work")
    assert a.filters.is_empty()


def test_infer_filters_disabled() -> None:
    a = _analyze("recent papers on attention", infer_filters=False)
    assert a.filters.is_empty()


# --------------------------- filter building -------------------------------


def test_build_filter_none_when_empty() -> None:
    a = _analyze("explain transformers")
    assert build_qdrant_filter(a.filters) is None


def test_build_filter_conditions() -> None:
    a = _analyze("recent foundational papers on attention")
    f = build_qdrant_filter(a.filters)
    assert isinstance(f, models.Filter)
    assert f.must is not None and len(f.must) >= 3  # year, tags, content_type, foundational


# --------------------------- end-to-end retrieval ---------------------------


@pytest.fixture()
def store() -> RnStore:
    emb = FakeEmbedder(dim=64)
    st = RnStore(QdrantClient(":memory:"), "rn_test", dense_dim=emb.dense_dim)
    st.create()
    docs = [
        (
            1,
            "attention is all you need self attention transformers",
            {
                "doc_id": "d1",
                "content_type": "arxiv_paper",
                "tags": ["attention"],
                "primary_category": "cs.CL",
                "is_foundational": True,
                "year": 2017,
                "title": "Attention",
                "section_title": "Abstract",
                "text": "attention...",
            },
        ),
        (
            2,
            "retrieval augmented generation rag with dense vectors",
            {
                "doc_id": "d2",
                "content_type": "arxiv_paper",
                "tags": ["RAG"],
                "primary_category": "cs.CL",
                "is_foundational": False,
                "year": 2024,
                "title": "RAG",
                "section_title": "Method",
                "text": "rag...",
            },
        ),
        (
            3,
            "long context windows scaling attention efficiently",
            {
                "doc_id": "d3",
                "content_type": "arxiv_paper",
                "tags": ["long_context", "attention"],
                "primary_category": "cs.CL",
                "is_foundational": False,
                "year": 2024,
                "title": "LongCtx",
                "section_title": "Intro",
                "text": "long...",
            },
        ),
    ]
    pts = []
    for pid, b, p in docs:
        qv = emb.embed_passages([b])[0]
        pts.append(PointRecord(id=pid, dense=qv.dense, sparse=qv.sparse, payload=p))
    st.upsert(pts)
    return st


def _retriever(store: RnStore, **overrides: object) -> Retriever:
    return Retriever(
        embedder=FakeEmbedder(dim=64),
        searcher=store,
        catalog=CATALOG,
        settings=RetrieveSettings(refusal_threshold=0.1, **overrides),  # type: ignore[arg-type]
    )


def test_hybrid_returns_ranked_chunks(store: RnStore) -> None:
    res = _retriever(store).retrieve("retrieval augmented generation", now_year=NOW)
    assert not res.refused
    assert res.chunks
    assert res.chunks[0].doc_id == "d2"  # best lexical+dense match


def test_metadata_filter_applied_server_side(store: RnStore) -> None:
    # "recent" -> year >= 2023, so the 2017 foundational paper must be excluded.
    res = _retriever(store).retrieve("recent work on attention", now_year=NOW)
    years = {c.year for c in res.chunks}
    assert 2017 not in years
    assert res.filtered is True


def test_dense_only_mode(store: RnStore) -> None:
    res = _retriever(store, dense_only=True).retrieve("attention", now_year=NOW)
    assert res.dense_only is True
    assert res.chunks


def test_refusal_on_off_corpus_query(store: RnStore) -> None:
    r = Retriever(
        embedder=FakeEmbedder(dim=64),
        searcher=store,
        catalog=CATALOG,
        settings=RetrieveSettings(refusal_threshold=0.99),
    )
    res = r.retrieve("zzzz quantum gastronomy plumbing", now_year=NOW)
    assert res.refused is True
    assert res.confidence < 0.99


def test_recency_orders_chronologically(store: RnStore) -> None:
    res = _retriever(store).retrieve("recent attention research", now_year=NOW)
    yrs = [c.year for c in res.chunks if c.year]
    assert yrs == sorted(yrs, reverse=True)


def test_filter_relaxation_recovers(store: RnStore) -> None:
    # Filter to a year with no docs -> should relax and still return chunks.
    res = _retriever(store).retrieve("attention work since 2099", now_year=NOW)
    assert res.chunks  # recovered
    assert res.filtered is False
