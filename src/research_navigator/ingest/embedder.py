"""Embedding backend: dense (bge-small) + sparse (BM25), behind a Protocol.

**Backend decision (S4):** FastEmbed, not sentence-transformers. FastEmbed is OSS,
ONNX-based (no torch), integrates natively with Qdrant, and produces *both* the dense
(``BAAI/bge-small-en-v1.5``, 384-dim) and sparse (``Qdrant/bm25``) representations we
need from one dependency.

The concrete backend is hidden behind the :class:`Embedder` Protocol so tests can inject
a deterministic fake with no model download or network, and so a future backend swap
touches one factory. Unlike the tokenizer's whitespace fallback, there is **no** silent
fallback here: a fake embedder in production would corrupt the whole index, so a failure
to build the real backend is raised loudly.

BM25 note: FastEmbed computes only the term-frequency component client-side. Inverse
document frequency is computed *server-side* by Qdrant when the sparse vector is
configured with ``Modifier.IDF`` (see :mod:`research_navigator.ingest.schema`). This
module therefore emits raw BM25 sparse vectors and relies on the collection's IDF
modifier for correct weighting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)


class SparseVec(NamedTuple):
    """A sparse vector as parallel index/value lists (Qdrant's on-wire shape)."""

    indices: list[int]
    values: list[float]


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding surface the ingest pipeline depends on."""

    @property
    def dense_dim(self) -> int: ...

    @property
    def dense_model(self) -> str: ...

    @property
    def sparse_model(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Dense passage embeddings, one per input text."""
        ...

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseVec]:
        """Sparse (BM25) passage embeddings, one per input text."""
        ...


class FastEmbedEmbedder:
    """Real backend using FastEmbed's ONNX dense + BM25 sparse models."""

    def __init__(self, dense_model: str, sparse_model: str) -> None:
        # Imported lazily: FastEmbed pulls onnxruntime and downloads model files on first
        # use, so importing at call time keeps unit tests (which inject a fake) hermetic.
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model
        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)
        # Probe the true output dimension rather than hardcoding 384: keeps the collection
        # schema correct if the dense model is swapped via config.
        probe = next(iter(self._dense.embed(["dimension probe"])))
        self._dim = len(probe)
        log.info(
            "embedder_ready",
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_dim=self._dim,
        )

    @property
    def dense_dim(self) -> int:
        return self._dim

    @property
    def dense_model(self) -> str:
        return self._dense_model_name

    @property
    def sparse_model(self) -> str:
        return self._sparse_model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._dense.embed(list(texts))]

    def embed_sparse(self, texts: Sequence[str]) -> list[SparseVec]:
        out: list[SparseVec] = []
        for emb in self._sparse.embed(list(texts)):
            out.append(
                SparseVec(
                    indices=[int(i) for i in emb.indices],
                    values=[float(v) for v in emb.values],
                )
            )
        return out


def build_embedder(dense_model: str, sparse_model: str) -> Embedder:
    """Construct the real embedding backend, failing loudly if it cannot load."""
    try:
        return FastEmbedEmbedder(dense_model, sparse_model)
    except Exception as exc:
        log.error(
            "embedder_build_failed",
            dense_model=dense_model,
            sparse_model=sparse_model,
            error=str(exc),
        )
        raise RuntimeError(
            f"Failed to build FastEmbed backend (dense={dense_model!r}, "
            f"sparse={sparse_model!r}): {exc}"
        ) from exc
