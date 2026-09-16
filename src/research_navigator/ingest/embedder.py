"""Embeddings for ingest (passages) and retrieval (queries).

Dense: BAAI/bge-small-en-v1.5 (384-dim, ONNX/no-torch via fastembed).
Sparse: Qdrant/bm25 — emits RAW term frequencies; IDF is applied server-side by
the collection (Modifier.IDF set at creation). One dependency (fastembed) gives
both. ``build_embedder`` fails LOUD: there is no fake/degraded fallback in prod.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..config import EmbeddingSettings


@dataclass(frozen=True)
class SparseVec:
    """A sparse vector: parallel term-id indices and raw-TF values."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class QueryVectors:
    """A dense + sparse encoding of one text (passage or query)."""

    dense: list[float]
    sparse: SparseVec


@runtime_checkable
class Embedder(Protocol):
    """Encodes passages (ingest) and queries (retrieval)."""

    @property
    def dense_dim(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[QueryVectors]: ...

    def embed_query(self, text: str) -> QueryVectors: ...


class FastEmbedEmbedder:
    """FastEmbed-backed dense+sparse embedder.

    Passages use the plain document embedding path; queries use the model's
    query path (bge query instruction + bm25 query encoder). Dense dimension is
    *probed* from the model, never hardcoded.
    """

    def __init__(self, dense_model: str, sparse_model: str) -> None:
        try:
            from fastembed import SparseTextEmbedding, TextEmbedding
        except ImportError as exc:  # fail loud, actionable
            raise RuntimeError(
                "fastembed is required for embeddings but is not installed. "
                "Install it (`uv add fastembed`) or run in the Docker image."
            ) from exc

        self._dense: Any = TextEmbedding(model_name=dense_model)
        self._sparse: Any = SparseTextEmbedding(model_name=sparse_model)
        probe = next(iter(self._dense.embed(["dimension probe"])))
        self._dense_dim = len(probe)

    @property
    def dense_dim(self) -> int:
        return self._dense_dim

    @staticmethod
    def _to_sparse(embedding: Any) -> SparseVec:
        return SparseVec(
            indices=[int(i) for i in embedding.indices],
            values=[float(v) for v in embedding.values],
        )

    def embed_passages(self, texts: Sequence[str]) -> list[QueryVectors]:
        docs = list(texts)
        dense_it = self._dense.embed(docs)
        sparse_it = self._sparse.embed(docs)
        out: list[QueryVectors] = []
        for dense, sparse in zip(dense_it, sparse_it, strict=True):
            out.append(
                QueryVectors(
                    dense=[float(x) for x in dense],
                    sparse=self._to_sparse(sparse),
                )
            )
        return out

    def embed_query(self, text: str) -> QueryVectors:
        dense = next(iter(self._dense.query_embed(text)))
        sparse = next(iter(self._sparse.query_embed(text)))
        return QueryVectors(
            dense=[float(x) for x in dense],
            sparse=self._to_sparse(sparse),
        )


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    """Construct the production embedder. Fails loud on any load error — never
    returns a fake or degraded embedder."""
    try:
        embedder: Embedder = FastEmbedEmbedder(
            dense_model=settings.dense_model,
            sparse_model=settings.sparse_model,
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to build embedder (dense={settings.dense_model!r}, "
            f"sparse={settings.sparse_model!r}): {exc}"
        ) from exc
    return embedder
