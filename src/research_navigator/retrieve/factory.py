"""Wire up a Retriever from settings + manifest. Reuses the ingest factory for
the qdrant client/store and the embedder."""

from __future__ import annotations

import json
from pathlib import Path

from .ports import HybridSearcher, QueryEmbedder
from .retriever import Retriever
from .settings import RetrieveSettings
from .vocab import QueryCatalog


def load_catalog(manifest_path: Path) -> QueryCatalog:
    """Derive the tag vocabulary + content types from corpus/manifest.json."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data["documents"] if isinstance(data, dict) else data
    return QueryCatalog.from_manifest_rows(rows)


def build_retriever(
    *,
    embedder: QueryEmbedder,
    searcher: HybridSearcher,
    manifest_path: Path,
    settings: RetrieveSettings,
) -> Retriever:
    """Assemble a Retriever. ``embedder``/``searcher`` are the ingest Embedder
    and RnStore, checked structurally against the protocols by mypy."""
    return Retriever(
        embedder=embedder,
        searcher=searcher,
        catalog=load_catalog(manifest_path),
        settings=settings,
    )
