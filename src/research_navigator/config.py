"""Application configuration via pydantic-settings.

All tunables (models, thresholds, paths, Qdrant connection) live here so that
nothing is hardcoded in business logic (M5 requirement).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated settings loaded from environment / .env.

    Environment variables are prefixed with ``RN_`` (e.g. ``RN_QDRANT_PORT``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RN_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Qdrant connection ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False
    qdrant_api_key: str | None = None
    collection_name: str = "research_navigator"

    # --- Embeddings (used from Session 4) ---
    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/bm25"

    # --- Generation (used from Session 6) ---
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None

    # --- Retrieval knobs (tuned in Sessions 5-6) ---
    retrieval_top_k: int = 8
    refusal_similarity_threshold: float = 0.35

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = False

    # --- Paths ---
    corpus_dir: Path = Path("corpus")
    documents_dir: Path = Path("documents")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
