"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from research_navigator.chunk.settings import ChunkingSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RN_",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # --- Qdrant connection ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False
    qdrant_api_key: str | None = None
    collection_name: str = "research_navigator"

    # --- Embeddings (Session 4) ---
    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/bm25"

    # --- Generation (Session 6) ---
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None

    # --- Retrieval knobs ---
    retrieval_top_k: int = 8
    refusal_similarity_threshold: float = 0.35

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = False

    # --- Paths ---
    corpus_dir: Path = Path("corpus")
    documents_dir: Path = Path("documents")
    parsed_dir: Path = Path("data/parsed")  # NEW: cached IR from Session 2

    # paths
    chunk_dir: Path = Path("data/chunks")  # NEW (Session 3)

    # chunking (NEW, Session 3)
    # Tokenizer used to MEASURE chunk length. Tracks the dense embedding model,
    # because that model's 512-token limit is the real constraint on chunk size.
    chunk_tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    chunking: ChunkingSettings = ChunkingSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
