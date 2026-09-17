"""Central configuration. All tunables live here (pydantic-settings, env prefix
``RN_``, nested delimiter ``__``). No hardcoded paths/models/thresholds elsewhere.

NOTE (reconstruction): this mirrors the config contract the rest of the project
depends on. Merge with your existing config.py rather than blind-overwriting —
keep any S0-S4 fields your modules already import.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .chunk.settings import ChunkingSettings
from .generate.settings import GenerateSettings
from .retrieve.settings import RetrieveSettings


class PathsSettings(BaseModel):
    root: Path = Path(".")
    data_dir: Path = Path("data")
    parsed_dir: Path = Path("data/parsed")
    chunks_dir: Path = Path("data/chunks")
    corpus_dir: Path = Path("corpus")

    @property
    def manifest(self) -> Path:
        return self.corpus_dir / "manifest.json"


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    timeout: float = 30.0
    collection: str = "research_navigator"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class EmbeddingSettings(BaseModel):
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"


class IngestSettings(BaseModel):
    upsert_batch_size: int = Field(default=128, ge=1)
    scroll_page_size: int = Field(default=256, ge=1)


class LLMSettings(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: str | None = None  # falls back to OPENAI_API_KEY env if None
    base_url: str | None = None  # set for an OpenAI-compatible OSS server


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_logs: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RN_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieve: RetrieveSettings = Field(default_factory=RetrieveSettings)
    generate: GenerateSettings = Field(default_factory=GenerateSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
