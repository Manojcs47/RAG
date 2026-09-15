"""Typed manifest models, shared by parsing and ingestion."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from research_navigator.common.types import ContentType


class ManifestEntry(BaseModel):
    doc_id: str
    content_type: ContentType
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int
    month: int | None = None
    primary_category: str
    secondary_categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_foundational: bool = False
    citation_count: int | None = None
    source_url: str
    local_path: str


class Manifest(BaseModel):
    schema_version: str | None = None
    documents: list[ManifestEntry]


def load_manifest(path: Path) -> Manifest:
    """Load and validate a manifest.json file."""
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
