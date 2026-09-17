"""Retrieval configuration. Nested under the top-level RN_ settings as
``retrieve`` (env prefix ``RN_RETRIEVE__*``). No thresholds/models hardcoded."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Fusion = Literal["rrf", "dbsf"]


class RetrieveSettings(BaseModel):
    """Tunables for query understanding + hybrid retrieval."""

    # --- ranking -------------------------------------------------------------
    top_k: int = Field(default=8, ge=1, description="Final chunks returned.")
    prefetch_limit: int = Field(
        default=40,
        ge=1,
        description="Candidates pulled per branch before fusion; forced >= top_k.",
    )
    fusion: Fusion = "rrf"
    dense_only: bool = Field(
        default=False, description="Disable the sparse branch (for A/B eval in M4)."
    )

    # --- refusal (cosine space, from the dense probe) ------------------------
    refusal_threshold: float = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        description="Refuse if max dense cosine < this. NOT the fused score.",
    )

    # --- query understanding -------------------------------------------------
    infer_filters: bool = True
    recency_window_years: int = Field(
        default=2,
        ge=1,
        description="'recent'/'latest' with no explicit year => year >= now - this.",
    )
    recency_reference_year: int | None = Field(
        default=None,
        description="Pin 'now' for deterministic recency (defaults to current year).",
    )
    # content-type trigger phrases -> canonical content_type value. Intersected
    # at runtime with the content types actually present in the corpus, so a
    # corpus swap can't produce a filter that excludes everything.
    content_type_triggers: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "arxiv_paper": ["paper", "papers", "arxiv", "preprint"],
            "course_chapter": ["course", "chapter", "tutorial", "lesson"],
            "survey_blog": ["survey", "surveys"],
            "lab_blog_post": ["blog", "post"],
        }
    )
    recency_cues: list[str] = Field(
        default_factory=lambda: [
            "recent",
            "latest",
            "newest",
            "new",
            "modern",
            "state of the art",
            "sota",
            "cutting edge",
            "nowadays",
            "these days",
            "this year",
        ]
    )
    foundational_cues: list[str] = Field(
        default_factory=lambda: [
            "foundational",
            "seminal",
            "landmark",
            "classic",
            "original",
            "pioneering",
        ]
    )

    @model_validator(mode="after")
    def _prefetch_ge_topk(self) -> RetrieveSettings:
        # Qdrant requires each prefetch limit >= final limit for stable fusion.
        if self.prefetch_limit < self.top_k:
            object.__setattr__(self, "prefetch_limit", self.top_k)
        return self
