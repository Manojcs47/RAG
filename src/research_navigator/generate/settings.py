"""Generation configuration (nested under RN_ as ``generate``)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateSettings(BaseModel):
    """Tunables for answer synthesis + citation rendering. No hardcoding."""

    # refusal
    refusal_message: str = (
        "I don't have enough relevant material in the corpus to answer this confidently."
    )
    refusal_sentinel: str = Field(
        default="INSUFFICIENT_CONTEXT",
        description="If the model emits this, we refuse (LLM-side abstention).",
    )
    require_citations: bool = Field(
        default=True,
        description="Refuse if the validated answer carries no real citation.",
    )

    # source construction (citation dedup happens here: one source per doc)
    max_chunks_per_doc: int = Field(
        default=3, ge=1, description="Chunks concatenated per document as context."
    )
    source_char_budget: int = Field(
        default=4000, ge=100, description="Max chars of context text per source."
    )
    max_sources: int = Field(
        default=8, ge=1, description="Cap on distinct documents shown to the model."
    )

    # citation formatting
    authors_etal_threshold: int = Field(
        default=3, ge=2, description="'First et al.' once authors >= this."
    )
    source_labels: dict[str, str] = Field(
        default_factory=lambda: {
            "survey_blog": "Lil'Log",
            "course_chapter": "Hugging Face Learn",
            "lab_blog_post": "Lab Blog",
        },
        description="content_type -> display source. arxiv_paper is special-cased to 'arXiv:<id>'.",
    )
