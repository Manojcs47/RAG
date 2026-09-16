"""Chunking policy — all sizing knobs live here (M5: no hardcoded thresholds).

Token budgets are expressed in the length unit produced by the configured
``Tokenizer`` (see ``chunk.tokenizer``). Defaults sit well under the
``bge-small-en-v1.5`` 512-token limit so a chunk is never silently truncated at
embed time, even allowing for tokenizer approximation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from research_navigator.common.types import ContentType


class ChunkPolicy(BaseModel):
    """Sizing policy for one content type."""

    target_tokens: int = 320  # soft target; a chunk is flushed once it reaches this
    max_tokens: int = 512  # hard cap; packing never crosses this (except atomic code)
    overlap_tokens: int = 64  # trailing prose carried into the next in-section chunk
    min_tokens: int = 32  # a trailing fragment below this is merged back if it fits


def _default_policies() -> dict[ContentType, ChunkPolicy]:
    return {
        # Dense academic prose — keep comfortably under 512.
        ContentType.arxiv_paper: ChunkPolicy(
            target_tokens=320, max_tokens=512, overlap_tokens=64, min_tokens=32
        ),
        # Long-form surveys (Lil'Log) — a little larger reads better.
        ContentType.survey_blog: ChunkPolicy(
            target_tokens=384, max_tokens=512, overlap_tokens=64, min_tokens=40
        ),
        ContentType.lab_blog_post: ChunkPolicy(
            target_tokens=384, max_tokens=512, overlap_tokens=64, min_tokens=40
        ),
        # Tutorial chapters (HF Learn) — often code-heavy; slightly tighter overlap.
        ContentType.course_chapter: ChunkPolicy(
            target_tokens=320, max_tokens=512, overlap_tokens=48, min_tokens=32
        ),
    }


class ChunkingSettings(BaseModel):
    """Top-level chunking configuration, nested under ``Settings.chunking``."""

    tokenizer_model: str = "BAAI/bge-small-en-v1.5"
    default: ChunkPolicy = ChunkPolicy()
    per_content_type: dict[ContentType, ChunkPolicy] = Field(default_factory=_default_policies)
    # Abstracts are emitted as a single distinct chunk unless they exceed this,
    # in which case they are packed but stay flagged ``is_abstract``.
    abstract_max_tokens: int = 512

    def policy_for(self, content_type: ContentType) -> ChunkPolicy:
        return self.per_content_type.get(content_type, self.default)
