"""Prompt construction. Pure: (query, sources, settings) -> chat messages."""

from __future__ import annotations

from .llm import ChatMessage
from .models import Source
from .settings import GenerateSettings

_SYSTEM = """You are the AI Research Navigator, a citation-grounded assistant for \
AI/ML learners. Answer the learner's question using ONLY the numbered sources \
provided. Follow these rules exactly:

1. Every factual sentence must end with a citation marker like [1] or [1, 2] that \
identifies the source(s) supporting it.
2. Use only the source numbers given. NEVER invent a source number or cite a \
source that is not listed.
3. Do not use outside knowledge. If the sources do not contain enough information \
to answer, reply with exactly this and nothing else: {sentinel}
4. Be concise and precise. Prefer synthesis across sources over copying long \
passages verbatim."""


def _render_sources(sources: list[Source], budget: int) -> str:
    blocks: list[str] = []
    for s in sources:
        header = f"[{s.index}] {s.title}"
        if s.section_title:
            header += f' — section "{s.section_title}"'
        meta = ", ".join(
            part for part in (s.content_type or None, str(s.year) if s.year else None) if part
        )
        if meta:
            header += f" ({meta})"
        blocks.append(f"{header}\n{s.text[:budget]}")
    return "\n\n".join(blocks)


def build_messages(
    query: str, sources: list[Source], settings: GenerateSettings
) -> list[ChatMessage]:
    system = _SYSTEM.format(sentinel=settings.refusal_sentinel)
    user = (
        f"Question: {query}\n\n"
        f"Sources:\n{_render_sources(sources, settings.source_char_budget)}\n\n"
        "Write the grounded, cited answer now."
    )
    return [ChatMessage("system", system), ChatMessage("user", user)]
