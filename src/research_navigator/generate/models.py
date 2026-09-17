"""Data models for generation. Pure data; no I/O, no LLM, no qdrant."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """One document's worth of retrieved context, collapsed from >=1 chunk.

    Citation dedup lives here: multiple same-doc chunks become one Source whose
    ``section_title`` is the most-relevant (highest-scoring) chunk's section.
    """

    index: int  # 1-based position as shown to the model
    doc_id: str
    title: str
    authors: list[str]
    year: int | None
    content_type: str
    section_title: str | None
    source_url: str | None
    text: str
    score: float  # best fused score among the doc's chunks (for ordering)


@dataclass(frozen=True)
class Citation:
    """A rendered citation entry mapped to a validated inline marker."""

    index: int  # 1-based, matches the [n] marker in the final answer text
    doc_id: str
    title: str
    authors: str  # display form ("First et al.")
    year: int | None
    source: str  # e.g. "arXiv:2305.14314", "Lil'Log", "Hugging Face Learn"
    section: str | None
    url: str | None

    def render(self) -> str:
        bits = [f"[{self.index}] {self.title}."]
        if self.authors:
            bits.append(f" {self.authors}")
        if self.year is not None:
            bits.append(f" ({self.year})")
        bits.append(f". {self.source}")
        if self.section:
            bits.append(f', "{self.section}"')
        if self.url:
            bits.append(f". {self.url}")
        return "".join(bits)


@dataclass(frozen=True)
class Answer:
    """Final result: either a cited answer or a graceful refusal."""

    query: str
    text: str
    citations: tuple[Citation, ...]
    refused: bool
    reason: str | None  # why refused / diagnostic tag
    intent: str
    confidence: float
    # diagnostic only (M4 judge owns per-claim faithfulness): sentences with no [n]
    uncited_sentences: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        if self.refused:
            return self.text
        lines = [self.text, "", "Sources:"]
        lines.extend(c.render() for c in self.citations)
        return "\n".join(lines)
