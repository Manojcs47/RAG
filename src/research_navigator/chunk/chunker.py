"""Content-type-aware chunking over the parsed IR (Session 3, M1b).

Guarantees:
  * Abstracts are emitted as a single distinct chunk (flagged ``is_abstract``).
  * Chunks never cross a section boundary.
  * Code blocks are never split; a lone oversized code block becomes its own
    chunk (logged, not silently truncated).
  * References are excluded from retrieval (we iterate ``retrievable_sections``)
    but retained in the cached IR for citation lookup.
  * Output is deterministic for a fixed (document, tokenizer, settings) triple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from research_navigator.chunk.hashing import content_hash
from research_navigator.chunk.models import Chunk
from research_navigator.chunk.settings import ChunkingSettings, ChunkPolicy
from research_navigator.chunk.tokenizer import Tokenizer
from research_navigator.common.manifest import ManifestEntry
from research_navigator.parse.models import BlockType, ParsedDocument, Section, SectionKind

log = structlog.get_logger(__name__)

# Sentence boundary: end punctuation + whitespace followed by an opener.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'\[])")


@dataclass(frozen=True)
class _Unit:
    """An atom of packable content. Prose units may be merged/overlapped; code
    units are indivisible and never duplicated into overlap."""

    text: str
    is_code: bool
    tokens: int


@dataclass(frozen=True)
class _Span:
    """A finished chunk's text plus its abstract flag, before metadata is attached."""

    text: str
    is_abstract: bool


def _split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT.split(text)) if s]


def _window_by_words(text: str, target_tokens: int, tokenizer: Tokenizer) -> list[str]:
    """Last-resort split of a single over-long prose sentence, by word windows."""
    words = text.split()
    windows: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if tokenizer.count(" ".join(current)) >= target_tokens:
            windows.append(" ".join(current))
            current = []
    if current:
        windows.append(" ".join(current))
    return windows


def _to_units(section: Section, policy: ChunkPolicy, tokenizer: Tokenizer) -> list[_Unit]:
    units: list[_Unit] = []
    for block in section.blocks:
        if block.type is BlockType.code:
            units.append(_Unit(block.text, is_code=True, tokens=tokenizer.count(block.text)))
            continue
        for sentence in _split_sentences(block.text):
            tokens = tokenizer.count(sentence)
            if tokens <= policy.max_tokens:
                units.append(_Unit(sentence, is_code=False, tokens=tokens))
            else:  # pathological long sentence — window it so no prose chunk overflows
                for piece in _window_by_words(sentence, policy.target_tokens, tokenizer):
                    units.append(_Unit(piece, is_code=False, tokens=tokenizer.count(piece)))
    return units


def _pack(units: list[_Unit], policy: ChunkPolicy) -> list[list[_Unit]]:
    """Greedy pack units into non-overlapping chunks; units are never split."""
    chunks: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_tokens = 0
    for unit in units:
        if current and current_tokens + unit.tokens > policy.max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit.tokens
        if current_tokens >= policy.target_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
    if current:
        chunks.append(current)
    return chunks


def _merge_small_tail(chunks: list[list[_Unit]], policy: ChunkPolicy) -> list[list[_Unit]]:
    """Fold a below-``min_tokens`` trailing chunk into its predecessor if it fits."""
    if len(chunks) < 2:
        return chunks
    tail_tokens = sum(u.tokens for u in chunks[-1])
    prev_tokens = sum(u.tokens for u in chunks[-2])
    if tail_tokens < policy.min_tokens and tail_tokens + prev_tokens <= policy.max_tokens:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    return chunks


def _tail_words(text: str, budget_tokens: int, tokenizer: Tokenizer) -> str:
    """Trailing words of ``text`` whose token count stays within ``budget_tokens``."""
    if budget_tokens <= 0 or not text:
        return ""
    tail: list[str] = []
    for word in reversed(text.split()):
        candidate = [word, *tail]
        if tail and tokenizer.count(" ".join(candidate)) > budget_tokens:
            break
        tail = candidate
        if tokenizer.count(" ".join(tail)) >= budget_tokens:
            break
    return " ".join(tail)


def _render(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i > 0:
            previous = units[i - 1]
            parts.append("\n\n" if unit.is_code or previous.is_code else " ")
        parts.append(unit.text)
    return "".join(parts).strip()


def _chunk_section(
    section: Section,
    policy: ChunkPolicy,
    settings: ChunkingSettings,
    tokenizer: Tokenizer,
    *,
    doc_id: str,
) -> list[_Span]:
    is_abstract = section.kind is SectionKind.abstract

    if is_abstract:
        whole = section.text
        if tokenizer.count(whole) <= settings.abstract_max_tokens:
            return [_Span(whole, is_abstract=True)]
        # Oversized abstract: pack but keep the flag on every piece.

    units = _to_units(section, policy, tokenizer)
    if not units:
        return []

    groups = _merge_small_tail(_pack(units, policy), policy)

    spans: list[_Span] = []
    for i, group in enumerate(groups):
        text = _render(group)
        if not text:
            continue

        # In-section overlap: prepend a token-bounded tail of the previous chunk's
        # prose. Never copy code, and never let a prose chunk exceed max_tokens.
        if i > 0 and policy.overlap_tokens > 0:
            has_code = any(u.is_code for u in group)
            budget = policy.overlap_tokens
            if not has_code:
                budget = min(budget, max(0, policy.max_tokens - tokenizer.count(text)))
            prev_prose = " ".join(u.text for u in groups[i - 1] if not u.is_code)
            tail = _tail_words(prev_prose, budget, tokenizer)
            if tail:
                text = f"{tail}\n\n{text}"

        if any(u.is_code for u in group) and tokenizer.count(text) > policy.max_tokens:
            log.warning(
                "chunk_oversized_code_block",
                doc_id=doc_id,
                section_index=section.index,
                tokens=tokenizer.count(text),
                max_tokens=policy.max_tokens,
            )
        spans.append(_Span(text, is_abstract=is_abstract))
    return spans


def chunk_document(
    doc: ParsedDocument,
    entry: ManifestEntry,
    tokenizer: Tokenizer,
    settings: ChunkingSettings,
) -> list[Chunk]:
    """Chunk one parsed document into retrievable chunks with full payloads."""
    policy = settings.policy_for(entry.content_type)
    doc_fields = entry.model_dump()

    chunks: list[Chunk] = []
    chunk_index = 0
    for section in doc.retrievable_sections:
        for span in _chunk_section(section, policy, settings, tokenizer, doc_id=entry.doc_id):
            chunks.append(
                Chunk(
                    **doc_fields,
                    section_index=section.index,
                    section_title=section.title,
                    section_kind=section.kind,
                    chunk_index=chunk_index,
                    text=span.text,
                    token_count=tokenizer.count(span.text),
                    content_hash=content_hash(span.text),
                    is_abstract=span.is_abstract,
                )
            )
            chunk_index += 1

    if not chunks:
        log.warning("chunk_no_retrievable_content", doc_id=entry.doc_id)
    return chunks
