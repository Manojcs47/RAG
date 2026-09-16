"""Unit tests for content-type-aware chunking (Session 3, M1b)."""

from __future__ import annotations

import pytest

from research_navigator.chunk.chunker import chunk_document
from research_navigator.chunk.settings import ChunkingSettings, ChunkPolicy
from research_navigator.chunk.tokenizer import WhitespaceTokenizer
from research_navigator.common.manifest import ManifestEntry
from research_navigator.common.types import ContentType
from research_navigator.parse.models import (
    Block,
    BlockType,
    ParsedDocument,
    Section,
    SectionKind,
)

TOK = WhitespaceTokenizer()


def _entry(content_type: ContentType = ContentType.arxiv_paper) -> ManifestEntry:
    return ManifestEntry(
        doc_id="arxiv-1706.03762",
        content_type=content_type,
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar"],
        year=2017,
        month=None,
        primary_category="cs.CL",
        secondary_categories=["cs.LG"],
        tags=["transformers", "attention"],
        is_foundational=True,
        citation_count=100000,
        source_url="https://arxiv.org/abs/1706.03762",
        local_path="corpus/arxiv-1706.03762.pdf",
    )


def _para(n_words: int, word: str = "w") -> str:
    # Deterministic prose made of single-word "sentences" so word-count == token-count.
    return " ".join(f"{word}{i}." for i in range(n_words))


def _tiny_settings() -> ChunkingSettings:
    policy = ChunkPolicy(target_tokens=10, max_tokens=16, overlap_tokens=3, min_tokens=3)
    return ChunkingSettings(default=policy, per_content_type={}, abstract_max_tokens=50)


def test_determinism_same_input_same_output() -> None:
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="Abstract",
                level=1,
                kind=SectionKind.abstract,
                blocks=[Block(type=BlockType.paragraph, text=_para(8))],
            ),
            Section(
                index=1,
                title="Intro",
                level=1,
                kind=SectionKind.body,
                blocks=[Block(type=BlockType.paragraph, text=_para(60))],
            ),
        ],
    )
    a = chunk_document(doc, _entry(), TOK, _tiny_settings())
    b = chunk_document(doc, _entry(), TOK, _tiny_settings())
    assert [c.uid for c in a] == [c.uid for c in b]
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.content_hash for c in a] == [c.content_hash for c in b]


def test_abstract_is_single_flagged_chunk() -> None:
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="Abstract",
                level=1,
                kind=SectionKind.abstract,
                blocks=[Block(type=BlockType.paragraph, text=_para(30))],
            )
        ],
    )
    chunks = chunk_document(doc, _entry(), TOK, _tiny_settings())
    abstracts = [c for c in chunks if c.is_abstract]
    assert len(abstracts) == 1  # emitted whole despite exceeding the body target
    assert abstracts[0].section_kind is SectionKind.abstract


def test_code_block_is_never_split_and_kept_verbatim() -> None:
    code = "def f():\n    " + "x = 1\n    " * 40 + "return x"
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.course_chapter,
        title="t",
        sections=[
            Section(
                index=0,
                title="Example",
                level=2,
                kind=SectionKind.body,
                blocks=[Block(type=BlockType.code, text=code, language="python")],
            )
        ],
    )
    chunks = chunk_document(doc, _entry(ContentType.course_chapter), TOK, _tiny_settings())
    holders = [c for c in chunks if "def f():" in c.text]
    assert len(holders) == 1  # the whole code block lives in exactly one chunk
    assert code in holders[0].text  # byte-for-byte intact


def test_references_excluded_but_body_kept() -> None:
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="Body",
                level=1,
                kind=SectionKind.body,
                blocks=[Block(type=BlockType.paragraph, text="Real body content here.")],
            ),
            Section(
                index=1,
                title="References",
                level=1,
                kind=SectionKind.references,
                blocks=[Block(type=BlockType.paragraph, text="Vaswani et al 2017 Attention.")],
            ),
        ],
    )
    chunks = chunk_document(doc, _entry(), TOK, _tiny_settings())
    assert all(c.section_kind is not SectionKind.references for c in chunks)
    assert not any("Vaswani et al 2017" in c.text for c in chunks)
    assert any("Real body content" in c.text for c in chunks)


def test_chunks_respect_section_boundaries_and_index_monotonic() -> None:
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="A",
                level=1,
                blocks=[Block(type=BlockType.paragraph, text=_para(40, "a"))],
            ),
            Section(
                index=1,
                title="B",
                level=1,
                blocks=[Block(type=BlockType.paragraph, text=_para(40, "b"))],
            ),
        ],
    )
    chunks = chunk_document(doc, _entry(), TOK, _tiny_settings())
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        letter = "a" if c.section_index == 0 else "b"
        other = "b" if letter == "a" else "a"
        assert other not in c.text  # no cross-section bleed


def test_size_discipline_prose_under_max() -> None:
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="Long",
                level=1,
                blocks=[Block(type=BlockType.paragraph, text=_para(200))],
            )
        ],
    )
    settings = _tiny_settings()
    chunks = chunk_document(doc, _entry(), TOK, settings)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= settings.default.max_tokens


def test_overlap_shares_context_between_adjacent_prose_chunks() -> None:
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="Long",
                level=1,
                blocks=[Block(type=BlockType.paragraph, text=_para(60))],
            )
        ],
    )
    chunks = chunk_document(doc, _entry(), TOK, _tiny_settings())
    assert len(chunks) >= 2
    first_words = set(chunks[0].text.split())
    second_words = chunks[1].text.split()
    assert any(w in first_words for w in second_words)  # tail carried forward


def test_payload_carries_every_manifest_field() -> None:
    entry = _entry()
    doc = ParsedDocument(
        doc_id="d",
        content_type=ContentType.arxiv_paper,
        title="t",
        sections=[
            Section(
                index=0,
                title="Body",
                level=1,
                blocks=[Block(type=BlockType.paragraph, text="Some content.")],
            )
        ],
    )
    chunk = chunk_document(doc, entry, TOK, _tiny_settings())[0]
    payload = chunk.payload()
    for field in ManifestEntry.model_fields:
        assert field in payload
    assert payload["tags"] == ["transformers", "attention"]
    assert payload["is_foundational"] is True
    assert payload["month"] is None
    # derived fields present too
    for derived in ("section_title", "section_index", "chunk_index", "content_hash"):
        assert derived in payload


def test_empty_document_yields_no_chunks() -> None:
    doc = ParsedDocument(doc_id="d", content_type=ContentType.arxiv_paper, title="t", sections=[])
    assert chunk_document(doc, _entry(), TOK, _tiny_settings()) == []


@pytest.mark.parametrize("content_type", list(ContentType))
def test_all_content_types_have_a_policy(content_type: ContentType) -> None:
    settings = ChunkingSettings()
    assert settings.policy_for(content_type).max_tokens > 0
