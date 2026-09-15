"""Intermediate representation (IR) for a parsed document.

This is the ONLY interface the chunker (Session 3) depends on. Parsers for new
content types must emit this shape; nothing downstream reads raw PDF/Markdown.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from research_navigator.common.types import ContentType


class BlockType(StrEnum):
    paragraph = "paragraph"
    code = "code"
    list_item = "list"
    caption = "caption"


class SectionKind(StrEnum):
    abstract = "abstract"
    body = "body"
    references = "references"
    appendix = "appendix"


class Block(BaseModel):
    type: BlockType
    text: str
    language: str | None = None  # for code fences


class Section(BaseModel):
    index: int
    title: str
    level: int
    kind: SectionKind = SectionKind.body
    blocks: list[Block] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks).strip()


class ParsedDocument(BaseModel):
    doc_id: str
    content_type: ContentType
    title: str
    sections: list[Section] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def retrievable_sections(self) -> list[Section]:
        """Sections eligible for retrieval (references excluded but retained on doc)."""
        return [s for s in self.sections if s.kind is not SectionKind.references]
