"""Markdown parsing for course chapters, Lil'Log surveys, and lab blog posts."""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from research_navigator.common.types import ContentType
from research_navigator.parse.models import Block, BlockType, ParsedDocument, Section, SectionKind

_FRONTMATTER = re.compile(r"^\ufeff?---\n.*?\n---\n", re.DOTALL)
_LINK_ONLY = re.compile(r"^!?\[[^\]]*\]\([^)]*\)$")
_NAV = re.compile(r"(share on|posted on|table of contents|^tags:|←|→)", re.IGNORECASE)

_md = MarkdownIt("commonmark", {"html": False})


def _strip_frontmatter(raw: str) -> str:
    return _FRONTMATTER.sub("", raw, count=1)


def _classify_kind(title: str) -> SectionKind:
    t = title.strip().lower()
    if t == "abstract":
        return SectionKind.abstract
    if t in {"references", "reference", "bibliography"}:
        return SectionKind.references
    return SectionKind.body


def _is_noise(text: str) -> bool:
    return bool(_LINK_ONLY.match(text)) or (bool(_NAV.search(text)) and len(text) < 80)


def parse_markdown(doc_id: str, content_type: ContentType, raw: str, title: str) -> ParsedDocument:
    tokens = _md.parse(_strip_frontmatter(raw))

    sections: list[Section] = []
    current = Section(index=0, title="", level=1, kind=SectionKind.body)
    list_depth = 0
    dropped = 0

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            if current.blocks or current.title:
                sections.append(current)
            heading_text = tokens[i + 1].content.strip() if i + 1 < len(tokens) else ""
            level = int(tok.tag[1])
            current = Section(
                index=len(sections),
                title=heading_text,
                level=level,
                kind=_classify_kind(heading_text),
            )
            i += 3  # heading_open, inline, heading_close
            continue
        if tok.type in {"fence", "code_block"}:
            lang = tok.info.split()[0] if (tok.type == "fence" and tok.info) else None
            current.blocks.append(
                Block(type=BlockType.code, text=tok.content.rstrip("\n"), language=lang)
            )
        elif tok.type in {"bullet_list_open", "ordered_list_open"}:
            list_depth += 1
        elif tok.type in {"bullet_list_close", "ordered_list_close"}:
            list_depth = max(0, list_depth - 1)
        elif tok.type == "inline":
            text = tok.content.strip()
            if not text:
                i += 1
                continue
            if _is_noise(text):
                dropped += 1
                i += 1
                continue
            btype = BlockType.list_item if list_depth > 0 else BlockType.paragraph
            current.blocks.append(Block(type=btype, text=text))
        i += 1

    if current.blocks or current.title:
        sections.append(current)
    sections = [s for s in sections if s.blocks or s.title]
    for n, s in enumerate(sections):
        s.index = n

    warnings: list[str] = []
    if dropped:
        warnings.append(f"dropped {dropped} navigation/link-only blocks")
    if not sections:
        warnings.append("no content extracted")

    return ParsedDocument(
        doc_id=doc_id,
        content_type=content_type,
        title=title,
        sections=sections,
        warnings=warnings,
    )
