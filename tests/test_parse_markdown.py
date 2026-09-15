from __future__ import annotations

from research_navigator.common.types import ContentType
from research_navigator.parse.markdown import parse_markdown
from research_navigator.parse.models import BlockType, SectionKind

MD = """---
title: Sample
---

# Chapter Title

Intro paragraph about transformers.

## Section One

Some explanatory text here.

```python
print("hello")
```

## References

- Vaswani et al., 2017
"""


def test_sections_and_kinds() -> None:
    doc = parse_markdown("hf-x", ContentType.course_chapter, MD, "Sample")
    titles = [s.title for s in doc.sections]
    assert "Section One" in titles
    refs = [s for s in doc.sections if s.kind is SectionKind.references]
    assert refs, "references section should be tagged"
    assert all(s.kind is not SectionKind.references for s in doc.retrievable_sections)


def test_code_block_preserved() -> None:
    doc = parse_markdown("hf-x", ContentType.course_chapter, MD, "Sample")
    codes = [b for s in doc.sections for b in s.blocks if b.type is BlockType.code]
    assert codes and codes[0].language == "python"
    assert 'print("hello")' in codes[0].text
