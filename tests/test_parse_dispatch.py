from __future__ import annotations

from pathlib import Path

from research_navigator.common.manifest import ManifestEntry
from research_navigator.common.types import ContentType
from research_navigator.parse.dispatch import parse_document


def test_dispatch_markdown(tmp_path: Path) -> None:
    doc_dir = tmp_path / "documents" / "hf-learn"
    doc_dir.mkdir(parents=True)
    (doc_dir / "hf-x.md").write_text("# Title\n\nHello world.\n", encoding="utf-8")

    entry = ManifestEntry(
        doc_id="hf-x",
        content_type=ContentType.course_chapter,
        title="Title",
        year=2024,
        primary_category="course",
        source_url="https://example.com",
        local_path="documents/hf-learn/hf-x.md",
    )
    parsed = parse_document(entry, corpus_root=tmp_path, repo_root=tmp_path)
    assert parsed.doc_id == "hf-x"
    assert parsed.sections
