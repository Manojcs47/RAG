"""Dispatch parsing by content type and drive whole-corpus parsing."""

from __future__ import annotations

from pathlib import Path

from research_navigator.common.manifest import ManifestEntry, load_manifest
from research_navigator.common.types import ContentType
from research_navigator.config import Settings
from research_navigator.logging import get_logger
from research_navigator.parse.markdown import parse_markdown
from research_navigator.parse.models import ParsedDocument
from research_navigator.parse.pdf import parse_pdf

log = get_logger(__name__)


def _resolve(local_path: str, corpus_root: Path, repo_root: Path) -> Path:
    for base in (corpus_root, repo_root):
        candidate = base / local_path
        if candidate.exists():
            return candidate
    return corpus_root / local_path  # non-existent; used for the error message


def parse_document(entry: ManifestEntry, corpus_root: Path, repo_root: Path) -> ParsedDocument:
    path = _resolve(entry.local_path, corpus_root, repo_root)
    if entry.content_type is ContentType.arxiv_paper:
        return parse_pdf(entry.doc_id, path, entry.title)
    raw = path.read_text(encoding="utf-8")
    return parse_markdown(entry.doc_id, entry.content_type, raw, entry.title)


def parse_corpus(settings: Settings, *, write: bool = True) -> list[ParsedDocument]:
    corpus_root = settings.corpus_dir
    repo_root = Path(".")
    manifest = load_manifest(corpus_root / "manifest.json")

    if write:
        settings.parsed_dir.mkdir(parents=True, exist_ok=True)

    results: list[ParsedDocument] = []
    for entry in manifest.documents:
        try:
            parsed = parse_document(entry, corpus_root, repo_root)
        except FileNotFoundError:
            log.error("parse_file_missing", doc_id=entry.doc_id, local_path=entry.local_path)
            continue
        except Exception as exc:
            log.error("parse_failed", doc_id=entry.doc_id, error=str(exc))
            continue

        if parsed.warnings:
            log.warning("parse_warnings", doc_id=entry.doc_id, warnings=parsed.warnings)
        if write:
            out = settings.parsed_dir / f"{entry.doc_id}.json"
            out.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        results.append(parsed)

    log.info(
        "parse_corpus_done",
        parsed=len(results),
        total=len(manifest.documents),
        with_warnings=sum(1 for d in results if d.warnings),
    )
    return results
