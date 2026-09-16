"""Corpus-level chunking: read cached IR -> chunk -> cache chunks (Session 3).

Mirrors the Session 2 parse dispatch: loud logging, skip-on-missing (never crash
the whole run), and a JSON cache per document at ``chunk_dir/{doc_id}.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from research_navigator.chunk.chunker import chunk_document
from research_navigator.chunk.models import Chunk
from research_navigator.chunk.tokenizer import Tokenizer, build_tokenizer
from research_navigator.common.manifest import Manifest
from research_navigator.config import Settings
from research_navigator.parse.models import ParsedDocument

log = structlog.get_logger(__name__)


def load_parsed(path: Path) -> ParsedDocument:
    return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    payload = [chunk.payload() for chunk in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_corpus(
    manifest: Manifest,
    settings: Settings,
    tokenizer: Tokenizer | None = None,
) -> dict[str, list[Chunk]]:
    """Chunk every document with a cached IR; return ``{doc_id: chunks}``."""
    tok = tokenizer or build_tokenizer(settings.chunking.tokenizer_model)
    settings.paths.chunks_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Chunk]] = {}
    for entry in manifest.documents:
        parsed_path = settings.paths.parsed_dir / f"{entry.doc_id}.json"
        if not parsed_path.exists():
            log.warning("chunk_parsed_missing", doc_id=entry.doc_id, path=str(parsed_path))
            continue
        doc = load_parsed(parsed_path)
        chunks = chunk_document(doc, entry, tok, settings.chunking)
        write_chunks(settings.paths.chunks_dir / f"{entry.doc_id}.json", chunks)
        results[entry.doc_id] = chunks
        log.info("chunk_document_done", doc_id=entry.doc_id, n_chunks=len(chunks))

    total = sum(len(c) for c in results.values())
    log.info("chunk_corpus_done", documents=len(results), total_chunks=total)
    return results
