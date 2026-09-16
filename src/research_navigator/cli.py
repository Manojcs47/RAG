"""Top-level Typer CLI. Entrypoints stay thin; logic lives in packages.

Commands by session: ``healthcheck`` (S1), ``parse`` (S2), ``chunk`` (S3),
``ingest`` / ``validate`` / ``reindex`` / ``stats`` (S4).
"""

from __future__ import annotations

import json
from typing import Annotated

import typer

from research_navigator.common.qdrant import check_health
from research_navigator.config import get_settings
from research_navigator.logging import configure_logging

app = typer.Typer(help="AI Research Navigator", no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def healthcheck() -> None:
    """Verify the local Qdrant instance is reachable."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    if not check_health(settings):
        raise typer.Exit(code=1)
    typer.echo("Qdrant is healthy.")


@app.command()
def parse(write: bool = True) -> None:
    """Parse the corpus into normalized IR and (optionally) cache it to disk."""
    from research_navigator.parse.dispatch import parse_corpus

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    docs = parse_corpus(settings, write=write)
    total_sections = sum(len(d.sections) for d in docs)
    with_warnings = sum(1 for d in docs if d.warnings)
    typer.echo(
        f"Parsed {len(docs)} documents, {total_sections} sections, {with_warnings} with warnings."
    )


@app.command()
def chunk() -> None:
    """Chunk the cached IR into retrievable chunks with full payloads."""
    from research_navigator.chunk.dispatch import chunk_corpus
    from research_navigator.common.manifest import load_manifest

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    manifest = load_manifest(settings.corpus_dir / "manifest.json")
    results = chunk_corpus(manifest, settings)
    total = sum(len(v) for v in results.values())
    typer.echo(f"Chunked {len(results)} documents into {total} chunks.")


@app.command()
def ingest(
    doc: Annotated[str | None, typer.Argument(help="Limit to a single doc_id.")] = None,
) -> None:
    """Embed cached chunks and idempotently upsert them into Qdrant."""
    from research_navigator.common.manifest import load_manifest
    from research_navigator.ingest import pipeline
    from research_navigator.ingest.embedder import build_embedder
    from research_navigator.ingest.factory import build_store

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    manifest = load_manifest(settings.corpus_dir / "manifest.json")
    embedder = build_embedder(settings.dense_embedding_model, settings.sparse_embedding_model)
    store = build_store(settings)
    doc_ids = [doc] if doc else None
    report = pipeline.ingest_corpus(settings, store, embedder, manifest, doc_ids=doc_ids)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def validate() -> None:
    """Check the collection matches the chunk cache (no missing/stale/orphan points)."""
    from research_navigator.common.manifest import load_manifest
    from research_navigator.ingest import pipeline
    from research_navigator.ingest.factory import build_store

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    manifest = load_manifest(settings.corpus_dir / "manifest.json")
    store = build_store(settings)
    report = pipeline.validate_corpus(settings, store, manifest)
    typer.echo(report.model_dump_json(indent=2))
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def reindex(
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation.")] = False,
) -> None:
    """Drop and recreate the collection, then re-ingest the whole corpus."""
    from research_navigator.common.manifest import load_manifest
    from research_navigator.ingest import pipeline
    from research_navigator.ingest.embedder import build_embedder
    from research_navigator.ingest.factory import build_store

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    if not yes:
        typer.confirm(f"Drop and rebuild collection {settings.collection_name!r}?", abort=True)
    manifest = load_manifest(settings.corpus_dir / "manifest.json")
    embedder = build_embedder(settings.dense_embedding_model, settings.sparse_embedding_model)
    store = build_store(settings)
    store.recreate(embedder.dense_dim)
    report = pipeline.ingest_corpus(settings, store, embedder, manifest)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def stats() -> None:
    """Print point counts overall and by content_type / year / foundational flag."""
    from research_navigator.common.manifest import load_manifest
    from research_navigator.ingest import pipeline
    from research_navigator.ingest.factory import build_store

    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    manifest = load_manifest(settings.corpus_dir / "manifest.json")
    store = build_store(settings)
    typer.echo(json.dumps(pipeline.collection_stats(store, manifest).model_dump(), indent=2))


if __name__ == "__main__":
    app()
