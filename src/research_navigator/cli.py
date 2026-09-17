"""Thin Typer CLI. No business logic here — every command wires config +
factories + pipeline/retrieval and prints the result.

Expected S4 factory/pipeline surface (adjust import names to match your repo):
  ingest.factory.build_client(qdrant_settings) -> QdrantClient
  ingest.factory.build_store(client, settings, dense_dim) -> RnStore
  ingest.embedder.build_embedder(embedding_settings) -> Embedder
  ingest.pipeline.ingest_corpus(store, embedder, settings) -> IngestReport
  ingest.pipeline.ingest_doc(store, embedder, settings, doc_id) -> DocIngestResult
  ingest.pipeline.validate_corpus(store, settings) -> ValidationReport   (.ok: bool)
  ingest.pipeline.collection_stats(store) -> CollectionStats
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import typer

from research_navigator.agents import build_agent

from .config import Settings, get_settings
from .generate import build_generator, build_llm
from .ingest.embedder import Embedder, build_embedder
from .ingest.factory import build_client, build_store
from .ingest.pipeline import (
    collection_stats,
    ingest_corpus,
    ingest_doc,
    validate_corpus,
)
from .ingest.store import RnStore
from .retrieve import analyze as analyze_query
from .retrieve import build_retriever

app = typer.Typer(add_completion=False, help="AI Research Navigator CLI.")


def _dump(obj: Any) -> str:
    """Best-effort structured print for dataclasses / pydantic / plain objects."""
    value: object = obj
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif hasattr(value, "model_dump"):
        value = value.model_dump()
    return json.dumps(value, indent=2, default=str)


def _wire(settings: Settings) -> tuple[Embedder, RnStore]:
    embedder = build_embedder(settings.embedding)
    client = build_client(settings.qdrant)
    store = build_store(client, settings, embedder.dense_dim)
    return embedder, store


# --------------------------------------------------------------------------- #
# Ingestion (M1)
# --------------------------------------------------------------------------- #
@app.command()
def ingest(doc: str | None = typer.Option(None, help="Ingest a single doc_id.")) -> None:
    """Ingest the corpus (or one document) into Qdrant."""
    settings = get_settings()
    embedder, store = _wire(settings)
    if doc is not None:
        typer.echo(_dump(ingest_doc(store, embedder, settings, doc)))
    else:
        typer.echo(_dump(ingest_corpus(store, embedder, settings)))


@app.command()
def validate() -> None:
    """Validate the collection against the corpus; exit != 0 on drift."""
    settings = get_settings()
    _, store = _wire(settings)
    report = validate_corpus(store, settings)
    typer.echo(_dump(report))
    if not getattr(report, "ok", False):
        raise typer.Exit(code=1)


@app.command()
def reindex(yes: bool = typer.Option(False, "--yes", help="Skip confirmation.")) -> None:
    """Drop and rebuild the collection from scratch."""
    settings = get_settings()
    if not yes:
        typer.confirm(
            f"This will DROP and rebuild '{settings.qdrant.collection}'. Continue?",
            abort=True,
        )
    embedder, store = _wire(settings)
    store.recreate()
    result = ingest_corpus(store, embedder, settings)
    typer.echo(_dump(result))


@app.command()
def stats() -> None:
    """Report chunk counts by content_type, year, and tags."""
    settings = get_settings()
    _, store = _wire(settings)
    typer.echo(_dump(collection_stats(store, settings)))


# --------------------------------------------------------------------------- #
# Retrieval (M2)
# --------------------------------------------------------------------------- #
@app.command()
def analyze(query: str) -> None:
    """Show inferred intent + metadata filters for a query (no retrieval)."""
    settings = get_settings()
    from .retrieve import load_catalog

    catalog = load_catalog(settings.paths.manifest)
    a = analyze_query(query, catalog=catalog, settings=settings.retrieve, now_year=None)
    typer.echo(
        _dump(
            {
                "intent": a.intent.value,
                "filters": dataclasses.asdict(a.filters),
                "reasons": list(a.reasons),
            }
        )
    )


@app.command()
def search(
    query: str,
    k: int | None = typer.Option(None, help="Override top_k."),
    dense_only: bool = typer.Option(False, help="Disable the sparse branch."),
    fusion: str | None = typer.Option(None, help="rrf | dbsf."),
) -> None:
    """Run the M2 retrieval pipeline: print ranked chunks or a refusal."""
    settings = get_settings()
    updates: dict[str, Any] = {"dense_only": dense_only}
    if k is not None:
        updates["top_k"] = k
    if fusion is not None:
        updates["fusion"] = fusion
    retrieve_settings = settings.retrieve.model_copy(update=updates)

    embedder, store = _wire(settings)
    retriever = build_retriever(
        embedder=embedder,
        searcher=store,
        manifest_path=settings.paths.manifest,
        settings=retrieve_settings,
    )
    result = retriever.retrieve(query)

    if result.refused:
        typer.echo(
            "REFUSED: I don't have enough relevant material in the corpus to answer "
            f"this confidently (confidence={result.confidence:.3f} < "
            f"{retrieve_settings.refusal_threshold})."
        )
        return
    typer.echo(
        f"intent={result.analysis.intent.value}  filtered={result.filtered}  "
        f"confidence={result.confidence:.3f}  fusion={result.fusion}"
    )
    for i, c in enumerate(result.chunks, 1):
        typer.echo(
            f"[{i}] {c.doc_id} · {c.title} · §{c.section_title} ({c.year})  fused={c.score:.4f}"
        )


@app.command()
def answer(
    query: str,
    k: int | None = typer.Option(None, help="Override top_k."),
    dense_only: bool = typer.Option(False, help="Disable the sparse branch."),
    fusion: str | None = typer.Option(None, help="rrf | dbsf."),
) -> None:
    """Full M2 pipeline: retrieve -> grounded, cited answer (or refusal)."""
    settings = get_settings()
    updates: dict[str, Any] = {"dense_only": dense_only}
    if k is not None:
        updates["top_k"] = k
    if fusion is not None:
        updates["fusion"] = fusion
    retrieve_settings = settings.retrieve.model_copy(update=updates)

    embedder, store = _wire(settings)
    retriever = build_retriever(
        embedder=embedder,
        searcher=store,
        manifest_path=settings.paths.manifest,
        settings=retrieve_settings,
    )
    generator = build_generator(
        retriever=retriever,
        llm=build_llm(settings.llm),
        settings=settings.generate,
    )
    typer.echo(generator.answer(query).render())


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question."),
    now_year: int | None = typer.Option(
        None, "--now-year", help="Override 'current' year for recency routing/tests."
    ),
) -> None:
    """Route the query through the M3 agent and print the (cited or refused) answer."""
    settings = get_settings()
    embedder, store = _wire(settings)
    llm = build_llm(settings.llm)
    retriever = build_retriever(
        embedder=embedder,
        searcher=store,
        manifest_path=settings.paths.manifest,
        settings=settings.retrieve,
    )
    generator = build_generator(retriever=retriever, llm=llm, settings=settings.generate)
    # `llm` is reused for the router's ambiguous-query fallback (rules run first, no cost
    # on the common path). Pass llm=None to force rules-only routing.
    agent = build_agent(
        generator=generator,
        manifest_path=settings.paths.manifest,
        llm=llm if settings.agents.use_llm_router else None,
        settings=settings.agents,
    )
    result = agent.run(query, now_year=now_year)
    typer.echo(result.render())


@app.command()
def route(
    query: str = typer.Argument(..., help="Query to classify."),
) -> None:
    """Show ONLY the routing decision (route · reason · confidence) — no retrieval/LLM
    answer generation. Useful for building the >=3-queries-per-route test table."""
    settings = get_settings()
    llm = build_llm(settings.llm) if settings.agents.use_llm_router else None
    from research_navigator.agents import Router  # local import: routing needs no corpus

    router = Router(settings=settings.agents, llm=llm)
    route_name, reason, confidence = router.route(query)
    typer.echo(f"{route_name}\t{confidence:.2f}\t{reason}")


if __name__ == "__main__":
    app()
