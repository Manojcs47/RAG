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
from pathlib import Path
from typing import Any

import typer

from research_navigator.agents import build_agent
from research_navigator.common.seeds import seed_everything

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


@app.callback()
def _main() -> None:
    """Runs before every command: pin RNG seeds for reproducibility (M5, ADR-0012)."""
    settings = get_settings()
    if settings.repro.seed_on_startup:
        seed_everything(settings.repro.seed)


@app.command()
def healthcheck() -> None:
    """Check that the configured Qdrant instance is reachable."""
    from research_navigator.common.qdrant import check_health

    settings = get_settings()
    if not check_health(settings):
        typer.echo(f"Qdrant unreachable at {settings.qdrant.url}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Qdrant reachable at {settings.qdrant.url}.")


# --------------------------------------------------------------------------- #
# Corpus preparation (M1a parse, M1b chunk) — run once per corpus change,
# before `ingest`. Cached to data/parsed/ and data/chunks/ respectively.
# --------------------------------------------------------------------------- #
@app.command()
def parse() -> None:
    """Parse the corpus (PDF + Markdown) into data/parsed/*.json."""
    from research_navigator.parse.dispatch import parse_corpus

    settings = get_settings()
    results = parse_corpus(settings)
    with_warnings = sum(1 for r in results if r.warnings)
    typer.echo(
        f"parsed {len(results)} document(s) -> {settings.paths.parsed_dir} "
        f"({with_warnings} with warnings; see logs)"
    )


@app.command()
def chunk() -> None:
    """Chunk cached parsed documents into data/chunks/*.json."""
    from research_navigator.chunk.dispatch import chunk_corpus
    from research_navigator.common.manifest import load_manifest

    settings = get_settings()
    manifest = load_manifest(settings.paths.manifest)
    results = chunk_corpus(manifest, settings)
    total = sum(len(v) for v in results.values())
    typer.echo(f"chunked {len(results)} document(s), {total} chunks -> {settings.paths.chunks_dir}")


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


# --------------------------------------------------------------------------- #
# Evaluation (M4)
# --------------------------------------------------------------------------- #
@app.command("eval")
def evaluate(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run the harness on in-repo fakes (no Qdrant/OpenAI). Ideal for CI.",
    ),
    golden: str | None = typer.Option(
        None, "--golden", help="Golden-set JSON path. Defaults to the packaged set."
    ),
    out: str | None = typer.Option(
        None, "--out", help="Output dir for report.json/report.md (default from settings)."
    ),
) -> None:
    """Evaluate the Navigator over the golden set; write JSON + one-page Markdown (M4).

    `make eval` runs the live sweep (needs OPENAI_API_KEY + a running Qdrant + FastEmbed
    egress); `--dry-run` runs the identical flow on fakes with no external services.
    """
    # Local imports: eval is a leaf package and the dry-run path must not touch the stack.
    from research_navigator.eval import load_golden_set, run_eval

    settings = get_settings()
    eval_settings = settings.eval
    if out is not None:
        eval_settings = eval_settings.model_copy(update={"report_dir": Path(out)})

    golden_path = Path(golden) if golden is not None else eval_settings.golden_set_path
    golden_set = load_golden_set(golden_path)

    if dry_run:
        from research_navigator.eval.factory import build_dry_run_builder

        builder = build_dry_run_builder(golden_set)
    else:
        from research_navigator.eval.factory import build_real_builder

        builder = build_real_builder(settings)

    report = run_eval(builder=builder, golden=golden_set, settings=eval_settings)
    for path in report.write(eval_settings):
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
