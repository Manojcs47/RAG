"""Integration tests for the Typer CLI (M5: "integration tests for the
ingestion CLI and the LangGraph router").

Every command wires real factories/pipelines (Qdrant, FastEmbed, the LLM), so
each test here patches exactly those seams with fakes/real-but-cheap objects
and drives the command through Typer's CliRunner — no network, no Qdrant,
no LLM calls. This exercises the CLI's own argument handling, output
rendering, and exit codes, which the unit tests for the underlying modules
don't cover.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from research_navigator import cli
from research_navigator.ingest.models import (
    CollectionStats,
    DocIngestResult,
    IngestReport,
    ValidationIssue,
    ValidationReport,
)
from research_navigator.retrieve.models import (
    InferredFilters,
    QueryAnalysis,
    QueryIntent,
    RetrievalResult,
    RetrievedChunk,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _fake_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every command starts with `_wire()` -> (embedder, store); neither must
    touch FastEmbed or Qdrant in these tests."""
    monkeypatch.setattr(cli, "build_embedder", lambda _settings: SimpleNamespace(dense_dim=384))
    monkeypatch.setattr(cli, "build_client", lambda _settings: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "build_store", lambda _client, _settings, _dim: SimpleNamespace())


# --------------------------------------------------------------------------- #
# healthcheck
# --------------------------------------------------------------------------- #
def test_healthcheck_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_navigator.common import qdrant

    monkeypatch.setattr(qdrant, "check_health", lambda _settings: True)
    result = runner.invoke(cli.app, ["healthcheck"])
    assert result.exit_code == 0
    assert "reachable" in result.stdout


def test_healthcheck_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_navigator.common import qdrant

    monkeypatch.setattr(qdrant, "check_health", lambda _settings: False)
    result = runner.invoke(cli.app, ["healthcheck"])
    assert result.exit_code == 1
    assert "unreachable" in result.output


# --------------------------------------------------------------------------- #
# parse / chunk (M1a/M1b)
# --------------------------------------------------------------------------- #
def test_parse_reports_document_and_warning_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_navigator.parse import dispatch as parse_dispatch

    docs = [SimpleNamespace(warnings=[]), SimpleNamespace(warnings=["dropped a figure"])]
    monkeypatch.setattr(parse_dispatch, "parse_corpus", lambda _settings: docs)
    result = runner.invoke(cli.app, ["parse"])
    assert result.exit_code == 0
    assert "parsed 2 document(s)" in result.stdout
    assert "1 with warnings" in result.stdout


def test_chunk_reports_document_and_chunk_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_navigator.chunk import dispatch as chunk_dispatch
    from research_navigator.common import manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "load_manifest", lambda _path: object())
    monkeypatch.setattr(
        chunk_dispatch, "chunk_corpus", lambda _manifest, _settings: {"a": [1, 2], "b": [3]}
    )
    result = runner.invoke(cli.app, ["chunk"])
    assert result.exit_code == 0
    assert "chunked 2 document(s), 3 chunks" in result.stdout


# --------------------------------------------------------------------------- #
# ingest / validate / reindex / stats (M1c-f, required by the problem
# statement's "1f. CLI" acceptance: ingest, validate, reindex, stats)
# --------------------------------------------------------------------------- #
def test_ingest_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    report = IngestReport(
        collection="rn", docs=[DocIngestResult(doc_id="d1", added=3, deleted=0, unchanged=0)]
    )
    monkeypatch.setattr(cli, "ingest_corpus", lambda _store, _embedder, _settings: report)
    result = runner.invoke(cli.app, ["ingest"])
    assert result.exit_code == 0
    assert '"writes": 3' in result.stdout


def test_ingest_single_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    doc_result = DocIngestResult(doc_id="d1", added=1, deleted=1, unchanged=2)
    calls: list[str] = []

    def fake_ingest_doc(
        _store: Any, _embedder: Any, _settings: Any, doc_id: str
    ) -> DocIngestResult:
        calls.append(doc_id)
        return doc_result

    monkeypatch.setattr(cli, "ingest_doc", fake_ingest_doc)
    result = runner.invoke(cli.app, ["ingest", "--doc", "d1"])
    assert result.exit_code == 0
    assert calls == ["d1"]
    assert '"writes": 2' in result.stdout


def test_validate_ok_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    report = ValidationReport(
        collection="rn", ok=True, expected_points=10, actual_points=10, issues=[]
    )
    monkeypatch.setattr(cli, "validate_corpus", lambda _store, _settings: report)
    result = runner.invoke(cli.app, ["validate"])
    assert result.exit_code == 0


def test_validate_drift_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    report = ValidationReport(
        collection="rn",
        ok=False,
        expected_points=10,
        actual_points=8,
        issues=[ValidationIssue(doc_id="d1", kind="missing_points", detail="2 missing")],
    )
    monkeypatch.setattr(cli, "validate_corpus", lambda _store, _settings: report)
    result = runner.invoke(cli.app, ["validate"])
    assert result.exit_code == 1
    assert "missing_points" in result.stdout


def test_reindex_requires_confirmation_without_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    recreate_called = []
    monkeypatch.setattr(cli, "ingest_corpus", lambda *_a: IngestReport(collection="rn", docs=[]))

    def fake_build_store(_client: Any, _settings: Any, _dim: Any) -> SimpleNamespace:
        return SimpleNamespace(recreate=lambda: recreate_called.append(True))

    monkeypatch.setattr(cli, "build_store", fake_build_store)
    result = runner.invoke(cli.app, ["reindex"], input="n\n")
    assert result.exit_code != 0
    assert not recreate_called


def test_reindex_yes_skips_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    recreate_called = []
    monkeypatch.setattr(cli, "ingest_corpus", lambda *_a: IngestReport(collection="rn", docs=[]))

    def fake_build_store(_client: Any, _settings: Any, _dim: Any) -> SimpleNamespace:
        return SimpleNamespace(recreate=lambda: recreate_called.append(True))

    monkeypatch.setattr(cli, "build_store", fake_build_store)
    result = runner.invoke(cli.app, ["reindex", "--yes"])
    assert result.exit_code == 0
    assert recreate_called == [True]


def test_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = CollectionStats(
        collection="rn",
        total_points=42,
        by_content_type={"arxiv_paper": 42},
        by_year={2024: 42},
        foundational=5,
    )
    monkeypatch.setattr(cli, "collection_stats", lambda _store, _settings: stats)
    result = runner.invoke(cli.app, ["stats"])
    assert result.exit_code == 0
    assert '"total_points": 42' in result.stdout


# --------------------------------------------------------------------------- #
# analyze / search (M2)
# --------------------------------------------------------------------------- #
def test_analyze(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_navigator import retrieve as retrieve_pkg

    analysis = QueryAnalysis(
        raw="recent RAG papers",
        normalized="recent rag papers",
        intent=QueryIntent.RECENT,
        filters=InferredFilters(year_gte=2024, tags=("RAG",)),
        reasons=("recency cue -> year>=2024",),
    )
    monkeypatch.setattr(retrieve_pkg, "load_catalog", lambda _path: object())
    monkeypatch.setattr(cli, "analyze_query", lambda _q, catalog, settings, now_year: analysis)
    result = runner.invoke(cli.app, ["analyze", "recent RAG papers"])
    assert result.exit_code == 0
    assert '"intent": "recent"' in result.stdout
    assert '"year_gte": 2024' in result.stdout


def test_search_refuses_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    refused = RetrievalResult(
        analysis=QueryAnalysis(
            raw="q", normalized="q", intent=QueryIntent.CONCEPT, filters=InferredFilters()
        ),
        chunks=(),
        confidence=0.1,
        refused=True,
        fusion="rrf",
        dense_only=False,
        top_k=8,
        filtered=False,
    )
    monkeypatch.setattr(
        cli, "build_retriever", lambda **_kw: SimpleNamespace(retrieve=lambda _q: refused)
    )
    result = runner.invoke(cli.app, ["search", "off corpus question"])
    assert result.exit_code == 0
    assert "REFUSED" in result.stdout


def test_search_prints_ranked_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = RetrievedChunk(
        point_id="p1",
        score=0.9,
        payload={"doc_id": "d1", "title": "Some Paper", "section_title": "Abstract", "year": 2024},
    )
    ok = RetrievalResult(
        analysis=QueryAnalysis(
            raw="q", normalized="q", intent=QueryIntent.CONCEPT, filters=InferredFilters()
        ),
        chunks=(hit,),
        confidence=0.8,
        refused=False,
        fusion="rrf",
        dense_only=False,
        top_k=8,
        filtered=True,
    )
    monkeypatch.setattr(
        cli, "build_retriever", lambda **_kw: SimpleNamespace(retrieve=lambda _q: ok)
    )
    result = runner.invoke(cli.app, ["search", "what is rag"])
    assert result.exit_code == 0
    assert "Some Paper" in result.stdout


# --------------------------------------------------------------------------- #
# answer / ask / route (M2 generation, M3 agent)
# --------------------------------------------------------------------------- #
def test_answer_renders_generator_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_llm", lambda _settings: SimpleNamespace())
    monkeypatch.setattr(cli, "build_retriever", lambda **_kw: SimpleNamespace())
    fake_answer = SimpleNamespace(render=lambda: "GROUNDED ANSWER [1]")
    monkeypatch.setattr(
        cli, "build_generator", lambda **_kw: SimpleNamespace(answer=lambda _q: fake_answer)
    )
    result = runner.invoke(cli.app, ["answer", "what is rag"])
    assert result.exit_code == 0
    assert "GROUNDED ANSWER [1]" in result.stdout


def test_ask_routes_through_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_llm", lambda _settings: SimpleNamespace())
    monkeypatch.setattr(cli, "build_retriever", lambda **_kw: SimpleNamespace())
    monkeypatch.setattr(cli, "build_generator", lambda **_kw: SimpleNamespace())
    calls: dict[str, Any] = {}

    def fake_build_agent(**_kw: Any) -> SimpleNamespace:
        def run(query: str, now_year: int | None = None) -> SimpleNamespace:
            calls["query"] = query
            calls["now_year"] = now_year
            return SimpleNamespace(render=lambda: "[route: concept_explanation] answer")

        return SimpleNamespace(run=run)

    monkeypatch.setattr(cli, "build_agent", fake_build_agent)
    result = runner.invoke(cli.app, ["ask", "what is a transformer", "--now-year", "2026"])
    assert result.exit_code == 0
    assert calls == {"query": "what is a transformer", "now_year": 2026}
    assert "concept_explanation" in result.stdout


def test_route_prints_decision_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    from research_navigator import agents as agents_pkg

    monkeypatch.setattr(cli, "build_llm", lambda _settings: SimpleNamespace())

    class FakeRouter:
        def __init__(self, *, settings: Any, llm: Any) -> None:
            pass

        def route(self, _query: str) -> tuple[str, str, float]:
            return "concept_explanation", "rule:concept(what is)", 0.65

    monkeypatch.setattr(agents_pkg, "Router", FakeRouter)
    result = runner.invoke(cli.app, ["route", "what is a transformer"])
    assert result.exit_code == 0
    assert "concept_explanation" in result.stdout
    assert "0.65" in result.stdout


# --------------------------------------------------------------------------- #
# eval --dry-run (M4) — genuinely offline, no patching needed.
# --------------------------------------------------------------------------- #
def test_eval_dry_run_writes_reports(tmp_path: Any) -> None:
    out_dir = tmp_path / "eval_out"
    result = runner.invoke(cli.app, ["eval", "--dry-run", "--out", str(out_dir)])
    assert result.exit_code == 0
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
