"""Wire the real M2/M3 stack into a :class:`ComponentBuilder` for the CLI.

Concrete imports (embedder, Qdrant store, retriever, generator, agent) are done
**lazily inside the builder** so importing this module never drags in the live stack;
the rest of the ``eval`` package stays offline-friendly. This module is intentionally
the seam that needs ``OPENAI_API_KEY`` + a running Qdrant + FastEmbed egress at run time.

It also provides :func:`build_dry_run_builder`, a pure, offline builder over in-repo
fakes, so ``rn eval --dry-run`` / ``make eval-dry-run`` and CI exercise the whole harness
with no external services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from research_navigator.eval.adapters import (
    AgentAdapter,
    RetrieverAdapter,
    judge_model_from_language_model,
)
from research_navigator.eval.golden import GoldenSet
from research_navigator.eval.harness import (
    ComponentBuilder,
    ComponentBundle,
    EvalConfig,
)

if TYPE_CHECKING:  # avoid importing the concrete Settings at module load
    from research_navigator.config import Settings

log = structlog.get_logger(__name__)


def _retrieval_overrides(overrides: dict[str, object], *, top_k: int) -> dict[str, object]:
    """Translate a config's overrides into a ``settings.retrieve.model_copy`` update.

    Recognised keys: ``retrieval_mode`` ('hybrid' | 'dense_only'). We always widen
    ``top_k`` to the harness's ``max_k`` so precision/recall@k can be scored at every k.
    Unknown modes are logged and ignored — no silent failure.
    """
    update: dict[str, object] = {"top_k": top_k}
    mode = overrides.get("retrieval_mode")
    if mode == "dense_only":
        update["dense_only"] = True
    elif mode == "hybrid":
        update["dense_only"] = False
    elif mode is not None:
        log.warning("unknown_retrieval_mode", mode=mode)
    return update


def build_real_builder(settings: Settings) -> ComponentBuilder:
    """Produce a :class:`ComponentBuilder` closed over the app settings.

    The returned callable constructs a fresh retriever+generator+agent per config so a
    config's retrieval overrides actually take effect. All concrete imports are lazy and
    live only inside ``_build``; the wiring mirrors the CLI's ``ask`` command.
    """

    def _build(cfg: EvalConfig) -> ComponentBundle:
        # Lazy imports of the concrete stack (kept out of module import time).
        from research_navigator.agents import build_agent
        from research_navigator.generate import build_generator, build_llm
        from research_navigator.ingest.embedder import build_embedder
        from research_navigator.ingest.factory import build_client, build_store
        from research_navigator.retrieve import build_retriever

        retrieve_settings = settings.retrieve.model_copy(
            update=_retrieval_overrides(cfg.overrides, top_k=settings.eval.max_k)
        )

        embedder = build_embedder(settings.embedding)
        client = build_client(settings.qdrant)
        store = build_store(client, settings, embedder.dense_dim)
        llm = build_llm(settings.llm)
        retriever = build_retriever(
            embedder=embedder,
            searcher=store,
            manifest_path=settings.paths.manifest,
            settings=retrieve_settings,
        )
        generator = build_generator(retriever=retriever, llm=llm, settings=settings.generate)
        agent = build_agent(
            generator=generator,
            manifest_path=settings.paths.manifest,
            llm=llm if settings.agents.use_llm_router else None,
            settings=settings.agents,
        )
        judge_model = judge_model_from_language_model(llm) if settings.eval.judge_enabled else None
        return ComponentBundle(
            agent=AgentAdapter(agent),
            retriever=RetrieverAdapter(retriever),
            judge_model=judge_model,
        )

    return _build


# --------------------------------------------------------------------------- #
# Offline dry-run builder (pure fakes; no external services, no network).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _FakeHit:
    doc_id: str
    score: float
    section_title: str | None = "Abstract"
    text: str = "stub source text"


@dataclass(frozen=True, slots=True)
class _FakeCite:
    doc_id: str
    title: str = "stub"
    section: str | None = "Abstract"
    text: str = "stub source text"


@dataclass(frozen=True, slots=True)
class _FakeOutcome:
    route: str
    refused: bool
    answer_text: str
    citations: list[_FakeCite] = field(default_factory=list)


class _FakeRetriever:
    def __init__(self, docs: dict[str, list[str]]) -> None:
        self._docs = docs

    def retrieve(self, query: str, *, top_k: int) -> list[_FakeHit]:
        ranked = self._docs.get(query, [])
        return [_FakeHit(d, 1.0 - i * 0.01) for i, d in enumerate(ranked[:top_k])]


class _FakeAgent:
    def __init__(self, outcomes: dict[str, _FakeOutcome]) -> None:
        self._outcomes = outcomes

    def run(self, query: str, *, now_year: int) -> _FakeOutcome:
        return self._outcomes[query]


def build_dry_run_builder(golden: GoldenSet) -> ComponentBuilder:
    """A fully offline :class:`ComponentBuilder` over fakes derived from the golden set.

    Every answerable question returns its expected docs (so retrieval/routing are perfect
    by construction) and every refusal question refuses; there is no judge. This exercises
    the whole harness — runner, metrics, report writer — with zero external services.
    """
    docs = {q.query: list(q.expected_doc_ids) for q in golden}
    outcomes = {
        q.query: _FakeOutcome(
            route=q.expected_route,
            refused=q.expect_refusal,
            answer_text="" if q.expect_refusal else "Stub grounded answer [1].",
            citations=[] if q.expect_refusal else [_FakeCite(d) for d in q.expected_doc_ids],
        )
        for q in golden
    }

    def _build(cfg: EvalConfig) -> ComponentBundle:
        return ComponentBundle(
            agent=_FakeAgent(outcomes),
            retriever=_FakeRetriever(docs),
            judge_model=None,
        )

    return _build
