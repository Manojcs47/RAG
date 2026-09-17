"""M3 agent tests — fully hermetic (no network, no qdrant, no real LLM, no disk).

Fakes mirror the S5/S6 pattern: a FakeGenerator implements the ``Generating`` port,
a FakeLLM implements ``LanguageModel`` for the router fallback, and the corpus is a
``CorpusIndex.from_rows`` over inline manifest dicts. Each of the six routes is
exercised by >= 3 queries, per the M3 acceptance criteria.
"""

from __future__ import annotations

import json

import pytest

from research_navigator.agents import (
    AgentSettings,
    CorpusIndex,
    Route,
    Router,
    classify_by_rules,
    recency_cutoff,
)
from research_navigator.agents.agent import Agent
from research_navigator.agents.graph import build_agent_graph, graph_mermaid
from research_navigator.agents.nodes import AgentDeps
from research_navigator.generate.llm import ChatMessage
from research_navigator.generate.models import Answer, Citation

NOW = 2026


# ------------------------------ fakes --------------------------------------- #
class FakeGenerator:
    """Implements the Generating port. Returns a cited answer, or refuses."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.calls: list[tuple[str, int | None]] = []

    def answer(self, query: str, *, now_year: int | None = None) -> Answer:
        self.calls.append((query, now_year))
        if self.refuse:
            return Answer(
                query=query,
                text="Not enough context.",
                citations=(),
                refused=True,
                reason="low_confidence",
                intent="concept",
                confidence=0.1,
            )
        cit = Citation(
            index=1,
            doc_id="d1",
            title="QLoRA",
            authors="Dettmers et al.",
            year=2023,
            source="arXiv:2305.14314",
            section="Method",
            url="https://arxiv.org/abs/2305.14314",
        )
        return Answer(
            query=query,
            text="QLoRA fine-tunes quantized models [1].",
            citations=(cit,),
            refused=False,
            reason=None,
            intent="concept",
            confidence=0.82,
        )


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage]) -> str:
        self.calls.append(messages)
        return self.reply


class ExplodingLLM:
    """Fails if reached — proves the rules path never calls the LLM."""

    def complete(self, messages: list[ChatMessage]) -> str:  # pragma: no cover
        raise AssertionError("router used the LLM when a rule should have fired")


MANIFEST_ROWS = [
    {
        "doc_id": "d1",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": ["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
        "year": 2023,
        "content_type": "arxiv_paper",
        "tags": ["quantization", "LoRA"],
        "is_foundational": False,
        "source_url": "https://arxiv.org/abs/2305.14314",
    },
    {
        "doc_id": "d2",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "content_type": "arxiv_paper",
        "tags": ["attention", "transformers"],
        "is_foundational": True,
        "source_url": "https://arxiv.org/abs/1706.03762",
    },
    {
        "doc_id": "d3",
        "title": "Direct Preference Optimization",
        "authors": ["Rafael Rafailov"],
        "year": 2023,
        "content_type": "arxiv_paper",
        "tags": ["DPO", "RLHF"],
        "is_foundational": False,
        "source_url": "https://arxiv.org/abs/2305.18290",
    },
    {
        "doc_id": "d4",
        "title": "A Survey of Long-Context Methods",
        "authors": ["Lilian Weng"],
        "year": 2025,
        "content_type": "survey_blog",
        "tags": ["long_context", "attention"],
        "is_foundational": False,
        "source_url": "https://lilianweng.github.io/",
    },
    {
        "doc_id": "d5",
        "title": "RAG for Beginners",
        "authors": [],
        "year": 2024,
        "content_type": "course_chapter",
        "tags": ["RAG"],
        "is_foundational": False,
        "source_url": "https://hf.co/learn",
    },
]


def _agent(
    *, llm_reply: str = "out_of_scope", refuse: bool = False, settings: AgentSettings | None = None
) -> tuple[Agent, AgentDeps, FakeGenerator]:
    settings = settings or AgentSettings(recency_reference_year=NOW)
    gen = FakeGenerator(refuse=refuse)
    corpus = CorpusIndex.from_rows(MANIFEST_ROWS)
    router = Router(settings=settings, llm=FakeLLM(llm_reply))
    deps = AgentDeps(generator=gen, corpus=corpus, router=router, settings=settings)
    return Agent(build_agent_graph(deps), deps), deps, gen


SETTINGS = AgentSettings(recency_reference_year=NOW)


# --------------------------- router: rules ---------------------------------- #
# Each route is covered by >= 3 queries that must classify by DETERMINISTIC RULES
# (ExplodingLLM guarantees the LLM is not consulted).
CONCEPT_Q = ["what is LoRA?", "explain how attention works", "define quantization"]
COMPARE_Q = ["compare DPO and RLHF", "RLHF versus DPO", "difference between LoRA and adapters"]
RECENT_Q = ["recent developments in RAG", "latest research on reasoning", "what's new in agents"]
DEEP_Q = [
    "walk me through the QLoRA paper",
    "summarize the paper by Vaswani",
    "explain the attention paper",
]
FIND_Q = [
    "find papers on quantization",
    "list papers about RLHF",
    "which papers cover long context",
]


@pytest.mark.parametrize(
    ("queries", "expected"),
    [
        (CONCEPT_Q, Route.CONCEPT_EXPLANATION),
        (COMPARE_Q, Route.COMPARE_APPROACHES),
        (RECENT_Q, Route.RECENT_DEVELOPMENTS),
        (DEEP_Q, Route.PAPER_DEEP_DIVE),
        (FIND_Q, Route.FIND_PAPERS),
    ],
)
def test_router_rules_classify_each_route(queries: list[str], expected: Route) -> None:
    router = Router(settings=SETTINGS, llm=ExplodingLLM())
    for q in queries:
        route, reason, conf = router.route(q)
        assert route == expected.value, f"{q!r} -> {route} (reason={reason})"
        assert conf >= SETTINGS.router_min_rule_confidence


def test_classify_by_rules_returns_reason() -> None:
    route, reason, _conf = classify_by_rules("compare A and B", SETTINGS)
    assert route == Route.COMPARE_APPROACHES.value
    assert reason.startswith("rule:compare")


def test_find_papers_beats_recent_when_both_cue() -> None:
    # "find recent papers ..." must route to find_papers (which applies recency itself).
    route, _reason, _c = classify_by_rules("find recent papers on RAG", SETTINGS)
    assert route == Route.FIND_PAPERS.value


# --------------------------- router: LLM fallback --------------------------- #
OOS_Q = ["recommend a good pizza place", "who won the 2018 world cup", "book me a flight to Tokyo"]


def test_out_of_scope_via_llm_fallback() -> None:
    router = Router(settings=SETTINGS, llm=FakeLLM("out_of_scope"))
    for q in OOS_Q:  # cue-free -> no rule fires -> LLM decides
        route, reason, _c = router.route(q)
        assert route == Route.OUT_OF_SCOPE.value
        assert reason == "llm"


def test_llm_disabled_defaults_to_concept() -> None:
    s = AgentSettings(recency_reference_year=NOW, use_llm_router=False)
    router = Router(settings=s, llm=None)
    route, reason, _c = router.route("book me a flight to Tokyo")
    assert route == Route.CONCEPT_EXPLANATION.value
    assert reason.startswith("fallback")


def test_unparseable_llm_reply_falls_back() -> None:
    router = Router(settings=SETTINGS, llm=FakeLLM("I think maybe it's about cooking?"))
    route, reason, _c = router.route("book me a flight to Tokyo")
    assert route == Route.CONCEPT_EXPLANATION.value  # graceful, no crash
    assert reason.startswith("fallback")


# --------------------------- end-to-end routes ------------------------------ #
def test_all_six_routes_invoke() -> None:
    invoke = {
        Route.CONCEPT_EXPLANATION: CONCEPT_Q,
        Route.COMPARE_APPROACHES: COMPARE_Q,
        Route.RECENT_DEVELOPMENTS: RECENT_Q,
        Route.PAPER_DEEP_DIVE: DEEP_Q,
        Route.FIND_PAPERS: FIND_Q,
        Route.OUT_OF_SCOPE: OOS_Q,
    }
    for route, queries in invoke.items():
        agent, _deps, _gen = _agent(llm_reply="out_of_scope")
        for q in queries:
            res = agent.run(q, now_year=NOW)
            assert res.route == route.value, (q, res.route)
            assert isinstance(res.answer_text, str) and res.answer_text


def test_concept_route_produces_cited_answer() -> None:
    agent, _deps, gen = _agent()
    res = agent.run("what is LoRA?", now_year=NOW)
    assert res.route == Route.CONCEPT_EXPLANATION.value
    assert res.refused is False
    assert res.answer is not None and res.answer["citations"]
    assert gen.calls  # the generator was invoked


def test_refusal_propagates_from_generator() -> None:
    agent, _deps, _gen = _agent(refuse=True)
    res = agent.run("what is some obscure off-corpus thing", now_year=NOW)
    assert res.refused is True


# --------------------------------- tools ------------------------------------ #
def test_recency_cutoff_math() -> None:
    assert recency_cutoff(2026, 2) == 2025
    assert recency_cutoff(2026, 1) == 2026
    assert recency_cutoff(2026, 3) == 2024


def test_recent_route_records_tool_calls_and_preamble() -> None:
    agent, _deps, _gen = _agent()
    res = agent.run("recent developments in RAG", now_year=NOW)
    names = [t["name"] for t in res.tool_calls]
    assert "date_math.recency_cutoff" in names
    assert "corpus.window_count" in names
    assert "2025" in res.answer_text  # cutoff = 2026 - (2-1)


def test_find_papers_is_deterministic_no_llm() -> None:
    # find_papers answers from the manifest; the generator must not be called.
    agent, _deps, gen = _agent()
    res = agent.run("find papers on quantization", now_year=NOW)
    assert res.route == Route.FIND_PAPERS.value
    assert "QLoRA" in res.answer_text
    assert gen.calls == []  # no generation, no hallucination
    assert any(t["name"] == "corpus.find" for t in res.tool_calls)


def test_find_papers_empty_refuses_gracefully() -> None:
    agent, _deps, _gen = _agent()
    res = agent.run("find papers on underwater basket weaving", now_year=NOW)
    assert res.route == Route.FIND_PAPERS.value
    assert res.refused is True
    assert "No documents" in res.answer_text


def test_paper_deep_dive_resolves_and_generates() -> None:
    agent, _deps, gen = _agent()
    res = agent.run("walk me through the QLoRA paper", now_year=NOW)
    assert res.route == Route.PAPER_DEEP_DIVE.value
    assert any(t["name"] == "corpus.resolve_paper" for t in res.tool_calls)
    assert gen.calls  # deep dive still generates the substantive answer


def test_out_of_scope_skips_generator() -> None:
    agent, _deps, gen = _agent(llm_reply="out_of_scope")
    res = agent.run("who won the 2018 world cup", now_year=NOW)
    assert res.route == Route.OUT_OF_SCOPE.value
    assert res.refused is True
    assert gen.calls == []  # no retrieval, no generation


def test_corpus_index_stats_and_find() -> None:
    idx = CorpusIndex.from_rows(MANIFEST_ROWS)
    stats = idx.stats()
    assert stats["total"] == 5
    assert stats["latest_year"] == 2025
    assert idx.find(is_foundational=True)[0].doc_id == "d2"
    assert {d.doc_id for d in idx.find(year_gte=2024)} == {"d4", "d5"}
    assert "quantization" in idx.all_tags


# ------------------------- state / graph properties ------------------------- #
def test_state_is_json_serializable() -> None:
    agent, _deps, _gen = _agent()
    final = dict(
        agent._graph.invoke({"query": "recent work on RAG", "now_year": NOW, "tool_calls": []})
    )
    json.dumps(final)  # must not raise -> checkpointable/serializable


def test_graph_compiles_and_mermaid_renders() -> None:
    _ag, deps, _gen = _agent()
    mmd = graph_mermaid(deps)
    assert "router" in mmd
    for r in Route:
        assert r.value in mmd


def test_agent_result_render_shows_route() -> None:
    agent, _deps, _gen = _agent()
    res = agent.run("what is LoRA?", now_year=NOW)
    rendered = res.render()
    assert "route: concept_explanation" in rendered
