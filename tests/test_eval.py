"""Hermetic tests for the M4 evaluation harness.

No network, no Qdrant, no real LLM, no disk (except tmp_path). Every external dependency
is a Fake that satisfies the structural ports in :mod:`research_navigator.eval.ports`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from research_navigator.agents.state import Route
from research_navigator.eval import metrics
from research_navigator.eval.adapters import (
    CallableModel,
    judge_model_from_language_model,
    outcome_from_agent_result,
)
from research_navigator.eval.cost import (
    CostModel,
    HeuristicCounter,
    LatencyStats,
    Stopwatch,
    make_token_counter,
)
from research_navigator.eval.golden import GoldenQuestion, GoldenSet, load_golden_set
from research_navigator.eval.harness import ComponentBundle, EvalConfig, run_eval
from research_navigator.eval.judge import CitationJudge, JudgeVerdict
from research_navigator.eval.ports import AgentOutcome, CitedSource, RetrievedChunk
from research_navigator.eval.report import EvalReport, summarize
from research_navigator.eval.runner import EvalRunner
from research_navigator.eval.settings import EvalSettings


# --------------------------------------------------------------------------- fakes
@dataclass(frozen=True)
class FakeHit:
    doc_id: str
    score: float = 1.0
    section_title: str | None = None
    text: str = ""


@dataclass(frozen=True)
class FakeCite:
    doc_id: str
    title: str = "Some Title"
    section: str | None = None
    text: str = "grounding text"


@dataclass(frozen=True)
class FakeOutcome:
    route: str
    refused: bool
    answer_text: str
    citations: Sequence[CitedSource]


class FakeRetriever:
    """Returns canned chunks per query; unknown queries return nothing."""

    def __init__(self, table: dict[str, list[str]]) -> None:
        self._table = table

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]:
        docs = self._table.get(query, [])
        return [FakeHit(d) for d in docs[:top_k]]


class FakeAgent:
    """Returns a canned outcome per query."""

    def __init__(self, table: dict[str, FakeOutcome]) -> None:
        self._table = table
        self.calls = 0

    def run(self, query: str, *, now_year: int) -> AgentOutcome:
        self.calls += 1
        return self._table[query]


class ExplodingAgent:
    def run(self, query: str, *, now_year: int) -> AgentOutcome:  # pragma: no cover - raises
        raise RuntimeError("boom")


class FakeJudgeModel:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, prompt: str) -> str:
        return self._reply


class ExplodingJudgeModel:
    def complete(self, prompt: str) -> str:  # pragma: no cover - raises
        raise RuntimeError("judge down")


def _settings(**kw: object) -> EvalSettings:
    base: dict[str, object] = {"k_values": (3, 5), "primary_k": 5, "now_year": 2026}
    base.update(kw)
    return EvalSettings(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- metrics
def test_precision_recall_basic() -> None:
    retrieved = ["a", "a", "b", "c"]  # dedupes to a, b, c
    relevant = frozenset({"a", "c"})
    assert metrics.precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)
    assert metrics.recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)
    assert metrics.recall_at_k(retrieved, relevant, 1) == pytest.approx(0.5)


def test_precision_empty_and_no_relevant() -> None:
    assert metrics.precision_at_k([], frozenset({"a"}), 3) == 0.0
    assert metrics.recall_at_k(["a"], frozenset(), 3) == 0.0
    assert metrics.f1(0.0, 0.0) == 0.0
    assert metrics.f1(0.5, 0.5) == pytest.approx(0.5)


def test_percentile_interpolation() -> None:
    vals = [10.0, 20.0, 30.0, 40.0]
    assert metrics.percentile(vals, 50) == pytest.approx(25.0)
    assert metrics.percentile([5.0], 95) == 5.0
    assert metrics.percentile([], 50) == 0.0
    with pytest.raises(ValueError):
        metrics.percentile(vals, 0)


def test_routing_accuracy_and_confusion() -> None:
    pairs = [("a", "a"), ("b", "a"), ("b", "b")]
    assert metrics.accuracy(pairs) == pytest.approx(2 / 3)
    per = metrics.per_label_accuracy(pairs)
    assert per["a"] == pytest.approx(0.5)
    assert per["b"] == pytest.approx(1.0)
    conf = metrics.confusion_counts(pairs)
    assert conf["a"] == {"a": 1, "b": 1}


def test_binary_refusal_rates() -> None:
    # actual refused, expected refuse
    actual = [True, False, True, False]
    expected = [True, False, False, False]
    r = metrics.binary_rates(actual, expected)
    assert r["accuracy"] == pytest.approx(3 / 4)
    assert r["recall"] == pytest.approx(1.0)  # every should-refuse was refused
    assert r["precision"] == pytest.approx(0.5)  # one over-refusal
    with pytest.raises(ValueError):
        metrics.binary_rates([True], [True, False])


# --------------------------------------------------------------------------- settings
def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        EvalSettings(k_values=(3, 5), primary_k=8)  # primary not in k_values
    with pytest.raises(ValueError):
        EvalSettings(k_values=())
    s = _settings()
    assert s.max_k == 5


# --------------------------------------------------------------------------- golden
def test_default_golden_set_loads_and_is_covered() -> None:
    gs = load_golden_set(None)
    assert len(gs) == 40
    gs.assert_route_coverage(minimum=3)
    assert len(gs.retrieval_subset) == 32
    assert len(gs.refusal_subset) == 8
    for r in Route:
        assert gs.route_coverage()[r.value] >= 3


def test_golden_validation_errors() -> None:
    with pytest.raises(ValueError):
        GoldenQuestion(id="x", query="q", expected_route="not_a_route")
    with pytest.raises(ValueError):
        GoldenQuestion(id="x", query="q", expected_route="out_of_scope", expect_refusal=False)
    with pytest.raises(ValueError):
        GoldenQuestion(id="x", query="q", expected_route="concept_explanation", expect_refusal=True)
    with pytest.raises(ValueError):  # duplicate id
        GoldenSet(
            (
                GoldenQuestion("d", "q1", "find_papers", ("a",)),
                GoldenQuestion("d", "q2", "find_papers", ("b",)),
            )
        )


def test_golden_roundtrip() -> None:
    gs = load_golden_set(None)
    rows = gs.to_dict()["questions"]
    again = GoldenSet.from_rows(rows)
    assert len(again) == len(gs)


# --------------------------------------------------------------------------- cost
def test_heuristic_counter_and_cost() -> None:
    c = HeuristicCounter()
    assert c.count("") == 0
    assert c.count("abcd") == 1
    assert c.count("abcde") == 2
    assert not c.is_exact
    cm = CostModel(price_per_1k_input_usd=0.00015, price_per_1k_output_usd=0.0006)
    assert cm.cost(1000, 1000) == pytest.approx(0.00075)


def test_make_token_counter_offline_falls_back() -> None:
    # In the sandbox tiktoken cannot fetch its vocab -> heuristic fallback, no crash.
    counter = make_token_counter("gpt-4o-mini")
    assert counter.count("hello world") > 0


def test_stopwatch_with_fake_clock() -> None:
    ticks = iter([1.0, 1.5])
    with Stopwatch(clock=lambda: next(ticks)) as sw:
        pass
    assert sw.elapsed_ms == pytest.approx(500.0)


def test_latency_stats() -> None:
    ls = LatencyStats.from_samples([10.0, 20.0, 30.0], [50, 95])
    assert ls.n == 3
    assert ls.percentiles_ms[50] == pytest.approx(20.0)
    assert "percentiles_ms" in ls.to_dict()


# --------------------------------------------------------------------------- judge
_GOOD = json.dumps(
    {
        "supported_claims": 3,
        "unsupported_claims": 0,
        "uncited_claims": 0,
        "faithful": True,
        "rationale": "ok",
    }
)
_FENCED = "here:\n```json\n" + _GOOD + "\n```\n"
_UNFAITHFUL = json.dumps(
    {
        "supported_claims": 1,
        "unsupported_claims": 2,
        "uncited_claims": 1,
        "faithful": False,
        "rationale": "bad",
    }
)


def _sources() -> list[FakeCite]:
    return [FakeCite("arxiv-1", text="a source"), FakeCite("arxiv-2", text="another")]


def test_judge_parses_plain_and_fenced() -> None:
    j = CitationJudge(FakeJudgeModel(_GOOD), min_score=0.8)
    v = j.judge("q", "answer [1]", _sources())
    assert v.parse_ok and v.faithful and v.score == pytest.approx(1.0)
    j2 = CitationJudge(FakeJudgeModel(_FENCED), min_score=0.8)
    assert j2.judge("q", "a [1]", _sources()).faithful


def test_judge_unfaithful_and_threshold() -> None:
    v = CitationJudge(FakeJudgeModel(_UNFAITHFUL), min_score=0.8).judge("q", "a [1]", _sources())
    assert v.parse_ok and not v.faithful
    assert v.unsupported_claims == 2 and v.uncited_claims == 1


def test_judge_no_citations_and_unparseable_and_error() -> None:
    j = CitationJudge(FakeJudgeModel(_GOOD), min_score=0.8)
    assert j.judge("q", "answer", []) == JudgeVerdict.no_citations()
    bad = CitationJudge(FakeJudgeModel("not json at all"), min_score=0.8)
    v = bad.judge("q", "a [1]", _sources())
    assert not v.parse_ok and not v.faithful
    err = CitationJudge(ExplodingJudgeModel(), min_score=0.8)
    assert not err.judge("q", "a [1]", _sources()).parse_ok


# --------------------------------------------------------------------------- runner
def _outcome(route: str, *, refused: bool = False, cited: bool = True) -> FakeOutcome:
    cites = [FakeCite("arxiv-2106.09685", text="LoRA text")] if cited else []
    return FakeOutcome(route=route, refused=refused, answer_text="answer [1]", citations=cites)


def test_runner_backfills_empty_citation_text_from_retrieval() -> None:
    """``Citation`` (the real M2 model) carries no source text — it's a display-only
    rendering, not a data carrier. The runner must backfill each citation's grounding
    text from the matching retrieved chunk before judging, or the judge sees an empty
    SOURCES block and can't meaningfully score grounding."""

    class RecordingJudgeModel:
        def __init__(self, reply: str) -> None:
            self._reply = reply
            self.prompts: list[str] = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return self._reply

    class TextRetriever:
        def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]:
            return [FakeHit("arxiv-2106.09685", text="LoRA freezes the base weights.")]

    q = GoldenQuestion("r2", "lora?", "paper_deep_dive", ("arxiv-2106.09685",))
    agent = FakeAgent(
        {
            "lora?": FakeOutcome(
                "paper_deep_dive",
                False,
                "LoRA is efficient [1].",
                [FakeCite("arxiv-2106.09685", text="")],
            )
        }
    )
    judge_model = RecordingJudgeModel(_GOOD)
    runner = EvalRunner(
        agent=agent,
        retriever=TextRetriever(),
        token_counter=HeuristicCounter(),
        cost_model=CostModel(0.00015, 0.0006),
        settings=_settings(),
        judge=CitationJudge(judge_model, min_score=0.8),
        clock=iter([0.0, 0.01]).__next__,
    )
    rec = runner.run_question(q)
    assert rec.judge is not None
    assert judge_model.prompts, "judge was never called"
    assert "LoRA freezes the base weights." in judge_model.prompts[0]


def test_runner_retrieval_routing_refusal_and_judge() -> None:
    q_ret = GoldenQuestion("r1", "lora?", "paper_deep_dive", ("arxiv-2106.09685",))
    q_oos = GoldenQuestion("o1", "biryani?", "out_of_scope", (), expect_refusal=True)
    retriever = FakeRetriever({"lora?": ["arxiv-2106.09685", "arxiv-x"]})
    agent = FakeAgent(
        {
            "lora?": _outcome("paper_deep_dive"),
            "biryani?": _outcome("out_of_scope", refused=True, cited=False),
        }
    )
    runner = EvalRunner(
        agent=agent,
        retriever=retriever,
        token_counter=HeuristicCounter(),
        cost_model=CostModel(0.00015, 0.0006),
        settings=_settings(),
        judge=CitationJudge(FakeJudgeModel(_GOOD), min_score=0.8),
        clock=iter([0.0, 0.01, 0.0, 0.005]).__next__,
    )
    rec_ret = runner.run_question(q_ret)
    assert rec_ret.route_correct and not rec_ret.refused
    assert rec_ret.precision_at_k[3] == pytest.approx(0.5)
    assert rec_ret.recall_at_k[3] == pytest.approx(1.0)
    assert rec_ret.judge is not None and rec_ret.judge.faithful
    assert rec_ret.latency_ms == pytest.approx(10.0)

    rec_oos = runner.run_question(q_oos)
    assert rec_oos.refusal_correct and rec_oos.refused
    assert rec_oos.judge is None  # refused -> not judged


def test_runner_records_error_and_continues() -> None:
    q = GoldenQuestion("e1", "x", "find_papers", ("a",))
    runner = EvalRunner(
        agent=ExplodingAgent(),
        retriever=FakeRetriever({"x": ["a"]}),
        token_counter=HeuristicCounter(),
        cost_model=CostModel(0.00015, 0.0006),
        settings=_settings(),
    )
    rec = runner.run_question(q)
    assert rec.error is not None and "boom" in rec.error
    assert rec.predicted_route is None and not rec.route_correct


def test_query_record_is_json_serializable() -> None:
    q = GoldenQuestion("r1", "lora?", "paper_deep_dive", ("arxiv-2106.09685",))
    runner = EvalRunner(
        agent=FakeAgent({"lora?": _outcome("paper_deep_dive")}),
        retriever=FakeRetriever({"lora?": ["arxiv-2106.09685"]}),
        token_counter=HeuristicCounter(),
        cost_model=CostModel(0.00015, 0.0006),
        settings=_settings(),
    )
    rec = runner.run_question(q)
    json.dumps(rec.to_dict())  # must not raise


# --------------------------------------------------------------------------- report
def test_summarize_and_markdown(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gs = GoldenSet(
        (
            GoldenQuestion("a", "lora?", "paper_deep_dive", ("arxiv-2106.09685",)),
            GoldenQuestion("b", "biryani?", "out_of_scope", (), expect_refusal=True),
            GoldenQuestion("c", "read rag?", "find_papers", ("arxiv-2005.11401",)),
        )
    )
    retriever = FakeRetriever(
        {
            "lora?": ["arxiv-2106.09685"],
            "read rag?": ["arxiv-2005.11401", "z"],
            "biryani?": [],
        }
    )
    agent = FakeAgent(
        {
            "lora?": _outcome("paper_deep_dive"),
            "read rag?": _outcome("find_papers"),
            "biryani?": _outcome("out_of_scope", refused=True, cited=False),
        }
    )
    s = _settings()
    runner = EvalRunner(
        agent=agent,
        retriever=retriever,
        token_counter=HeuristicCounter(),
        cost_model=CostModel(0.00015, 0.0006),
        settings=s,
        judge=CitationJudge(FakeJudgeModel(_GOOD), min_score=0.8),
    )
    run = runner.run(gs, config_name="hybrid")
    cr = summarize(run, s)
    assert cr.routing_accuracy == pytest.approx(1.0)
    assert cr.refusal["accuracy"] == pytest.approx(1.0)
    assert cr.judge_faithfulness_rate == pytest.approx(1.0)

    report = EvalReport.build([run], s, golden_set_size=len(gs), token_counts_exact=False)
    md = report.to_markdown()
    assert "Evaluation Report" in md and "hybrid" in md and "Limitations" in md
    json.loads(report.to_json())  # round-trips
    jp, mp = report.write(s.model_copy(update={"report_dir": tmp_path}))
    assert jp.exists() and mp.exists()


# --------------------------------------------------------------------------- harness
def test_run_eval_two_configs() -> None:
    gs = load_golden_set(None)
    outcomes = {
        q.query: _outcome(q.expected_route, refused=q.expect_refusal, cited=not q.expect_refusal)
        for q in gs
    }
    table = {q.query: list(q.expected_doc_ids) for q in gs}

    def builder(cfg: EvalConfig) -> ComponentBundle:
        return ComponentBundle(
            agent=FakeAgent(outcomes),
            retriever=FakeRetriever(table),
            judge_model=FakeJudgeModel(_GOOD),
        )

    report = run_eval(builder=builder, golden=gs, settings=_settings(k_values=(3, 5, 8)))
    assert {c.config_name for c in report.configs} == {"hybrid", "dense_only"}
    for c in report.configs:
        assert c.routing_accuracy == pytest.approx(1.0)
        assert c.refusal["accuracy"] == pytest.approx(1.0)
        # A perfect retriever recovers every expected doc once k covers the
        # largest expected-doc count (one golden question expects 6 docs), so
        # recall is exact at k=8 but legitimately below 1.0 at k=5.
        assert c.retrieval_recall_at_k[8] == pytest.approx(1.0)
        assert c.retrieval_recall_at_k[5] >= 0.98


def test_run_eval_rejects_thin_coverage() -> None:
    thin = GoldenSet((GoldenQuestion("a", "q", "find_papers", ("x",)),))

    def builder(cfg: EvalConfig) -> ComponentBundle:  # pragma: no cover - not reached
        return ComponentBundle(FakeAgent({}), FakeRetriever({}))

    with pytest.raises(ValueError):
        run_eval(builder=builder, golden=thin, settings=_settings())


# --------------------------------------------------------------------------- adapters
def test_outcome_adapter_maps_duck_typed_result() -> None:
    @dataclass
    class RawCite:
        doc_id: str
        title: str
        section: str | None
        text: str

    @dataclass
    class RawAnswer:
        citations: list[RawCite]

    @dataclass
    class RawResult:
        route: str
        refused: bool
        answer_text: str
        answer: RawAnswer

    raw = RawResult(
        "concept_explanation",
        False,
        "hello [1]",
        RawAnswer([RawCite("arxiv-1", "T", "Intro", "body")]),
    )
    out = outcome_from_agent_result(raw)
    assert out.route == "concept_explanation"
    assert out.citations[0].doc_id == "arxiv-1"
    assert out.citations[0].text == "body"


def test_outcome_adapter_maps_real_serialized_answer_dict() -> None:
    """Regression test: ``AgentResult.answer`` is actually ``dict[str, Any] | None``
    (``dataclasses.asdict``'d by ``agents/nodes.py`` for JSON-serializable state), not
    an attribute-bearing object. Using ``getattr`` on a dict never raises but always
    returns the default, so this shape previously made citations silently read as empty
    on every real run — the M4 citation-faithfulness judge never evaluated a real
    answer as a result. This is the exact shape a real ``AgentResult`` produces."""

    @dataclass
    class RawResult:
        route: str
        refused: bool
        answer_text: str
        answer: dict[str, object] | None

    raw = RawResult(
        route="concept_explanation",
        refused=False,
        answer_text="RAG improves grounding [1].",
        answer={
            "query": "what is rag",
            "text": "RAG improves grounding [1].",
            "citations": [
                {
                    "index": 1,
                    "doc_id": "arxiv-2005.11401",
                    "title": "Retrieval-Augmented Generation",
                    "authors": "Lewis et al.",
                    "year": 2020,
                    "source": "arXiv:2005.11401",
                    "section": "Abstract",
                    "url": "https://arxiv.org/abs/2005.11401",
                }
            ],
            "refused": False,
            "reason": None,
            "intent": "concept",
            "confidence": 0.9,
            "uncited_sentences": [],
        },
    )
    out = outcome_from_agent_result(raw)
    assert len(out.citations) == 1
    assert out.citations[0].doc_id == "arxiv-2005.11401"
    assert out.citations[0].title == "Retrieval-Augmented Generation"
    assert out.citations[0].section == "Abstract"


def test_judge_model_adapter_variants() -> None:
    class HasComplete:
        def complete(self, prompt: str) -> str:
            return "c"

    class HasGenerate:
        def generate(self, prompt: str) -> str:
            return "g"

    assert judge_model_from_language_model(HasComplete()).complete("x") == "c"
    assert judge_model_from_language_model(HasGenerate()).complete("x") == "g"
    assert CallableModel(lambda p: p.upper()).complete("hi") == "HI"
    with pytest.raises(TypeError):
        judge_model_from_language_model(object())


def test_judge_model_adapter_wraps_real_language_model_shape() -> None:
    """Regression test: the real ``generate.llm.LanguageModel.complete`` takes
    ``list[ChatMessage]``, not a bare string. Passing the raw prompt straight through
    previously made every live judge call fail with
    ``'str' object has no attribute 'role'`` (iterating a string yields characters, and
    ``OpenAIChatModel.complete`` reads ``.role``/``.content`` off each element)."""
    from research_navigator.generate.llm import ChatMessage

    class RealShapedLanguageModel:
        def __init__(self) -> None:
            self.received: list[ChatMessage] | None = None

        def complete(self, messages: list[ChatMessage]) -> str:
            self.received = messages
            return "judged"

    model = RealShapedLanguageModel()
    adapted = judge_model_from_language_model(model)
    assert adapted.complete("judge this") == "judged"
    assert model.received == [ChatMessage(role="user", content="judge this")]


# --------------------------------------------------------------------------- dry-run builder
def test_dry_run_builder_offline(tmp_path: object) -> None:
    """The packaged offline builder drives the whole harness with no external services."""
    from research_navigator.eval import EvalSettings, load_golden_set, run_eval
    from research_navigator.eval.factory import build_dry_run_builder

    golden = load_golden_set(None)
    settings = EvalSettings(now_year=2026, report_dir=tmp_path, judge_enabled=False)
    report = run_eval(builder=build_dry_run_builder(golden), golden=golden, settings=settings)

    # Fakes return expected docs and expected routes, so retrieval@8 and routing are perfect.
    for c in report.configs:
        assert c.routing_accuracy == pytest.approx(1.0)
        assert c.refusal["accuracy"] == pytest.approx(1.0)
        assert c.judge_n == 0  # judge disabled in the dry run

    md = report.to_markdown()
    assert "n/a" in md  # faithfulness column renders n/a, not 0.00, when unjudged
    assert "not evaluated" in md
    paths = report.write(settings)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
