"""The evaluation runner: turn a golden set into per-question records + aggregates.

For each question the runner does up to three independent measurements:

* **Retrieval** (retrieval subset only) — call the retriever directly for the ranked
  ``doc_id``s and score precision/recall@k at the document level against expected
  sources. This is measured at the retrieval layer, independent of generation, exactly
  as the acceptance criterion ("P/R@k against expected sources") intends.
* **Agent** (every question) — run the agent, timing it, and record the chosen route,
  the refusal flag and the answer text. Routing accuracy and refusal correctness fall
  out of these.
* **Faithfulness** (answered, non-refused questions with citations) — run the LLM judge.

Every per-question step is wrapped so that one failure is logged and recorded on the
record (``error=...``) and the run continues — no silent failure, no aborted sweep."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import structlog

from research_navigator.eval import metrics
from research_navigator.eval.cost import (
    CostModel,
    LatencyStats,
    Stopwatch,
    TokenCounter,
)
from research_navigator.eval.golden import GoldenQuestion, GoldenSet
from research_navigator.eval.judge import CitationJudge, JudgeVerdict
from research_navigator.eval.ports import Agent, CitedSource, Retriever
from research_navigator.eval.settings import EvalSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """Fully serialisable per-question result (JSON-round-trippable)."""

    id: str
    query: str
    expected_route: str
    predicted_route: str | None
    route_correct: bool
    expect_refusal: bool
    refused: bool
    refusal_correct: bool
    is_retrieval: bool
    expected_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    judge: JudgeVerdict | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "expected_route": self.expected_route,
            "predicted_route": self.predicted_route,
            "route_correct": self.route_correct,
            "expect_refusal": self.expect_refusal,
            "refused": self.refused,
            "refusal_correct": self.refusal_correct,
            "is_retrieval": self.is_retrieval,
            "expected_doc_ids": list(self.expected_doc_ids),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "precision_at_k": {str(k): round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): round(v, 4) for k, v in self.recall_at_k.items()},
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "judge": self.judge.to_dict() if self.judge is not None else None,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """Records for one configuration plus the settings snapshot they ran under."""

    config_name: str
    records: tuple[QueryRecord, ...] = field(default_factory=tuple)

    def latency_stats(self, pcts: Sequence[int]) -> LatencyStats:
        samples = [r.latency_ms for r in self.records if r.error is None]
        return LatencyStats.from_samples(samples, pcts)


class EvalRunner:
    """Orchestrates one full pass over a golden set for a single configuration."""

    def __init__(
        self,
        *,
        agent: Agent,
        retriever: Retriever,
        token_counter: TokenCounter,
        cost_model: CostModel,
        settings: EvalSettings,
        judge: CitationJudge | None = None,
        clock: object | None = None,
    ) -> None:
        self._agent = agent
        self._retriever = retriever
        self._tokens = token_counter
        self._cost = cost_model
        self._settings = settings
        self._judge = judge
        self._clock = clock

    @property
    def _now_year(self) -> int:
        return self._settings.now_year if self._settings.now_year is not None else date.today().year

    def _score_retrieval(
        self, q: GoldenQuestion
    ) -> tuple[tuple[str, ...], dict[int, float], dict[int, float], dict[str, str]]:
        hits = self._retriever.retrieve(q.query, top_k=self._settings.max_k)
        retrieved = tuple(h.doc_id for h in hits)
        relevant = frozenset(q.expected_doc_ids)
        precision = {
            k: metrics.precision_at_k(retrieved, relevant, k) for k in self._settings.k_values
        }
        recall = {k: metrics.recall_at_k(retrieved, relevant, k) for k in self._settings.k_values}
        # First-seen chunk text per doc_id, so the judge sees the actual retrieved
        # source rather than an empty string (Citation itself carries no text — it's
        # a display-only rendering of a retrieved chunk, not a data carrier).
        text_by_doc: dict[str, str] = {}
        for h in hits:
            if h.doc_id and h.doc_id not in text_by_doc and h.text:
                text_by_doc[h.doc_id] = h.text
        return retrieved, precision, recall, text_by_doc

    @staticmethod
    def _with_grounding_text(
        citations: Sequence[CitedSource], text_by_doc: dict[str, str]
    ) -> list[CitedSource]:
        """Backfill each citation's ``.text`` from the matching retrieved chunk when the
        citation itself carries none, so the judge has real source text to check
        grounding against instead of an empty string."""
        enriched: list[CitedSource] = []
        for c in citations:
            if c.text or c.doc_id not in text_by_doc:
                enriched.append(c)
            else:
                enriched.append(replace(c, text=text_by_doc[c.doc_id]))  # type: ignore[type-var]
        return enriched

    def run_question(self, q: GoldenQuestion) -> QueryRecord:
        retrieved: tuple[str, ...] = ()
        precision: dict[int, float] = {}
        recall: dict[int, float] = {}
        predicted_route: str | None = None
        refused = False
        answer_text = ""
        judge_verdict: JudgeVerdict | None = None
        input_tokens = 0
        output_tokens = 0
        latency_ms = 0.0
        error: str | None = None

        text_by_doc: dict[str, str] = {}
        try:
            if q.is_retrieval:
                retrieved, precision, recall, text_by_doc = self._score_retrieval(q)

            with Stopwatch(self._clock) as sw:
                outcome = self._agent.run(q.query, now_year=self._now_year)
            latency_ms = sw.elapsed_ms

            predicted_route = outcome.route
            refused = outcome.refused
            answer_text = outcome.answer_text
            citations = self._with_grounding_text(outcome.citations, text_by_doc)

            # Token/cost accounting from what we can observe: the query + cited source
            # text as the input surface, the answer as the output surface. This is an
            # estimate (we don't see the exact generation prompt) and is reported so.
            source_text = " ".join(c.text for c in citations)
            input_tokens = self._tokens.count(q.query) + self._tokens.count(source_text)
            output_tokens = self._tokens.count(answer_text)

            if self._judge is not None and not refused and citations:
                judge_verdict = self._judge.judge(q.query, answer_text, citations)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("eval_question_failed", id=q.id, error=error)

        route_correct = predicted_route == q.expected_route
        refusal_correct = refused == q.expect_refusal
        cost_usd = self._cost.cost(input_tokens, output_tokens)

        return QueryRecord(
            id=q.id,
            query=q.query,
            expected_route=q.expected_route,
            predicted_route=predicted_route,
            route_correct=route_correct,
            expect_refusal=q.expect_refusal,
            refused=refused,
            refusal_correct=refusal_correct,
            is_retrieval=q.is_retrieval,
            expected_doc_ids=q.expected_doc_ids,
            retrieved_doc_ids=retrieved,
            precision_at_k=precision,
            recall_at_k=recall,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            judge=judge_verdict,
            error=error,
        )

    def run(self, golden: GoldenSet, *, config_name: str) -> RunResult:
        log.info("eval_run_start", config=config_name, n=len(golden))
        records = tuple(self.run_question(q) for q in golden)
        n_err = sum(1 for r in records if r.error is not None)
        log.info("eval_run_done", config=config_name, n=len(records), errors=n_err)
        return RunResult(config_name=config_name, records=records)
