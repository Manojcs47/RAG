"""The graph's nodes: the router node plus the six route nodes.

Each node is a plain callable ``AgentState -> StateUpdate`` (a partial-state dict that
LangGraph merges). Nodes are built by factories that close over ``AgentDeps`` so the
graph wiring stays declarative and the nodes stay unit-testable with fakes.

Route behaviours:
- concept_explanation / compare_approaches : straight cited RAG via the M2 Generator.
- recent_developments : date-math + corpus-window tool calls, then recency-biased RAG,
  with a grounded (manifest-derived) preamble.
- paper_deep_dive : resolve the paper via the corpus tool, then cited RAG.
- find_papers : answered DETERMINISTICALLY from the manifest (no LLM) -> no hallucination.
- out_of_scope : a fixed graceful decline (no retrieval, no LLM).
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import structlog

from research_navigator.generate.models import Answer

from .router import Router
from .settings import AgentSettings
from .state import AgentState, Route, StateUpdate, ToolCall
from .tools import CorpusIndex, DocMeta, recency_cutoff

log = structlog.get_logger(__name__)

Node = Callable[[AgentState], StateUpdate]


class Generating(Protocol):
    """Any object exposing the M2 Generator's ``answer`` method (structural typing,
    so the concrete Generator is injected without an import dependency)."""

    def answer(self, query: str, *, now_year: int | None = None) -> Answer: ...


@dataclass(frozen=True)
class AgentDeps:
    """Everything the nodes need, injected once at graph-build time."""

    generator: Generating
    corpus: CorpusIndex
    router: Router
    settings: AgentSettings


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _serialize_answer(ans: Answer) -> dict[str, object]:
    """Frozen dataclass -> JSON-serializable dict (tuples become lists)."""
    return dataclasses.asdict(ans)


def _now_year(state: AgentState, settings: AgentSettings) -> int:
    y = state.get("now_year")
    if y is not None:
        return y
    if settings.recency_reference_year is not None:
        return settings.recency_reference_year
    return datetime.date.today().year


def _rag_update(ans: Answer, tool_calls: list[ToolCall]) -> StateUpdate:
    return {
        "answer": _serialize_answer(ans),
        "answer_text": ans.render(),
        "refused": ans.refused,
        "tool_calls": tool_calls,
    }


def _fmt_paper(d: DocMeta) -> str:
    authors = ", ".join(d.authors[:3]) + (" et al." if len(d.authors) > 3 else "")
    year = f" ({d.year})" if d.year is not None else ""
    url = f" — {d.source_url}" if d.source_url else ""
    who = f" {authors}." if authors else ""
    return f"{d.title}{year}.{who}{url}"


# --------------------------------------------------------------------------- #
# router node
# --------------------------------------------------------------------------- #
def make_router_node(deps: AgentDeps) -> Node:
    def router_node(state: AgentState) -> StateUpdate:
        route, reason, conf = deps.router.route(state["query"])
        return {"route": route, "route_reason": reason, "route_confidence": conf}

    return router_node


# --------------------------------------------------------------------------- #
# RAG-only routes
# --------------------------------------------------------------------------- #
def _make_rag_node(deps: AgentDeps, label: str) -> Node:
    def node(state: AgentState) -> StateUpdate:
        ans = deps.generator.answer(state["query"], now_year=state.get("now_year"))
        log.info("agent.route", route=label, refused=ans.refused, n_cit=len(ans.citations))
        return _rag_update(ans, tool_calls=[])

    return node


def make_concept_node(deps: AgentDeps) -> Node:
    return _make_rag_node(deps, Route.CONCEPT_EXPLANATION.value)


def make_compare_node(deps: AgentDeps) -> Node:
    return _make_rag_node(deps, Route.COMPARE_APPROACHES.value)


# --------------------------------------------------------------------------- #
# recent_developments  (date-math tool + corpus-window tool + recency RAG)
# --------------------------------------------------------------------------- #
def make_recent_node(deps: AgentDeps) -> Node:
    def node(state: AgentState) -> StateUpdate:
        s = deps.settings
        now = _now_year(state, s)
        cutoff = recency_cutoff(now, s.recency_window_years)

        calls: list[ToolCall] = [
            ToolCall(
                name="date_math.recency_cutoff",
                args={"now_year": now, "window_years": s.recency_window_years},
                result={"cutoff_year": cutoff},
            )
        ]
        window = deps.corpus.find(year_gte=cutoff)
        calls.append(
            ToolCall(
                name="corpus.window_count",
                args={"year_gte": cutoff},
                result={"n_docs": len(window), "latest_year": deps.corpus.latest_year()},
            )
        )

        ans = deps.generator.answer(state["query"], now_year=now)
        latest = deps.corpus.latest_year()
        preamble = (
            f"(Recent = {cutoff}\u2013present. The corpus holds {len(window)} document(s) "
            f"in that window; newest is {latest}.)\n\n"
        )
        update = _rag_update(ans, calls)
        if not ans.refused:
            update["answer_text"] = preamble + str(update["answer_text"])
        log.info("agent.route", route="recent_developments", cutoff=cutoff, window=len(window))
        return update

    return node


# --------------------------------------------------------------------------- #
# paper_deep_dive  (corpus resolve tool + cited RAG)
# --------------------------------------------------------------------------- #
def make_deep_dive_node(deps: AgentDeps) -> Node:
    def node(state: AgentState) -> StateUpdate:
        query = state["query"]
        matches = deps.corpus.find(text=query, limit=1)
        resolved = matches[0] if matches else None
        calls: list[ToolCall] = [
            ToolCall(
                name="corpus.resolve_paper",
                args={"text": query},
                result={"doc_id": resolved.doc_id, "title": resolved.title}
                if resolved
                else {"doc_id": None, "title": None},
            )
        ]

        ans = deps.generator.answer(query, now_year=state.get("now_year"))
        update = _rag_update(ans, calls)
        if resolved is not None and not ans.refused:
            header = f"On \u201c{resolved.title}\u201d ({_fmt_paper(resolved)}):\n\n"
            update["answer_text"] = header + str(update["answer_text"])
        log.info(
            "agent.route",
            route="paper_deep_dive",
            resolved=resolved.doc_id if resolved else None,
            refused=ans.refused,
        )
        return update

    return node


# --------------------------------------------------------------------------- #
# find_papers  (deterministic manifest query — NO LLM, NO hallucination)
# --------------------------------------------------------------------------- #
def make_find_papers_node(deps: AgentDeps) -> Node:
    def node(state: AgentState) -> StateUpdate:
        s = deps.settings
        query = state["query"]
        tags = deps.corpus.match_tags(query)
        recent_hit = any(cue in query.lower() for cue in s.recent_cues)
        year_gte = (
            recency_cutoff(_now_year(state, s), s.recency_window_years) if recent_hit else None
        )

        # If a known corpus tag is mentioned, filter by it; otherwise fall back to a
        # title/keyword substring scan. Never call find() with *no* constraint (that
        # would match the whole corpus and drown the answer).
        if tags:
            results = deps.corpus.find(tags=tags, year_gte=year_gte, limit=s.find_papers_limit)
        else:
            results = deps.corpus.find(text=query, year_gte=year_gte, limit=s.find_papers_limit)

        calls: list[ToolCall] = [
            ToolCall(
                name="corpus.find",
                args={"tags": list(tags), "year_gte": year_gte, "limit": s.find_papers_limit},
                result={"n_results": len(results)},
            )
        ]

        if not results:
            text = (
                "No documents in the corpus match that request"
                + (f" (tags: {', '.join(tags)})" if tags else "")
                + "."
            )
            refused = True
        else:
            lines = [f"[{i}] {_fmt_paper(d)}" for i, d in enumerate(results, 1)]
            head = "Matching documents in the corpus"
            if tags:
                head += f" (tags: {', '.join(tags)})"
            if year_gte is not None:
                head += f", {year_gte}\u2013present"
            text = head + ":\n" + "\n".join(lines)
            refused = False

        log.info("agent.route", route="find_papers", n_results=len(results), tags=list(tags))
        return {"answer": None, "answer_text": text, "refused": refused, "tool_calls": calls}

    return node


# --------------------------------------------------------------------------- #
# out_of_scope  (graceful decline; no retrieval, no LLM)
# --------------------------------------------------------------------------- #
def make_out_of_scope_node(deps: AgentDeps) -> Node:
    def node(state: AgentState) -> StateUpdate:
        log.info("agent.route", route="out_of_scope")
        return {
            "answer": None,
            "answer_text": deps.settings.out_of_scope_message,
            "refused": True,
            "tool_calls": [],
        }

    return node
