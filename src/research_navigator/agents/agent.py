"""Thin facade over the compiled graph. The CLI and (future) API talk to ``Agent``,
never to LangGraph directly, so the orchestration layer stays swappable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .nodes import AgentDeps
from .state import AgentState, ToolCall

if TYPE_CHECKING:
    from .state import CompiledAgentGraph


@dataclass(frozen=True)
class AgentResult:
    """Everything a caller (CLI / API / eval harness) needs, fully serializable."""

    query: str
    route: str
    route_reason: str
    route_confidence: float
    answer_text: str
    refused: bool
    tool_calls: tuple[ToolCall, ...]
    answer: dict[str, Any] | None  # serialized generate.Answer when a RAG route ran

    def render(self) -> str:
        head = f"[route: {self.route} · {self.route_reason} · conf={self.route_confidence:.2f}]"
        return f"{head}\n\n{self.answer_text}"


class Agent:
    def __init__(self, graph: CompiledAgentGraph, deps: AgentDeps) -> None:
        self._graph = graph
        self._deps = deps

    def run(self, query: str, *, now_year: int | None = None) -> AgentResult:
        initial: AgentState = {"query": query, "now_year": now_year, "tool_calls": []}
        final: dict[str, Any] = dict(self._graph.invoke(initial))
        return AgentResult(
            query=query,
            route=final.get("route", ""),
            route_reason=final.get("route_reason", ""),
            route_confidence=float(final.get("route_confidence", 0.0)),
            answer_text=final.get("answer_text", ""),
            refused=bool(final.get("refused", False)),
            tool_calls=tuple(final.get("tool_calls", []) or ()),
            answer=final.get("answer"),
        )

    def route_of(self, query: str) -> tuple[str, str, float]:
        """Run only the router (no route node) — handy for the CLI ``route`` command."""
        return self._deps.router.route(query)

    def mermaid(self) -> str:
        return self._graph.get_graph().draw_mermaid()
