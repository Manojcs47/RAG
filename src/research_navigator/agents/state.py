"""M3 agent state + the official route vocabulary.

The state is a plain ``TypedDict`` of JSON-serializable primitives (``total=False``
so nodes may return *partial* updates that LangGraph merges in). Keeping the state
JSON-friendly is deliberate: it is what makes the graph checkpointable/serializable
(rubric requirement) and is exactly what a future streaming UI (bonus B1) will read.

``Route`` is the single source of truth for the six official routes; both the router
and the graph wiring derive their node names from it, so there is no place for a
route name to be typo'd into existence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypedDict


class Route(StrEnum):
    """The six official M3 routes. Values double as LangGraph node names."""

    CONCEPT_EXPLANATION = "concept_explanation"
    PAPER_DEEP_DIVE = "paper_deep_dive"
    COMPARE_APPROACHES = "compare_approaches"
    RECENT_DEVELOPMENTS = "recent_developments"
    FIND_PAPERS = "find_papers"
    OUT_OF_SCOPE = "out_of_scope"


class ToolCall(TypedDict):
    """A serializable record of one tool invocation (for the visible trace)."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class AgentState(TypedDict, total=False):
    """Graph state. ``total=False`` -> every key is optional, so a node may return
    just the slice it produced and LangGraph merges it into the running state."""

    # --- inputs (set at invoke) ---------------------------------------------
    query: str
    now_year: int | None

    # --- routing (set by the router node) -----------------------------------
    route: str
    route_reason: str
    route_confidence: float

    # --- tool trace (set/extended by route nodes) ---------------------------
    tool_calls: list[ToolCall]

    # --- output (set by the selected route node) ----------------------------
    answer: dict[str, Any] | None  # serialized generate.Answer, when a RAG route ran
    answer_text: str  # final human-readable text (always set)
    refused: bool


# Nodes return a *partial* state update. Because AgentState is total=False, a dict
# carrying any subset of its keys is a valid AgentState, so we alias the update type
# to AgentState itself — this also unifies cleanly with LangGraph's add_node overloads
# under mypy --strict.
StateUpdate = AgentState


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    # LangGraph's CompiledStateGraph is generic over (StateT, ContextT, InputT,
    # OutputT). We pin the full 4-arg form ONCE here so that if a future LangGraph
    # release changes the parameter list, only this line needs touching — every
    # module that returns a compiled graph refers to this alias instead.
    type CompiledAgentGraph = CompiledStateGraph[AgentState, Any, AgentState, AgentState]
