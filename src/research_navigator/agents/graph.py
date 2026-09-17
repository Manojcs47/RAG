"""LangGraph wiring for the M3 agent.

Topology (verified against the LangGraph Graph API):

    START -> router --(conditional on state["route"])--> one of the 6 route nodes -> END

The router node writes ``state["route"]``; ``_route_selector`` reads it and the
conditional edge maps each route value to the same-named node. Node names ARE the
``Route`` enum values, so the mapping can't drift out of sync with the router.

``compile()`` validates the whole graph at build time (every node reachable, every
edge target real, state schema consistent) — a mis-wired route fails here, not at
run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes import (
    AgentDeps,
    make_compare_node,
    make_concept_node,
    make_deep_dive_node,
    make_find_papers_node,
    make_out_of_scope_node,
    make_recent_node,
    make_router_node,
)
from .state import AgentState, Route

if TYPE_CHECKING:
    from .state import CompiledAgentGraph

ROUTER_NODE = "router"


def _route_selector(state: AgentState) -> str:
    """Conditional-edge path function: hand control to the chosen route node."""
    return state["route"]


def build_agent_graph(deps: AgentDeps) -> CompiledAgentGraph:
    """Assemble and compile the agent state machine."""
    builder = StateGraph(AgentState)

    builder.add_node(ROUTER_NODE, make_router_node(deps))
    builder.add_node(Route.CONCEPT_EXPLANATION.value, make_concept_node(deps))
    builder.add_node(Route.PAPER_DEEP_DIVE.value, make_deep_dive_node(deps))
    builder.add_node(Route.COMPARE_APPROACHES.value, make_compare_node(deps))
    builder.add_node(Route.RECENT_DEVELOPMENTS.value, make_recent_node(deps))
    builder.add_node(Route.FIND_PAPERS.value, make_find_papers_node(deps))
    builder.add_node(Route.OUT_OF_SCOPE.value, make_out_of_scope_node(deps))

    builder.add_edge(START, ROUTER_NODE)
    builder.add_conditional_edges(
        ROUTER_NODE,
        _route_selector,
        {r.value: r.value for r in Route},
    )
    for r in Route:
        builder.add_edge(r.value, END)

    return builder.compile()


def graph_mermaid(deps: AgentDeps) -> str:
    """Return the compiled graph as Mermaid source (offline; no network).

    Use ``build_agent_graph(deps).get_graph().draw_mermaid_png()`` for a PNG, but note
    that path calls the mermaid.ink API (network) unless a local renderer is set.
    """
    return build_agent_graph(deps).get_graph().draw_mermaid()
