"""M3 agent: a LangGraph state machine that routes each query to one of six routes,
reusing the M2 Retriever + Generator inside the route nodes, with >=1 tool call and a
serializable, checkpointable state."""

from __future__ import annotations

from .agent import Agent, AgentResult
from .factory import build_agent
from .graph import build_agent_graph, graph_mermaid
from .nodes import AgentDeps, Generating
from .router import Router, classify_by_rules
from .settings import AgentSettings
from .state import AgentState, Route, StateUpdate, ToolCall
from .tools import CorpusIndex, DocMeta, recency_cutoff

__all__ = [
    "Agent",
    "AgentDeps",
    "AgentResult",
    "AgentSettings",
    "AgentState",
    "CorpusIndex",
    "DocMeta",
    "Generating",
    "Route",
    "Router",
    "StateUpdate",
    "ToolCall",
    "build_agent",
    "build_agent_graph",
    "classify_by_rules",
    "graph_mermaid",
    "recency_cutoff",
]
