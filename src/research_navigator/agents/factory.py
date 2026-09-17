"""Assemble an ``Agent`` from its parts. Mirrors ``generate.build_generator`` and
``retrieve.build_retriever`` — the CLI supplies concrete pieces, this wires them."""

from __future__ import annotations

from pathlib import Path

from research_navigator.generate.llm import LanguageModel

from .agent import Agent
from .graph import build_agent_graph
from .nodes import AgentDeps, Generating
from .router import Router
from .settings import AgentSettings
from .tools import CorpusIndex


def build_agent(
    *,
    generator: Generating,
    manifest_path: Path,
    llm: LanguageModel | None,
    settings: AgentSettings,
) -> Agent:
    """Build the agent. ``generator`` is the M2 Generator (retriever+LLM inside);
    ``llm`` is used only for the router's ambiguous-query fallback (may be None to
    run rules-only)."""
    corpus = CorpusIndex.from_manifest(manifest_path)
    router = Router(settings=settings, llm=llm)
    deps = AgentDeps(generator=generator, corpus=corpus, router=router, settings=settings)
    graph = build_agent_graph(deps)
    return Agent(graph, deps)
