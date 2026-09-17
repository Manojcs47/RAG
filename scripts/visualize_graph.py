"""Render the compiled M3 agent graph for the design notes.

Writes ``docs/agent_graph.mmd`` (Mermaid source, produced fully offline) and, when
``--png`` is passed, ``docs/agent_graph.png`` as well. The PNG path uses LangGraph's
``draw_mermaid_png()`` which, unless a local renderer is configured, calls the
mermaid.ink web API — hence it is opt-in and never part of ``make``/CI.

The graph *topology* is static (it does not depend on live model or corpus state), so
we assemble it with a null generator whose ``answer`` is never invoked during
rendering. This keeps visualization cheap: no Qdrant, no embedder, no API key.

Usage:
    python scripts/visualize_graph.py            # writes docs/agent_graph.mmd
    python scripts/visualize_graph.py --png      # also writes docs/agent_graph.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_navigator.agents import AgentDeps, CorpusIndex, Router, build_agent_graph
from research_navigator.agents.graph import graph_mermaid
from research_navigator.config import get_settings
from research_navigator.generate.models import Answer


class _NullGenerator:
    """Satisfies the ``Generating`` protocol. Never called during rendering — the
    graph structure is independent of any node's runtime behaviour."""

    def answer(self, query: str, *, now_year: int | None = None) -> Answer:
        raise RuntimeError("visualization must not execute nodes")


def _build_deps() -> AgentDeps:
    settings = get_settings()
    corpus = CorpusIndex.from_manifest(settings.paths.manifest)
    router = Router(settings=settings.agents, llm=None)
    return AgentDeps(
        generator=_NullGenerator(),
        corpus=corpus,
        router=router,
        settings=settings.agents,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the M3 agent graph.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs"),
        help="Directory for the rendered graph (default: docs/).",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also write a PNG (uses the mermaid.ink API — requires network).",
    )
    args = parser.parse_args()

    deps = _build_deps()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mmd_path = args.out_dir / "agent_graph.mmd"
    mmd_path.write_text(graph_mermaid(deps), encoding="utf-8")
    print(f"wrote {mmd_path}")

    if args.png:
        png_path = args.out_dir / "agent_graph.png"
        png = build_agent_graph(deps).get_graph().draw_mermaid_png()
        png_path.write_bytes(png)
        print(f"wrote {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
