"""M4 evaluation harness for the AI Research Navigator.

Public surface:
* :mod:`.golden` — the labelled golden set (loader, subsets, validation).
* :mod:`.harness` — ``run_eval`` + ``EvalConfig`` (the config comparison).
* :mod:`.report` — ``EvalReport`` (JSON + one-page Markdown).
* :mod:`.factory` — ``build_real_builder`` (wires the live M2/M3 stack; CLI only).

The harness depends on the concrete stack only through the structural ports in
:mod:`.ports`, so everything except :mod:`.factory` runs and tests offline.
"""

from __future__ import annotations

from research_navigator.eval.cost import CostModel, make_token_counter
from research_navigator.eval.golden import (
    GoldenQuestion,
    GoldenSet,
    load_golden_set,
    route_names,
)
from research_navigator.eval.harness import (
    ComponentBuilder,
    ComponentBundle,
    EvalConfig,
    default_configs,
    run_eval,
)
from research_navigator.eval.judge import (
    CITATION_JUDGE_RUBRIC,
    CitationJudge,
    JudgeVerdict,
)
from research_navigator.eval.report import ConfigReport, EvalReport, summarize
from research_navigator.eval.runner import EvalRunner, QueryRecord, RunResult
from research_navigator.eval.settings import EvalSettings

__all__ = [
    "CITATION_JUDGE_RUBRIC",
    "CitationJudge",
    "ComponentBuilder",
    "ComponentBundle",
    "ConfigReport",
    "CostModel",
    "EvalConfig",
    "EvalReport",
    "EvalRunner",
    "EvalSettings",
    "GoldenQuestion",
    "GoldenSet",
    "JudgeVerdict",
    "QueryRecord",
    "RunResult",
    "default_configs",
    "load_golden_set",
    "make_token_counter",
    "route_names",
    "run_eval",
    "summarize",
]
