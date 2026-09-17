"""Top-level harness: run every configuration over the golden set and build the report.

The harness never constructs the concrete stack itself. It receives a ``builder``
callable ``(EvalConfig) -> ComponentBundle`` supplied by the CLI/factory, so ``eval``
stays free of Qdrant/FastEmbed/OpenAI imports and the whole flow can be exercised with
fakes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import structlog

from research_navigator.eval.cost import CostModel, TokenCounter, make_token_counter
from research_navigator.eval.golden import GoldenSet
from research_navigator.eval.judge import CitationJudge
from research_navigator.eval.ports import Agent, Retriever, SupportsComplete
from research_navigator.eval.report import EvalReport
from research_navigator.eval.runner import EvalRunner
from research_navigator.eval.settings import EvalSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """A named retrieval/generation configuration to evaluate.

    ``overrides`` are opaque to the harness; the concrete builder interprets them (e.g.
    ``{"retrieval_mode": "dense_only"}`` or ``{"use_metadata_filter": False}``).
    """

    name: str
    overrides: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComponentBundle:
    """What the builder must return for a given config."""

    agent: Agent
    retriever: Retriever
    judge_model: SupportsComplete | None = None


def default_configs() -> tuple[EvalConfig, ...]:
    """The two configurations the report compares by default (hybrid vs dense-only)."""
    return (
        EvalConfig("hybrid", {"retrieval_mode": "hybrid"}),
        EvalConfig("dense_only", {"retrieval_mode": "dense_only"}),
    )


ComponentBuilder = Callable[[EvalConfig], ComponentBundle]


def run_eval(
    *,
    builder: ComponentBuilder,
    golden: GoldenSet,
    settings: EvalSettings,
    configs: Sequence[EvalConfig] | None = None,
    limitations: Sequence[str] = (),
) -> EvalReport:
    """Run each configuration over the golden set and fold results into one report."""
    variants = tuple(configs) if configs is not None else default_configs()
    golden.assert_route_coverage(minimum=3)

    token_counter: TokenCounter = make_token_counter(settings.tokenizer_model)
    cost_model = CostModel(
        price_per_1k_input_usd=settings.price_per_1k_input_usd,
        price_per_1k_output_usd=settings.price_per_1k_output_usd,
    )

    runs = []
    for cfg in variants:
        log.info("eval_config_start", config=cfg.name, overrides=cfg.overrides)
        bundle = builder(cfg)
        judge = None
        if settings.judge_enabled and bundle.judge_model is not None:
            judge = CitationJudge(bundle.judge_model, min_score=settings.judge_min_score)
        elif settings.judge_enabled and bundle.judge_model is None:
            log.warning("judge_enabled_but_no_model", config=cfg.name)
        runner = EvalRunner(
            agent=bundle.agent,
            retriever=bundle.retriever,
            token_counter=token_counter,
            cost_model=cost_model,
            settings=settings,
            judge=judge,
        )
        runs.append(runner.run(golden, config_name=cfg.name))

    return EvalReport.build(
        runs,
        settings,
        golden_set_size=len(golden),
        token_counts_exact=token_counter.is_exact,
        limitations=limitations,
    )
