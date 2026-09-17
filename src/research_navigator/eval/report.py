"""Aggregate per-question records into a comparison report (JSON + Markdown)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_navigator.eval import metrics
from research_navigator.eval.runner import RunResult
from research_navigator.eval.settings import EvalSettings


@dataclass(frozen=True, slots=True)
class ConfigReport:
    """Aggregated metrics for a single configuration."""

    config_name: str
    n_questions: int
    n_errors: int
    routing_accuracy: float
    routing_per_route: dict[str, float]
    routing_confusion: dict[str, dict[str, int]]
    retrieval_precision_at_k: dict[int, float]
    retrieval_recall_at_k: dict[int, float]
    retrieval_f1_primary: float
    refusal: dict[str, float]
    judge_faithfulness_rate: float
    judge_mean_score: float
    judge_parse_failures: int
    judge_n: int
    latency: dict[str, Any]
    total_cost_usd: float
    mean_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "n_questions": self.n_questions,
            "n_errors": self.n_errors,
            "routing_accuracy": round(self.routing_accuracy, 4),
            "routing_per_route": {k: round(v, 4) for k, v in self.routing_per_route.items()},
            "routing_confusion": self.routing_confusion,
            "retrieval_precision_at_k": {
                str(k): round(v, 4) for k, v in self.retrieval_precision_at_k.items()
            },
            "retrieval_recall_at_k": {
                str(k): round(v, 4) for k, v in self.retrieval_recall_at_k.items()
            },
            "retrieval_f1_primary": round(self.retrieval_f1_primary, 4),
            "refusal": {k: round(v, 4) for k, v in self.refusal.items()},
            "judge_faithfulness_rate": round(self.judge_faithfulness_rate, 4),
            "judge_mean_score": round(self.judge_mean_score, 4),
            "judge_parse_failures": self.judge_parse_failures,
            "judge_n": self.judge_n,
            "latency": self.latency,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "mean_cost_usd": round(self.mean_cost_usd, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


def summarize(run: RunResult, settings: EvalSettings) -> ConfigReport:
    """Fold a run's records into a single :class:`ConfigReport`."""
    records = run.records
    n = len(records)
    n_err = sum(1 for r in records if r.error is not None)

    routing_pairs = [
        (r.predicted_route or "<none>", r.expected_route) for r in records if r.error is None
    ]
    routing_acc = metrics.accuracy(routing_pairs)
    routing_per_route = metrics.per_label_accuracy(routing_pairs)
    confusion = metrics.confusion_counts(routing_pairs)

    retr = [r for r in records if r.is_retrieval and r.error is None]
    prec_at_k = {
        k: metrics.mean([r.precision_at_k.get(k, 0.0) for r in retr]) for k in settings.k_values
    }
    rec_at_k = {
        k: metrics.mean([r.recall_at_k.get(k, 0.0) for r in retr]) for k in settings.k_values
    }
    pk = settings.primary_k
    f1_primary = metrics.f1(prec_at_k.get(pk, 0.0), rec_at_k.get(pk, 0.0))

    refusal_records = [r for r in records if r.error is None]
    refusal = metrics.binary_rates(
        [r.refused for r in refusal_records],
        [r.expect_refusal for r in refusal_records],
    )

    judged = [r.judge for r in records if r.judge is not None]
    parse_fail = sum(1 for v in judged if not v.parse_ok)
    faithful_rate = sum(1 for v in judged if v.faithful) / len(judged) if judged else 0.0
    mean_score = metrics.mean([v.score for v in judged])

    latency = run.latency_stats(settings.latency_percentiles).to_dict()

    total_cost = sum(r.cost_usd for r in records)
    total_in = sum(r.input_tokens for r in records)
    total_out = sum(r.output_tokens for r in records)

    return ConfigReport(
        config_name=run.config_name,
        n_questions=n,
        n_errors=n_err,
        routing_accuracy=routing_acc,
        routing_per_route=routing_per_route,
        routing_confusion=confusion,
        retrieval_precision_at_k=prec_at_k,
        retrieval_recall_at_k=rec_at_k,
        retrieval_f1_primary=f1_primary,
        refusal=refusal,
        judge_faithfulness_rate=faithful_rate,
        judge_mean_score=mean_score,
        judge_parse_failures=parse_fail,
        judge_n=len(judged),
        latency=latency,
        total_cost_usd=total_cost,
        mean_cost_usd=total_cost / n if n else 0.0,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
    )


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The full comparison report across configurations."""

    generated_at: str
    golden_set_size: int
    primary_k: int
    token_counts_exact: bool
    configs: tuple[ConfigReport, ...]
    per_question: dict[str, list[dict[str, Any]]]
    limitations: tuple[str, ...]

    @classmethod
    def build(
        cls,
        runs: Sequence[RunResult],
        settings: EvalSettings,
        *,
        golden_set_size: int,
        token_counts_exact: bool,
        limitations: Sequence[str] = (),
    ) -> EvalReport:
        configs = tuple(summarize(r, settings) for r in runs)
        per_q = {r.config_name: [rec.to_dict() for rec in r.records] for r in runs}
        return cls(
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            golden_set_size=golden_set_size,
            primary_k=settings.primary_k,
            token_counts_exact=token_counts_exact,
            configs=configs,
            per_question=per_q,
            limitations=tuple(limitations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "golden_set_size": self.golden_set_size,
            "primary_k": self.primary_k,
            "token_counts_exact": self.token_counts_exact,
            "configs": [c.to_dict() for c in self.configs],
            "per_question": self.per_question,
            "limitations": list(self.limitations),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    def to_markdown(self) -> str:
        pk = self.primary_k
        lines: list[str] = []
        lines.append("# AI Research Navigator — Evaluation Report")
        lines.append("")
        token_mode = "exact (tiktoken)" if self.token_counts_exact else "approximate (heuristic)"
        lines.append(
            f"_Generated {self.generated_at} · golden set: "
            f"{self.golden_set_size} questions · primary k = {pk} · "
            f"token counts {token_mode}._"
        )
        lines.append("")

        # --- headline comparison table -------------------------------------
        lines.append("## Configuration comparison")
        lines.append("")
        header = (
            "| Config | Routing acc | "
            f"P@{pk} | R@{pk} | F1@{pk} | "
            "Faithfulness | Refusal acc | p50 ms | p95 ms | $/query | Errors |"
        )
        sep = "|" + "---|" * 11
        lines.append(header)
        lines.append(sep)
        for c in self.configs:
            p = c.retrieval_precision_at_k.get(pk, 0.0)
            r = c.retrieval_recall_at_k.get(pk, 0.0)
            p50 = c.latency["percentiles_ms"].get("50", 0.0)
            p95 = c.latency["percentiles_ms"].get("95", 0.0)
            faith = f"{c.judge_faithfulness_rate:.2f}" if c.judge_n else "n/a"
            lines.append(
                f"| {c.config_name} | {c.routing_accuracy:.2f} | {p:.2f} | {r:.2f} | "
                f"{c.retrieval_f1_primary:.2f} | {faith} | "
                f"{c.refusal['accuracy']:.2f} | {p50:.0f} | {p95:.0f} | "
                f"{c.mean_cost_usd:.5f} | {c.n_errors} |"
            )
        lines.append("")

        # --- per-config detail ---------------------------------------------
        for c in self.configs:
            lines.append(f"## {c.config_name}")
            lines.append("")
            lines.append(f"- **Questions:** {c.n_questions} ({c.n_errors} errored)")
            lines.append(
                "- **Retrieval (mean over retrieval subset):** "
                + ", ".join(
                    f"P@{k}={c.retrieval_precision_at_k[k]:.2f}/R@{k}={c.retrieval_recall_at_k[k]:.2f}"
                    for k in sorted(c.retrieval_precision_at_k)
                )
            )
            lines.append(
                f"- **Routing accuracy:** {c.routing_accuracy:.2f} overall; per route: "
                + ", ".join(
                    f"{route}={acc:.2f}" for route, acc in sorted(c.routing_per_route.items())
                )
            )
            lines.append(
                f"- **Refusal:** acc={c.refusal['accuracy']:.2f}, "
                f"recall={c.refusal['recall']:.2f} (refuses when it should), "
                f"precision={c.refusal['precision']:.2f} (does not over-refuse), "
                f"n={int(c.refusal['n'])}"
            )
            if c.judge_n:
                lines.append(
                    f"- **Citation faithfulness (LLM judge):** "
                    f"rate={c.judge_faithfulness_rate:.2f}, "
                    f"mean score={c.judge_mean_score:.2f} over n={c.judge_n} answers "
                    f"({c.judge_parse_failures} judge parse failures)"
                )
            else:
                lines.append(
                    "- **Citation faithfulness (LLM judge):** not evaluated "
                    "(judge disabled or no cited answers)"
                )
            lat = c.latency
            pctl = ", ".join(f"p{p}={v:.0f}ms" for p, v in lat["percentiles_ms"].items())
            lines.append(
                f"- **Latency:** mean={lat['mean_ms']:.0f}ms, {pctl}, max={lat['max_ms']:.0f}ms"
            )
            lines.append(
                f"- **Cost:** total=${c.total_cost_usd:.4f}, "
                f"mean=${c.mean_cost_usd:.5f}/query, "
                f"tokens in/out={c.total_input_tokens}/{c.total_output_tokens}"
            )
            lines.append("")

        # --- honest limitations --------------------------------------------
        lines.append("## Limitations & honest notes")
        lines.append("")
        default_limits = [
            "Retrieval P/R is scored at the document level against a hand-labelled "
            "golden set; expected-source labels are best-effort and may under-count "
            "valid sources.",
            "Token/cost figures are estimated from the query, cited source text and "
            "the answer (the exact generation prompt is not observed by the harness) "
            "and are indicative, not billing-accurate.",
            "Citation faithfulness uses a single LLM judge; it is a proxy and inherits "
            "the judge model's biases. Parse failures are counted as non-faithful.",
        ]
        for lim in [*self.limitations, *default_limits]:
            lines.append(f"- {lim}")
        lines.append("")
        return "\n".join(lines)

    def write(self, settings: EvalSettings) -> tuple[Path, Path]:
        settings.report_dir.mkdir(parents=True, exist_ok=True)
        json_path = settings.report_dir / settings.json_report_name
        md_path = settings.report_dir / settings.markdown_report_name
        json_path.write_text(self.to_json(), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path
