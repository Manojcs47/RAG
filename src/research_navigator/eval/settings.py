"""Evaluation settings — every threshold, price and k-value lives here (no hardcoding).

Mirrors the S7 ``AgentSettings`` pattern: a plain :class:`pydantic.BaseModel` nested
under the top-level ``Settings`` as ``Settings.eval`` (env prefix ``RN_EVAL__*``). The
one-line patch that adds the field to ``config.Settings`` ships in
``patches/config.py.patch.md``.

Prices default to the published gpt-4o-mini rates but are *settings*, not constants —
they change, so they must be overridable and are recorded as "verify" in the ADR."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class EvalSettings(BaseModel):
    """Tunables for the M4 evaluation harness."""

    # --- golden set ---------------------------------------------------------
    golden_set_path: Path | None = Field(
        default=None,
        description="Golden-set JSON. None -> the packaged data/golden_set.json.",
    )

    # --- retrieval P/R@k ----------------------------------------------------
    k_values: tuple[int, ...] = Field(
        default=(3, 5, 8),
        description="k's at which precision/recall are reported (document level).",
    )
    primary_k: int = Field(
        default=5,
        description="The headline k used for the single-number retrieval summary.",
    )

    # --- recency-dependent routes ------------------------------------------
    now_year: int | None = Field(
        default=None,
        description="'Current' year for recency routes. None -> date.today().year.",
    )

    # --- citation-faithfulness judge ---------------------------------------
    judge_enabled: bool = Field(default=True)
    judge_model: str | None = Field(
        default=None,
        description="LLM name for the judge. None -> reuse the generation llm.",
    )
    judge_min_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="An answer counts as 'faithful' iff its judge score >= this.",
    )

    # --- cost accounting ----------------------------------------------------
    tokenizer_model: str = Field(default="gpt-4o-mini")
    price_per_1k_input_usd: float = Field(default=0.00015, ge=0.0)
    price_per_1k_output_usd: float = Field(default=0.00060, ge=0.0)

    # --- latency ------------------------------------------------------------
    latency_percentiles: tuple[int, ...] = Field(default=(50, 95))

    # --- output -------------------------------------------------------------
    report_dir: Path = Field(default=Path("eval"))
    json_report_name: str = Field(default="report.json")
    markdown_report_name: str = Field(default="report.md")

    @model_validator(mode="after")
    def _check_coherent(self) -> EvalSettings:
        if not self.k_values:
            raise ValueError("k_values must be non-empty")
        if any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must be positive")
        if self.primary_k not in self.k_values:
            raise ValueError(f"primary_k={self.primary_k} must be one of k_values={self.k_values}")
        if any(not 0 < p < 100 for p in self.latency_percentiles):
            raise ValueError("latency_percentiles must be in (0, 100)")
        return self

    @property
    def max_k(self) -> int:
        """The largest k we need to retrieve to cover every reported k."""
        return max(self.k_values)
