"""Token, cost and latency accounting.

``tiktoken`` gives exact OpenAI token counts but lazily downloads its BPE vocabulary the
first time it encodes; in an air-gapped or sandboxed run that download fails. So the
token counter treats ``tiktoken`` as **optional**: it is used when it works and silently
— but logged once — falls back to a ``len/4`` heuristic otherwise. Counts are therefore
"exact-or-approximate-but-stable", which is honest and enough to compare configurations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import structlog

from research_navigator.eval.metrics import mean, percentile

log = structlog.get_logger(__name__)

_CHARS_PER_TOKEN = 4  # heuristic used only when tiktoken is unavailable


class TokenCounter(Protocol):
    """Anything that can turn text into an (approximate) token count."""

    def count(self, text: str) -> int: ...

    @property
    def is_exact(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class HeuristicCounter:
    """Offline fallback: ~1 token per 4 characters. Deterministic, no dependencies."""

    def count(self, text: str) -> int:
        return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN) if text else 0

    @property
    def is_exact(self) -> bool:
        return False


class TiktokenCounter:
    """Exact OpenAI token counts via tiktoken, with a lazy, fail-soft encoder."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._encoder: object | None = None
        self._degraded = False
        self._fallback = HeuristicCounter()

    def _ensure_encoder(self) -> object | None:
        if self._encoder is not None or self._degraded:
            return self._encoder
        try:
            import tiktoken

            try:
                self._encoder = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("o200k_base")
        except Exception as exc:
            self._degraded = True
            log.warning(
                "tiktoken_unavailable_falling_back_to_heuristic",
                model=self._model,
                error=str(exc),
            )
        return self._encoder

    def count(self, text: str) -> int:
        if not text:
            return 0
        enc = self._ensure_encoder()
        if enc is None:
            return self._fallback.count(text)
        # tiktoken's Encoding.encode returns list[int]
        return len(enc.encode(text))  # type: ignore[attr-defined]

    @property
    def is_exact(self) -> bool:
        return self._ensure_encoder() is not None


def make_token_counter(model: str) -> TokenCounter:
    """Prefer exact tiktoken counts; fall back to the heuristic if vocab can't load."""
    counter = TiktokenCounter(model)
    if counter.is_exact:
        return counter
    return HeuristicCounter()


@dataclass(frozen=True, slots=True)
class CostModel:
    """USD pricing per 1000 tokens (values come from :class:`EvalSettings`)."""

    price_per_1k_input_usd: float
    price_per_1k_output_usd: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000.0 * self.price_per_1k_input_usd
            + output_tokens / 1000.0 * self.price_per_1k_output_usd
        )


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency summary over a set of per-query wall-clock samples (milliseconds)."""

    n: int
    mean_ms: float
    max_ms: float
    percentiles_ms: dict[int, float]

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float], pcts: Sequence[int]) -> LatencyStats:
        return cls(
            n=len(samples_ms),
            mean_ms=mean(samples_ms),
            max_ms=max(samples_ms) if samples_ms else 0.0,
            percentiles_ms={p: percentile(samples_ms, p) for p in pcts},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "mean_ms": round(self.mean_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "percentiles_ms": {str(p): round(v, 2) for p, v in self.percentiles_ms.items()},
        }


class Stopwatch:
    """Context manager capturing elapsed wall-clock time in milliseconds.

    Uses ``time.perf_counter`` (monotonic). Injectable so tests can supply a fake clock.
    """

    def __init__(self, clock: object | None = None) -> None:
        self._clock = clock if clock is not None else time.perf_counter
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = float(self._clock())  # type: ignore[operator]
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (float(self._clock()) - self._start) * 1000.0  # type: ignore[operator]
