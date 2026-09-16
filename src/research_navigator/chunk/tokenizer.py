"""Length measurement for chunking.

Chunk length only matters insofar as it drives the embedding model's context
limit (``bge-small-en-v1.5`` truncates at 512 subword tokens). So the *right*
unit to measure in is that model's own subword tokenizer. We hide this behind a
``Tokenizer`` protocol so it is injectable in tests and swappable in Session 4.

``WhitespaceTokenizer`` is a deterministic, offline word-counter used in unit
tests and as a loud fallback if the model tokenizer cannot be loaded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)


@runtime_checkable
class Tokenizer(Protocol):
    """Anything that can measure text length in tokens."""

    def count(self, text: str) -> int: ...


class WhitespaceTokenizer:
    """Deterministic word-count tokenizer. No network, no model files."""

    def count(self, text: str) -> int:
        return len(text.split())


class HFTokenizer:
    """Subword tokenizer via the ``tokenizers`` library (no torch dependency).

    Length is measured in the embedding model's own subword units — the unit
    that actually governs 512-token truncation.
    """

    def __init__(self, model_id: str) -> None:
        from tokenizers import Tokenizer as _HFTokenizer  # lazy: keep import cost off hot path

        self._tok = _HFTokenizer.from_pretrained(model_id)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)


def build_tokenizer(model_id: str) -> Tokenizer:
    """Build the embedding-faithful tokenizer, falling back loudly if unavailable."""
    try:
        return HFTokenizer(model_id)
    except Exception as exc:
        log.warning(
            "tokenizer_load_failed_fallback_whitespace",
            model_id=model_id,
            error=str(exc),
        )
        return WhitespaceTokenizer()
