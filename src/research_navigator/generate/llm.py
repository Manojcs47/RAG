"""Language-model port + OpenAI-compatible implementation.

The generator depends only on the ``LanguageModel`` protocol, so tests inject a
fake and the backend is swappable. ``base_url`` lets an OpenAI-compatible OSS
server (vLLM, Ollama, LM Studio) slot in without code changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import structlog

if TYPE_CHECKING:
    from ..config import LLMSettings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LanguageModel(Protocol):
    def complete(self, messages: list[ChatMessage]) -> str: ...


class OpenAIChatModel:
    """Chat-completions backend (default model: gpt-4o-mini)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        api_key: str | None,
        base_url: str | None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # fail loud
            raise RuntimeError(
                "openai is required for generation but is not installed (`uv add openai`)."
            ) from exc
        self._client: Any = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    def complete(self, messages: list[ChatMessage]) -> str:
        from openai import APIStatusError, RateLimitError

        payload = [{"role": m.role, "content": m.content} for m in messages]
        attempt = 0
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=payload,
                )
            except (RateLimitError, APIStatusError) as exc:
                transient = isinstance(exc, RateLimitError) or (
                    isinstance(exc, APIStatusError) and exc.status_code >= 500
                )
                if not transient or attempt >= self._max_retries:
                    log.error(
                        "llm_call_failed",
                        model=self._model,
                        attempt=attempt,
                        error=str(exc),
                    )
                    raise
                delay = self._retry_base_delay * (2**attempt)
                log.warning(
                    "llm_call_retrying",
                    model=self._model,
                    attempt=attempt,
                    delay_s=delay,
                    error=str(exc),
                )
                time.sleep(delay)
                attempt += 1
                continue
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM returned empty content")
            return str(content)


def build_llm(settings: LLMSettings) -> LanguageModel:
    """Construct the production language model. Fails loud on config errors."""
    try:
        model: LanguageModel = OpenAIChatModel(
            settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=settings.max_retries,
            retry_base_delay=settings.retry_base_delay,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to build LLM ({settings.model!r}): {exc}") from exc
    return model
