"""Language-model port + OpenAI-compatible implementation.

The generator depends only on the ``LanguageModel`` protocol, so tests inject a
fake and the backend is swappable. ``base_url`` lets an OpenAI-compatible OSS
server (vLLM, Ollama, LM Studio) slot in without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..config import LLMSettings


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

    def complete(self, messages: list[ChatMessage]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
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
        )
    except Exception as exc:
        raise RuntimeError(f"failed to build LLM ({settings.model!r}): {exc}") from exc
    return model
