"""Retry/backoff behaviour of OpenAIChatModel (M5 resilience follow-up).

The live `make eval` run against a rate-limited free-tier backend showed most
question failures were transient 429s. These tests pin down: retry on
429/5xx, no retry on 4xx, exhaustion still fails loud, and a successful
retry returns the completion. The real ``openai.OpenAI`` client is
constructed (no network call happens in its constructor) then its
``chat.completions.create`` is replaced with a mock so no network I/O occurs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError, RateLimitError

from research_navigator.generate.llm import ChatMessage, OpenAIChatModel

_REQUEST = httpx.Request("POST", "http://test.invalid")


def _status_error(cls: type[APIStatusError], status_code: int) -> APIStatusError:
    resp = httpx.Response(status_code, request=_REQUEST)
    return cls(f"status {status_code}", response=resp, body=None)


def _fake_response(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _model(
    max_retries: int = 3, retry_base_delay: float = 0.0
) -> tuple[OpenAIChatModel, MagicMock]:
    model = OpenAIChatModel(
        "test-model",
        temperature=0.0,
        max_tokens=10,
        api_key="test-key",  # pragma: allowlist secret
        base_url=None,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    create = MagicMock()
    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return model, create


def test_succeeds_first_try() -> None:
    model, create = _model()
    create.return_value = _fake_response("hello")
    assert model.complete([ChatMessage(role="user", content="hi")]) == "hello"
    assert create.call_count == 1


def test_retries_on_rate_limit_then_succeeds() -> None:
    model, create = _model()
    create.side_effect = [_status_error(RateLimitError, 429), _fake_response("ok")]
    assert model.complete([ChatMessage(role="user", content="hi")]) == "ok"
    assert create.call_count == 2


def test_retries_on_server_error_then_succeeds() -> None:
    model, create = _model()
    create.side_effect = [_status_error(APIStatusError, 500), _fake_response("ok")]
    assert model.complete([ChatMessage(role="user", content="hi")]) == "ok"
    assert create.call_count == 2


def test_does_not_retry_on_client_error() -> None:
    model, create = _model()
    create.side_effect = _status_error(APIStatusError, 400)
    with pytest.raises(APIStatusError):
        model.complete([ChatMessage(role="user", content="hi")])
    assert create.call_count == 1


def test_exhausts_retries_and_raises() -> None:
    model, create = _model(max_retries=2)
    create.side_effect = [_status_error(RateLimitError, 429) for _ in range(3)]
    with pytest.raises(RateLimitError):
        model.complete([ChatMessage(role="user", content="hi")])
    assert create.call_count == 3  # initial attempt + 2 retries


def test_empty_content_raises() -> None:
    model, create = _model()
    create.return_value = _fake_response(None)
    with pytest.raises(RuntimeError, match="empty content"):
        model.complete([ChatMessage(role="user", content="hi")])
