from __future__ import annotations

import pytest

from research_navigator.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.qdrant.port == 6333
    assert s.qdrant.collection == "research_navigator"
    assert 0.0 <= s.retrieve.refusal_threshold <= 1.0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RN_QDRANT__PORT", "7000")
    s = Settings()
    assert s.qdrant.port == 7000
