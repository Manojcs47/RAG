from __future__ import annotations

import pytest

from research_navigator.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.qdrant_port == 6333
    assert s.collection_name == "research_navigator"
    assert 0.0 <= s.refusal_similarity_threshold <= 1.0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RN_QDRANT_PORT", "7000")
    s = Settings()
    assert s.qdrant_port == 7000
