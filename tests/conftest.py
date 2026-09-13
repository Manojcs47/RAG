"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from research_navigator.config import Settings


@pytest.fixture()
def settings() -> Settings:
    return Settings()
