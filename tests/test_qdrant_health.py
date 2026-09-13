from __future__ import annotations

import pytest

from research_navigator.common.qdrant import check_health
from research_navigator.config import get_settings


@pytest.mark.integration
def test_qdrant_reachable() -> None:
    assert check_health(get_settings()) is True
