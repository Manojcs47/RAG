"""Qdrant health check."""

from __future__ import annotations

from research_navigator.config import Settings
from research_navigator.ingest.factory import build_client
from research_navigator.logging import get_logger

log = get_logger(__name__)


def check_health(settings: Settings) -> bool:
    """Return True if Qdrant is reachable, logging loudly on failure.

    We never swallow the error silently (a ground rule): we log it as a
    structured event and surface a boolean the caller must handle.
    """
    client = build_client(settings.qdrant)
    try:
        client.get_collections()
    except Exception as exc:
        log.error("qdrant_healthcheck_failed", error=str(exc))
        return False
    else:
        log.info("qdrant_healthcheck_ok", host=settings.qdrant.host)
        return True
    finally:
        client.close()
