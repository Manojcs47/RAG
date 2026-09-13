"""Qdrant client factory and health check."""

from __future__ import annotations

from qdrant_client import QdrantClient

from research_navigator.config import Settings
from research_navigator.logging import get_logger

log = get_logger(__name__)


def make_client(settings: Settings) -> QdrantClient:
    """Construct a QdrantClient from settings."""
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        api_key=settings.qdrant_api_key,
    )


def check_health(settings: Settings) -> bool:
    """Return True if Qdrant is reachable, logging loudly on failure.

    We never swallow the error silently (a ground rule): we log it as a
    structured event and surface a boolean the caller must handle.
    """
    client = make_client(settings)
    try:
        client.get_collections()
    except Exception as exc:
        log.error("qdrant_healthcheck_failed", error=str(exc))
        return False
    else:
        log.info("qdrant_healthcheck_ok", host=settings.qdrant_host)
        return True
    finally:
        client.close()
