"""The chunk record — this model IS the Qdrant payload (Session 4 attaches vectors).

It inherits from ``ManifestEntry`` so **every** document-level field is carried
through to every chunk (assignment §1c), and adds the derived per-chunk fields.
"""

from __future__ import annotations

from typing import Any

from research_navigator.chunk.hashing import chunk_uid
from research_navigator.common.manifest import ManifestEntry
from research_navigator.parse.models import SectionKind


class Chunk(ManifestEntry):
    """A retrievable chunk plus its full metadata payload."""

    # --- provenance / structure ---
    section_index: int
    section_title: str
    section_kind: SectionKind
    chunk_index: int  # global, monotonic within the document (0..n-1)

    # --- content + derived ---
    text: str
    token_count: int
    content_hash: str
    is_abstract: bool = False

    @property
    def uid(self) -> str:
        """Deterministic ``(doc_id, chunk_index, content_hash)`` identity."""
        return chunk_uid(self.doc_id, self.chunk_index, self.content_hash)

    def payload(self) -> dict[str, Any]:
        """JSON-ready payload for Qdrant.

        ``Any`` is justified here (M5 exception): a vector-store payload is
        arbitrary JSON by nature.
        """
        return self.model_dump(mode="json")
