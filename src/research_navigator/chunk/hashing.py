"""Content hashing for change detection and deterministic chunk identity.

The chunk id basis is ``(doc_id, chunk_index, content_hash)`` (assignment §1d).
``content_hash`` captures the chunk's *semantic content* so that editing a
document changes the hash of exactly the chunks whose text changed, enabling
Session 4's idempotent, targeted upserts.
"""

from __future__ import annotations

import hashlib
import re

_HORIZONTAL_WS = re.compile(r"[ \t]+")


def normalize_for_hash(text: str) -> str:
    """Normalize line endings and trailing/edge whitespace before hashing.

    Internal single spaces are preserved (collapsing all whitespace could merge
    semantically distinct chunks), but runs of spaces/tabs and trailing
    whitespace — which carry no meaning — are normalized so cosmetic diffs do
    not spuriously invalidate a chunk.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_WS.sub(" ", line).rstrip() for line in unified.split("\n")]
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    """SHA-256 hex digest of the normalized chunk text."""
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def chunk_uid(doc_id: str, chunk_index: int, content_hash_hex: str) -> str:
    """Stable chunk identity from ``(doc_id, chunk_index, content_hash)``.

    Session 4 maps this to a Qdrant point id (e.g. ``uuid5`` over this string).
    """
    return f"{doc_id}:{chunk_index}:{content_hash_hex[:16]}"
