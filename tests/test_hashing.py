"""Unit tests for content hashing and chunk identity."""

from __future__ import annotations

from research_navigator.chunk.hashing import chunk_uid, content_hash, normalize_for_hash


def test_normalize_collapses_cosmetic_whitespace() -> None:
    assert normalize_for_hash("a  b\t c \r\n") == "a b c"
    assert normalize_for_hash("x\r\ny") == "x\ny"


def test_content_hash_is_stable_and_cosmetically_insensitive() -> None:
    a = content_hash("The cat sat.")
    b = content_hash("The  cat sat.   ")  # extra internal/trailing whitespace
    c = content_hash("The cat sat.\r\n")
    assert a == b == c
    assert len(a) == 64  # sha256 hex


def test_content_hash_changes_on_real_edit() -> None:
    assert content_hash("The cat sat.") != content_hash("The dog sat.")


def test_chunk_uid_is_deterministic_and_composed() -> None:
    h = content_hash("hello world")
    uid = chunk_uid("arxiv-1706.03762", 3, h)
    assert uid == f"arxiv-1706.03762:3:{h[:16]}"
    assert chunk_uid("d", 3, h) == chunk_uid("d", 3, h)
