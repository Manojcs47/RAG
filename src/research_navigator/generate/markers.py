"""Inline citation-marker handling: parse, VALIDATE against real sources, drop
fabricated markers, and renumber contiguously by first appearance.

This is the hard guarantee behind 'no fabricated citations': a marker survives
only if it points at a source that was actually retrieved. Anything else is
removed here, deterministically, regardless of what the model wrote.
"""

from __future__ import annotations

import re

from .models import Source

# matches [1], [1,2], [1, 2, 3]
_MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def parse_markers(text: str) -> list[int]:
    """All integer indices referenced by [n]/[n,m] markers, in order."""
    out: list[int] = []
    for m in _MARKER.finditer(text):
        out.extend(int(x) for x in re.split(r"\s*,\s*", m.group(1)))
    return out


def render_citations(text: str, sources: list[Source]) -> tuple[str, list[Source], list[int]]:
    """Rewrite the answer so every surviving marker maps to a real source.

    Returns (clean_text, used_sources_in_new_order, dropped_indices).
    - out-of-range markers (fabricated) are removed
    - kept markers are renumbered 1..N by first appearance
    - used_sources carry their NEW index; dropped_indices is for logging
    """
    n = len(sources)
    old_to_new: dict[int, int] = {}
    appearance_order: list[int] = []
    dropped: list[int] = []

    def _repl(match: re.Match[str]) -> str:
        kept: list[int] = []
        for old in (int(x) for x in re.split(r"\s*,\s*", match.group(1))):
            if 1 <= old <= n:
                if old not in old_to_new:
                    appearance_order.append(old)
                    old_to_new[old] = len(appearance_order)
                kept.append(old_to_new[old])
            else:
                dropped.append(old)
        if not kept:
            return ""
        return "[" + ", ".join(str(k) for k in kept) + "]"

    rewritten = _MARKER.sub(_repl, text)
    cleaned = _tidy(rewritten)

    used = [
        Source(**{**vars(sources[old - 1]), "index": old_to_new[old]}) for old in appearance_order
    ]
    return cleaned, used, dropped


def _tidy(text: str) -> str:
    """Cosmetic cleanup after markers are dropped."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def uncited_sentences(text: str) -> list[str]:
    """Diagnostic: sentences carrying no [n] marker (best-effort split)."""
    out: list[str] = []
    for sent in _SENTENCE.split(text.strip()):
        s = sent.strip()
        if s and not _MARKER.search(s):
            out.append(s)
    return out
