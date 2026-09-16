"""Query-side vocabulary derived from the corpus manifest.

The tag list and available content types are *loaded*, never hardcoded, so a
wholesale corpus swap keeps filter inference correct (assignment ground rule:
'treat the corpus as a variable, not a constant')."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ALNUM = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and collapse to space-separated alphanumeric tokens."""
    return " ".join(_ALNUM.findall(text.lower()))


@dataclass(frozen=True)
class QueryCatalog:
    """Everything filter inference needs to know about the current corpus."""

    tags: tuple[str, ...]  # canonical, case-sensitive payload tag values
    content_types: frozenset[str]  # content_type values present in the corpus

    @classmethod
    def from_manifest_rows(cls, rows: list[dict[str, object]]) -> QueryCatalog:
        """Build from manifest document rows (each has ``tags`` + ``content_type``)."""
        tags: dict[str, None] = {}
        ctypes: set[str] = set()
        for row in rows:
            raw_tags = row.get("tags")
            if isinstance(raw_tags, list):
                for tag in raw_tags:
                    if isinstance(tag, str):
                        tags.setdefault(tag, None)
            ct = row.get("content_type")
            if isinstance(ct, str):
                ctypes.add(ct)
        return cls(tags=tuple(tags), content_types=frozenset(ctypes))

    def match_tags(self, query: str) -> list[str]:
        """Return canonical tags whose normalized phrase appears as whole
        token(s) in the query. Case-insensitive on the query side; multiword
        tags like ``long_context`` match the phrase 'long context'."""
        padded = f" {normalize(query)} "
        hits: list[str] = []
        for tag in self.tags:
            phrase = normalize(tag.replace("_", " "))
            if phrase and f" {phrase} " in padded:
                hits.append(tag)
        return hits
