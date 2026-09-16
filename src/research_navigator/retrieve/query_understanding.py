"""Deterministic query understanding: intent hint + metadata filter inference.

Pure functions — no network, no LLM — so they are cheap and fully unit-testable
(M5 requires unit tests for filter inference). An LLM pass could later augment
this behind the same ``analyze`` signature.
"""

from __future__ import annotations

import datetime
import re

from .models import InferredFilters, QueryAnalysis, QueryIntent
from .settings import RetrieveSettings
from .vocab import QueryCatalog, normalize

# explicit 4-digit year, e.g. "since 2023", "2024 work"
_YEAR = re.compile(r"\b(20\d{2})\b")
# "last/past N years|months"
_LAST_N = re.compile(r"\b(?:last|past)\s+(\d+)\s+(year|month)s?\b")
# comparison cues
_COMPARE = re.compile(r"\b(?:vs\.?|versus|compare[ds]?|comparison|difference between)\b")
# paper-deep-dive cues
_PAPER_REF = re.compile(r"\b(?:arxiv|paper titled|the paper|this paper|deep[- ]dive)\b")
# reading-list cues
_FIND = re.compile(
    r"\b(?:recommend|reading list|papers on|papers about|what should i read|suggest)\b"
)


def _phrase_present(needle: str, normalized_query: str) -> bool:
    return f" {normalize(needle)} " in f" {normalized_query} "


def analyze(
    query: str,
    *,
    catalog: QueryCatalog,
    settings: RetrieveSettings,
    now_year: int | None = None,
) -> QueryAnalysis:
    """Infer intent + metadata filters from a raw query string."""
    raw = query
    norm = normalize(query)
    reasons: list[str] = []

    if now_year is None:
        now_year = settings.recency_reference_year or datetime.date.today().year

    # ---- recency -> year_gte ------------------------------------------------
    year_gte: int | None = None
    recency_hit = False
    if settings.infer_filters:
        m_last = _LAST_N.search(norm)
        if m_last:
            n, unit = int(m_last.group(1)), m_last.group(2)
            years_back = n if unit == "year" else max(1, (n + 11) // 12)
            year_gte = now_year - years_back
            recency_hit = True
            reasons.append(f"'last {n} {unit}s' -> year>={year_gte}")
        else:
            explicit = [int(y) for y in _YEAR.findall(norm)]
            if explicit:
                year_gte = min(explicit)
                reasons.append(f"explicit year -> year>={year_gte}")
            elif any(_phrase_present(cue, norm) for cue in settings.recency_cues):
                year_gte = now_year - settings.recency_window_years
                recency_hit = True
                reasons.append(f"recency cue -> year>={year_gte}")

    # ---- tags ---------------------------------------------------------------
    tags: tuple[str, ...] = ()
    if settings.infer_filters:
        matched = catalog.match_tags(query)
        if matched:
            tags = tuple(matched)
            reasons.append(f"tags {list(tags)}")

    # ---- content types ------------------------------------------------------
    content_types: tuple[str, ...] = ()
    if settings.infer_filters:
        found: list[str] = []
        for ctype, triggers in settings.content_type_triggers.items():
            if ctype not in catalog.content_types:
                continue  # not in this corpus -> never filter it in
            if any(_phrase_present(t, norm) for t in triggers):
                found.append(ctype)
        if found:
            content_types = tuple(dict.fromkeys(found))
            reasons.append(f"content_type {list(content_types)}")

    # ---- is_foundational ----------------------------------------------------
    is_foundational: bool | None = None
    if settings.infer_filters and any(
        _phrase_present(cue, norm) for cue in settings.foundational_cues
    ):
        is_foundational = True
        reasons.append("foundational cue -> is_foundational=True")

    # ---- intent (advisory) --------------------------------------------------
    if _COMPARE.search(norm):
        intent = QueryIntent.COMPARE
    elif _FIND.search(norm):
        intent = QueryIntent.FIND_PAPERS
    elif _PAPER_REF.search(norm):
        intent = QueryIntent.PAPER_SPECIFIC
    elif recency_hit:
        intent = QueryIntent.RECENT
    else:
        intent = QueryIntent.CONCEPT

    filters = InferredFilters(
        year_gte=year_gte,
        tags=tags,
        content_types=content_types,
        is_foundational=is_foundational,
    )
    return QueryAnalysis(
        raw=raw,
        normalized=norm,
        intent=intent,
        filters=filters,
        reasons=tuple(reasons),
    )
