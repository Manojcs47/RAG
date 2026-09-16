"""Translate InferredFilters -> a Qdrant Filter. Applied server-side at
retrieval time (never as a post-hoc Python filter, per M2)."""

from __future__ import annotations

from qdrant_client import models

from .models import InferredFilters


def build_qdrant_filter(f: InferredFilters) -> models.Filter | None:
    """Return a Qdrant Filter, or None when nothing was inferred.

    Semantics: constraints across *fields* are AND (``must``); multiple values
    within a field (tags, content_types) are OR (``MatchAny``).
    """
    if f.is_empty():
        return None

    must: list[models.Condition] = []

    if f.year_gte is not None or f.year_lte is not None:
        must.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=f.year_gte, lte=f.year_lte),
            )
        )
    if f.tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=list(f.tags))))
    if f.content_types:
        must.append(
            models.FieldCondition(
                key="content_type", match=models.MatchAny(any=list(f.content_types))
            )
        )
    if f.primary_categories:
        must.append(
            models.FieldCondition(
                key="primary_category",
                match=models.MatchAny(any=list(f.primary_categories)),
            )
        )
    if f.is_foundational is not None:
        must.append(
            models.FieldCondition(
                key="is_foundational",
                match=models.MatchValue(value=f.is_foundational),
            )
        )
    if f.doc_ids:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(f.doc_ids))))

    return models.Filter(must=must)
