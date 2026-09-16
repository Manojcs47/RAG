"""Result models for ingest / stats / validate (all JSON-serializable)."""

from __future__ import annotations

from pydantic import BaseModel, computed_field


class DocIngestResult(BaseModel):
    doc_id: str
    added: int
    deleted: int
    unchanged: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def writes(self) -> int:
        """Point-level writes performed (adds + deletes)."""
        return self.added + self.deleted


class IngestReport(BaseModel):
    collection: str
    docs: list[DocIngestResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def added(self) -> int:
        return sum(d.added for d in self.docs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def deleted(self) -> int:
        return sum(d.deleted for d in self.docs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unchanged(self) -> int:
        return sum(d.unchanged for d in self.docs)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def writes(self) -> int:
        return sum(d.writes for d in self.docs)


class CollectionStats(BaseModel):
    collection: str
    total_points: int
    by_content_type: dict[str, int]
    by_year: dict[int, int]
    foundational: int


class ValidationIssue(BaseModel):
    doc_id: str | None
    kind: str  # e.g. "missing_points", "stale_points", "count_mismatch"
    detail: str


class ValidationReport(BaseModel):
    collection: str
    ok: bool
    expected_points: int
    actual_points: int
    issues: list[ValidationIssue]
