"""The golden set: ~40 labelled questions spanning the six routes.

Each :class:`GoldenQuestion` carries the label(s) the harness scores against:
* ``expected_route`` — for routing accuracy (every question),
* ``expected_doc_ids`` — for retrieval precision/recall@k (the retrieval subset),
* ``expect_refusal`` — for refusal correctness (the deliberately out-of-corpus subset).

``Route`` is imported from :mod:`research_navigator.agents.state` so the set of valid
route names has exactly one source of truth and cannot drift from the agent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_navigator.agents.state import Route

_VALID_ROUTES: frozenset[str] = frozenset(r.value for r in Route)


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """One labelled evaluation question (immutable, JSON-serialisable)."""

    id: str
    query: str
    expected_route: str
    expected_doc_ids: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    expect_refusal: bool = False
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("GoldenQuestion.id must be non-empty")
        if not self.query.strip():
            raise ValueError(f"[{self.id}] query must be non-empty")
        if self.expected_route not in _VALID_ROUTES:
            raise ValueError(
                f"[{self.id}] expected_route={self.expected_route!r} "
                f"is not one of {sorted(_VALID_ROUTES)}"
            )
        if self.expect_refusal and self.expected_route != Route.OUT_OF_SCOPE.value:
            raise ValueError(
                f"[{self.id}] expect_refusal is only valid for the "
                f"{Route.OUT_OF_SCOPE.value!r} route"
            )
        if not self.expect_refusal and self.expected_route == Route.OUT_OF_SCOPE.value:
            raise ValueError(f"[{self.id}] out_of_scope questions must set expect_refusal=True")

    @property
    def is_retrieval(self) -> bool:
        """True when this question has document labels to score retrieval against."""
        return bool(self.expected_doc_ids)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldenQuestion:
        return cls(
            id=str(raw["id"]),
            query=str(raw["query"]),
            expected_route=str(raw["expected_route"]),
            expected_doc_ids=tuple(raw.get("expected_doc_ids", ()) or ()),
            expected_sections=tuple(raw.get("expected_sections", ()) or ()),
            expect_refusal=bool(raw.get("expect_refusal", False)),
            tags=tuple(raw.get("tags", ()) or ()),
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "expected_route": self.expected_route,
            "expected_doc_ids": list(self.expected_doc_ids),
            "expected_sections": list(self.expected_sections),
            "expect_refusal": self.expect_refusal,
            "tags": list(self.tags),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class GoldenSet:
    """An immutable, validated collection of golden questions."""

    questions: tuple[GoldenQuestion, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for q in self.questions:
            if q.id in seen:
                raise ValueError(f"duplicate golden id: {q.id!r}")
            seen.add(q.id)

    def __iter__(self) -> Iterator[GoldenQuestion]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    # --- subsets ------------------------------------------------------------
    def by_route(self, route: str) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if q.expected_route == route)

    @property
    def retrieval_subset(self) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if q.is_retrieval)

    @property
    def refusal_subset(self) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if q.expect_refusal)

    @property
    def answerable_subset(self) -> tuple[GoldenQuestion, ...]:
        return tuple(q for q in self.questions if not q.expect_refusal)

    def route_coverage(self) -> dict[str, int]:
        return {r: len(self.by_route(r)) for r in sorted(_VALID_ROUTES)}

    def assert_route_coverage(self, minimum: int = 3) -> None:
        """Fail loudly if any route is under-represented (acceptance is >=3/route)."""
        thin = {r: n for r, n in self.route_coverage().items() if n < minimum}
        if thin:
            raise ValueError(f"routes with fewer than {minimum} questions: {thin}")

    # --- loaders ------------------------------------------------------------
    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> GoldenSet:
        return cls(tuple(GoldenQuestion.from_dict(r) for r in rows))

    @classmethod
    def from_json(cls, path: Path) -> GoldenSet:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["questions"] if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"golden set at {path} must be a list or {{'questions': [...]}}")
        return cls.from_rows(rows)

    @classmethod
    def load_default(cls) -> GoldenSet:
        """Load the golden set packaged next to this module."""
        return cls.from_json(default_golden_path())

    def to_dict(self) -> dict[str, Any]:
        return {"questions": [q.to_dict() for q in self.questions]}


def default_golden_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "golden_set.json"


def load_golden_set(path: Path | None) -> GoldenSet:
    """Load from ``path`` or the packaged default; a single choke-point for the CLI."""
    return GoldenSet.from_json(path) if path is not None else GoldenSet.load_default()


def route_names() -> Sequence[str]:
    return sorted(_VALID_ROUTES)
