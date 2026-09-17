"""Adapters mapping the concrete M2/M3 objects onto the eval ports.

These are the *only* place ``eval`` names the real classes, and every mapping here is a
one-liner over an already-public field/method (Agent.run, AgentResult, the M2 citation
block, generate.llm.LanguageModel). Keeping them isolated means a change to a concrete
interface touches this file alone — the rest of the harness is protocol-typed.

NOTE: the retriever surface is confirmed against the repo — ``retriever.retrieve(query)``
returns a result whose ``.chunks`` each expose ``doc_id`` / ``score`` / ``section_title`` /
``text``. The concrete ``AgentResult`` internals (``route`` / ``refused`` / the cited
answer object) are read defensively via ``_field`` (dict-or-attribute) with safe
fallbacks, so a field rename degrades to a recorded empty rather than a crash — adjust
the ``_field`` keys here if they differ. ``AgentResult.answer`` is
``dict[str, Any] | None`` (``dataclasses.asdict``'d by ``agents/nodes.py`` for
JSON-serializable state), so its citations are plain dicts too, not attribute-bearing
objects — this was a real bug (fixed): using ``getattr`` on a dict never raises but
always returns the default, so citations silently read as empty and the M4
citation-faithfulness judge never ran on a real answer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from research_navigator.eval.ports import (
    AgentOutcome,
    CitedSource,
    RetrievedChunk,
    SupportsComplete,
)


@dataclass(frozen=True, slots=True)
class _Cite:
    """Concrete, serialisable citation satisfying the CitedSource port."""

    doc_id: str
    title: str
    section: str | None
    text: str


@dataclass(frozen=True, slots=True)
class _Outcome:
    route: str
    refused: bool
    answer_text: str
    citations: Sequence[CitedSource]


def _field(obj: object, key: str, default: object = "") -> object:
    """Read ``key`` off ``obj``, which may be a plain dict (the real ``AgentResult.answer``
    is ``dict[str, Any] | None`` — ``dataclasses.asdict``'d for JSON-serializable agent
    state, per ``agents/nodes.py``) or an attribute-bearing object (test doubles)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def outcome_from_agent_result(result: object) -> AgentOutcome:
    """Map an S7 ``AgentResult`` onto the :class:`AgentOutcome` port.

    Pulls citations from ``result.answer`` (the M2 answer object, serialized to a plain
    dict by the agent graph). Missing or renamed fields fall back to safe empties so the
    harness records rather than crashes.
    """
    route = str(getattr(result, "route", ""))
    refused = bool(getattr(result, "refused", False))
    answer_text = str(getattr(result, "answer_text", "") or "")
    answer = getattr(result, "answer", None)
    raw_citations: Sequence[Any] = (
        _field(answer, "citations", ()) if answer is not None else ()  # type: ignore[assignment]
    )
    citations = [
        _Cite(
            doc_id=str(_field(c, "doc_id", "")),
            title=str(_field(c, "title", "")),
            section=_field(c, "section", None),  # type: ignore[arg-type]
            text=str(_field(c, "text", "") or _field(c, "snippet", "") or ""),
        )
        for c in raw_citations
    ]
    return _Outcome(route=route, refused=refused, answer_text=answer_text, citations=citations)


class AgentAdapter:
    """Wrap the concrete ``Agent`` facade so ``run`` yields an :class:`AgentOutcome`."""

    def __init__(self, agent: object) -> None:
        self._agent = agent

    def run(self, query: str, *, now_year: int) -> AgentOutcome:
        result = self._agent.run(query, now_year=now_year)  # type: ignore[attr-defined]
        return outcome_from_agent_result(result)


@dataclass(frozen=True, slots=True)
class _Hit:
    doc_id: str
    score: float
    section_title: str | None
    text: str


class RetrieverAdapter:
    """Wrap the concrete M2 retriever so ``retrieve`` yields :class:`RetrievedChunk`.

    The real ``retriever.retrieve(query)`` takes no per-call ``top_k`` — the depth is
    baked into the retrieval settings the retriever was built with (see
    :func:`research_navigator.eval.factory.build_real_builder`, which sets ``top_k`` to
    the harness's ``max_k``). We call it, read the ranked ``result.chunks`` (empty when
    the retriever refuses), and slice to the requested ``top_k`` defensively.
    """

    def __init__(self, retriever: object) -> None:
        self._retriever = retriever

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]:
        result = self._retriever.retrieve(query)  # type: ignore[attr-defined]
        chunks = getattr(result, "chunks", ()) or ()
        return [
            _Hit(
                doc_id=str(getattr(h, "doc_id", "")),
                score=float(getattr(h, "score", 0.0)),
                section_title=getattr(h, "section_title", None),
                text=str(getattr(h, "text", "") or ""),
            )
            for h in list(chunks)[:top_k]
        ]


class CallableModel:
    """Wrap any ``complete``-shaped callable as a :class:`SupportsComplete` judge model.

    Bridges ``generate.llm.LanguageModel`` regardless of whether its method is
    ``complete`` or something else — pass ``lambda p: llm.<method>(p)``.
    """

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def complete(self, prompt: str) -> str:
        return self._fn(prompt)


def judge_model_from_language_model(llm: object) -> SupportsComplete:
    """Best-effort adapter from a generate.llm ``LanguageModel`` to the judge port.

    The real ``LanguageModel.complete`` takes ``list[ChatMessage]``, not a bare string —
    this was a real bug (fixed): wrapping the raw prompt string straight through made
    every live judge call fail with ``'str' object has no attribute 'role'``
    (``OpenAIChatModel.complete`` iterates its argument expecting ``.role``/``.content``
    per element; iterating a string instead yields one-character strings). We wrap the
    prompt in a single user ``ChatMessage`` before calling it.
    """
    if hasattr(llm, "complete"):
        from research_navigator.generate.llm import ChatMessage

        def _call(p: str) -> str:
            try:
                return str(llm.complete([ChatMessage(role="user", content=p)]))
            except TypeError:
                # A .complete that genuinely takes a bare string (e.g. a test double).
                return str(llm.complete(p))

        return CallableModel(_call)
    if hasattr(llm, "generate"):
        return CallableModel(lambda p: str(llm.generate(p)))
    raise TypeError(
        "LanguageModel must expose .complete(prompt) or .generate(prompt) for the judge"
    )
