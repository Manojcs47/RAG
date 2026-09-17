"""Ports: the narrow, structural interfaces the eval harness depends on.

The evaluation harness must never import the concrete M2/M3 classes directly — that
would couple ``eval`` to Qdrant, FastEmbed and the OpenAI client and make the harness
un-runnable offline and un-testable without the live stack. Instead we depend on
:class:`typing.Protocol` "ports" (structural typing), mirroring the S7 ``Generating``
protocol that structurally matched ``Generator.answer``.

Anything with the right shape satisfies these protocols — no inheritance, no import. The
concrete adapters that wrap the real ``Retriever``/``Agent``/``LanguageModel`` live in
:mod:`research_navigator.eval.adapters`; the ``Fake*`` doubles in the tests satisfy the
very same protocols, which is what keeps the whole harness hermetic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class RetrievedChunk(Protocol):
    """A single chunk returned by the retriever (structural view of an M2 hit)."""

    @property
    def doc_id(self) -> str: ...

    @property
    def score(self) -> float: ...

    @property
    def section_title(self) -> str | None: ...

    @property
    def text(self) -> str: ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieval port. ``retrieve`` returns ranked chunks for a raw query.

    Metadata-filter inference and hybrid fusion happen *inside* the concrete
    retriever (M2); the eval harness only observes the ranked ``doc_id``s so it can
    score precision/recall@k at the document level against expected sources.
    """

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]: ...


@runtime_checkable
class CitedSource(Protocol):
    """A citation as it appears in an M2 answer, plus the source text it rests on.

    ``text`` is the retrieved chunk text the citation was grounded in; it may be an
    empty string if the concrete citation block does not retain it, in which case the
    faithfulness judge degrades to a structural check (see :mod:`.judge`).
    """

    @property
    def doc_id(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def section(self) -> str | None: ...

    @property
    def text(self) -> str: ...


@runtime_checkable
class AgentOutcome(Protocol):
    """Structural view of the S7 ``AgentResult`` (route + cited/refusing answer)."""

    @property
    def route(self) -> str: ...

    @property
    def refused(self) -> bool: ...

    @property
    def answer_text(self) -> str: ...

    @property
    def citations(self) -> Sequence[CitedSource]: ...


@runtime_checkable
class Agent(Protocol):
    """Agent port: the S7 ``Agent.run`` facade (route + reuse of the M2 stack)."""

    def run(self, query: str, *, now_year: int) -> AgentOutcome: ...


@runtime_checkable
class SupportsComplete(Protocol):
    """Minimal LLM port used by the citation-faithfulness judge.

    This is intentionally the narrowest useful interface. The real
    ``research_navigator.generate.llm.LanguageModel`` is adapted onto it with a
    one-line wrapper in :mod:`.adapters` (its structural shape is confirmed at S8).
    """

    def complete(self, prompt: str) -> str: ...
