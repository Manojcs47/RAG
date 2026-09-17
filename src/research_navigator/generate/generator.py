"""M2 generation orchestrator: grounded answer with validated inline citations,
same-doc dedup, and graceful low-confidence / insufficient-context refusal."""

from __future__ import annotations

from typing import Protocol

import structlog

from research_navigator.retrieve.models import RetrievalResult

from .citations import to_citation
from .llm import LanguageModel
from .markers import render_citations, uncited_sentences
from .models import Answer
from .prompt import build_messages
from .settings import GenerateSettings
from .sources import build_sources

log = structlog.get_logger(__name__)


class Retrieving(Protocol):
    def retrieve(self, query: str, *, now_year: int | None = None) -> RetrievalResult: ...


class Generator:
    def __init__(
        self,
        *,
        retriever: Retrieving,
        llm: LanguageModel,
        settings: GenerateSettings,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    def answer(self, query: str, *, now_year: int | None = None) -> Answer:
        s = self._settings
        res = self._retriever.retrieve(query, now_year=now_year)

        if res.refused:
            return self._refuse(query, res, reason="low_confidence")

        sources = build_sources(res.chunks, s)
        if not sources:
            return self._refuse(query, res, reason="no_sources")

        messages = build_messages(query, sources, s)
        raw = self._llm.complete(messages)

        if s.refusal_sentinel and s.refusal_sentinel in raw:
            return self._refuse(query, res, reason="model_insufficient")

        text, used, dropped = render_citations(raw, sources)
        if dropped:
            # no silent failure: the model referenced sources that don't exist
            log.warning(
                "generate.dropped_fabricated_markers",
                dropped=dropped,
                n_sources=len(sources),
            )

        if s.require_citations and not used:
            return self._refuse(query, res, reason="no_valid_citations")

        citations = tuple(
            to_citation(
                src,
                src.index,
                etal_threshold=s.authors_etal_threshold,
                labels=s.source_labels,
            )
            for src in used
        )
        answer = Answer(
            query=query,
            text=text,
            citations=citations,
            refused=False,
            reason=None,
            intent=res.analysis.intent.value,
            confidence=res.confidence,
            uncited_sentences=tuple(uncited_sentences(text)),
        )
        log.info(
            "generate.done",
            intent=answer.intent,
            n_citations=len(citations),
            n_uncited=len(answer.uncited_sentences),
            confidence=round(res.confidence, 4),
        )
        return answer

    def _refuse(self, query: str, res: RetrievalResult, *, reason: str) -> Answer:
        log.info("generate.refused", reason=reason, confidence=round(res.confidence, 4))
        return Answer(
            query=query,
            text=self._settings.refusal_message,
            citations=(),
            refused=True,
            reason=reason,
            intent=res.analysis.intent.value,
            confidence=res.confidence,
        )
