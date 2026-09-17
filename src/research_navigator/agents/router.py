"""The router: query -> one of the six routes.

Hybrid, mirroring the S5 filter-inference decision (rules-first, LLM augments):

1. **Deterministic rules** (``classify_by_rules``) — high-precision keyword cues in a
   fixed priority order. Fully unit-testable, no network, and every decision carries a
   human-readable ``reason`` (which cue fired) for logs / the visible-route bonus.
2. **LLM fallback** — only when no rule fires at/above ``router_min_rule_confidence``.
   The LLM is reached through the same ``LanguageModel`` protocol the generator uses,
   so tests inject a fake and OSS backends swap in unchanged.

If the LLM is disabled and no rule fires, we default to ``concept_explanation``: the
retrieval refusal gate (M2) then declines gracefully on genuinely off-corpus queries,
so the honest failure mode is "refuse", never "fabricate".
"""

from __future__ import annotations

import structlog

from research_navigator.generate.llm import ChatMessage, LanguageModel

from .settings import AgentSettings
from .state import Route

log = structlog.get_logger(__name__)

_ROUTE_VALUES = tuple(r.value for r in Route)


def _any(hay: str, cues: list[str]) -> str | None:
    for c in cues:
        if c in hay:
            return c.strip()
    return None


def classify_by_rules(query: str, settings: AgentSettings) -> tuple[str | None, str, float]:
    """Return (route | None, reason, confidence). ``None`` route => defer to the LLM.

    Priority order matters: the most *specific* intents are checked first so that,
    e.g., "find recent papers on X" resolves to find_papers (which then applies its
    own recency filter) rather than to recent_developments.
    """
    q = f" {query.lower().strip()} "
    spec = settings.rule_confidence_specific

    # 1) find_papers — explicit browse phrasing, or a browse verb + plural "papers".
    hit = _any(q, settings.find_papers_cues)
    if hit is None and "papers" in q and _any(q, settings.find_paper_browse_verbs):
        hit = "papers+browse-verb"
    if hit is not None:
        return Route.FIND_PAPERS.value, f"rule:find_papers({hit})", spec

    # 2) compare_approaches — comparison connectives.
    hit = _any(q, settings.compare_cues)
    if hit is not None:
        return Route.COMPARE_APPROACHES.value, f"rule:compare({hit})", spec

    # 3) recent_developments — recency cues.
    hit = _any(q, settings.recent_cues)
    if hit is not None:
        return Route.RECENT_DEVELOPMENTS.value, f"rule:recent({hit})", spec

    # 4) paper_deep_dive — explicit deep-dive cues, or a *singular* "paper" reference.
    hit = _any(q, settings.deep_dive_cues)
    if hit is None and " paper " in q and " papers " not in q:
        hit = "paper(singular)"
    if hit is not None:
        return Route.PAPER_DEEP_DIVE.value, f"rule:deep_dive({hit})", spec

    # 5) concept_explanation — definitional/explanatory cues (moderate confidence).
    hit = _any(q, settings.concept_cues)
    if hit is not None:
        return (
            Route.CONCEPT_EXPLANATION.value,
            f"rule:concept({hit})",
            settings.rule_confidence_concept,
        )

    # No confident rule.
    return None, "rule:none", 0.0


def _llm_prompt(query: str) -> list[ChatMessage]:
    routes = ", ".join(_ROUTE_VALUES)
    system = (
        "You are a router for a RAG assistant over a fixed corpus of AI/ML research "
        "(papers, course chapters, technical blog posts). Classify the user's question "
        f"into EXACTLY ONE of these routes: {routes}. "
        "Use out_of_scope when the question is not about AI/ML/deep-learning topics that "
        "such a corpus would cover. Reply with ONLY the route name, nothing else."
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=query)]


def _parse_route(reply: str) -> str | None:
    low = reply.strip().lower()
    # exact first, then substring (defensive against a chatty model)
    if low in _ROUTE_VALUES:
        return low
    for r in _ROUTE_VALUES:
        if r in low:
            return r
    return None


class Router:
    """Holds the settings and (optionally) an LLM for the fallback path."""

    def __init__(self, *, settings: AgentSettings, llm: LanguageModel | None) -> None:
        self._settings = settings
        self._llm = llm

    def route(self, query: str) -> tuple[str, str, float]:
        s = self._settings
        route, reason, conf = classify_by_rules(query, s)
        if route is not None and conf >= s.router_min_rule_confidence:
            log.info("router.rule", query=query, route=route, reason=reason, confidence=conf)
            return route, reason, conf

        if s.use_llm_router and self._llm is not None:
            reply = self._llm.complete(_llm_prompt(query))
            parsed = _parse_route(reply)
            if parsed is not None:
                log.info("router.llm", query=query, route=parsed, raw=reply.strip())
                return parsed, "llm", s.llm_router_confidence
            # No silent failure: log the unparseable reply, then fall through.
            log.warning("router.llm_unparsed", query=query, raw=reply.strip())

        # Deterministic default: concept_explanation (refusal gate handles true OOS).
        fallback = route or Route.CONCEPT_EXPLANATION.value
        log.info("router.fallback", query=query, route=fallback)
        return fallback, "fallback:concept_default", s.rule_confidence_concept
