"""Agent configuration (nested under RN_ as ``agents`` -> env ``RN_AGENTS__*``).

Every tunable the router / tool / route nodes need lives here, per the project's
no-hardcoding rule. The router is a *hybrid*: deterministic keyword rules run first
(fast, explainable, unit-testable), and only genuinely ambiguous queries fall back
to the LLM classifier. Retrieval's cosine refusal gate (M2) remains the safety net,
so a mis-route still degrades to a graceful decline rather than a fabrication.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentSettings(BaseModel):
    """Tunables for routing + the six route nodes."""

    # --- router: hybrid rules -> LLM fallback -------------------------------
    use_llm_router: bool = Field(
        default=True,
        description="If a rule doesn't fire confidently, ask the LLM. If False, "
        "default to concept_explanation and let retrieval's refusal gate catch OOS.",
    )
    router_min_rule_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="A rule hit at/above this confidence short-circuits the LLM.",
    )
    rule_confidence_specific: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Confidence for a specific-cue rule hit."
    )
    rule_confidence_concept: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Confidence for the concept default hit."
    )
    llm_router_confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Assigned when the LLM decides the route."
    )

    # Deterministic cue lists (checked in priority order in router.py). Tunable so a
    # corpus/domain swap doesn't require code edits.
    find_papers_cues: list[str] = Field(
        default_factory=lambda: [
            "find papers",
            "list papers",
            "which papers",
            "show me papers",
            "recommend papers",
            "papers on",
            "papers about",
            "reading list",
            "any papers",
            "search for papers",
            "look for papers",
        ]
    )
    compare_cues: list[str] = Field(
        default_factory=lambda: [
            "compare",
            "comparison",
            " versus ",
            " vs ",
            " vs. ",
            "difference between",
            "differences between",
            "better than",
            "trade-off",
            "tradeoff",
            "trade off",
            "pros and cons",
            "advantages and disadvantages",
        ]
    )
    recent_cues: list[str] = Field(
        default_factory=lambda: [
            "recent",
            "latest",
            "newest",
            "state of the art",
            "sota",
            "cutting edge",
            "this year",
            "nowadays",
            "recent developments",
            "recent advances",
            "recent work",
            "new developments",
            "latest research",
            "what's new",
            "whats new",
        ]
    )
    deep_dive_cues: list[str] = Field(
        default_factory=lambda: [
            "this paper",
            "deep dive",
            "walk me through",
            "summarize the paper",
            "paper by",
            "arxiv",
            "et al",
        ]
    )
    concept_cues: list[str] = Field(
        default_factory=lambda: [
            "what is",
            "what are",
            "what's",
            "whats",
            "explain",
            "how does",
            "how do",
            "how can",
            "define",
            "definition of",
            "intuition behind",
            "why does",
            "why do",
            "meaning of",
            "describe",
        ]
    )
    # Verbs that, together with the plural word "papers", signal a browse/find intent.
    find_paper_browse_verbs: list[str] = Field(
        default_factory=lambda: ["find", "list", "show", "which", "recommend", "search", "look"]
    )

    # --- recent_developments date-math --------------------------------------
    recency_window_years: int = Field(
        default=2,
        ge=1,
        description="recent_developments focuses on year >= now - this window.",
    )
    recency_reference_year: int | None = Field(
        default=None,
        description="Pin 'now' for deterministic date-math (defaults to current year).",
    )

    # --- find_papers node ----------------------------------------------------
    find_papers_limit: int = Field(
        default=10, ge=1, description="Max papers listed by the find_papers route."
    )

    # --- out_of_scope node ---------------------------------------------------
    out_of_scope_message: str = (
        "That question falls outside this corpus of AI/ML research material, "
        "so I can't answer it from grounded sources."
    )
