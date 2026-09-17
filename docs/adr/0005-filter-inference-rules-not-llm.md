# ADR-0005: Query filter inference = deterministic rules, not an LLM call

Status: Accepted (Session 5)

## Context

M2 needs to turn a natural-language query ("recent work on speculative
decoding") into Qdrant metadata filters (`year_gte=2024`, `tags ⊇
["speculative_decoding"]`). The assignment's own example is pattern-based.
Two approaches were on the table: an LLM classification call per query, or a
deterministic rule engine over a vocabulary derived from the corpus.

## Decision

`retrieve/query_understanding.py::analyze()` is a **pure function**, no LLM,
no I/O:
- Recency: an explicit year, a "last N years/months" phrase, or a recency cue
  word → `year_gte`.
- Tags: whole-token/phrase matching against a vocabulary built from the
  manifest's own `tags` field (`vocab.py::QueryCatalog.from_manifest_rows`),
  not a fixed external taxonomy.
- Content type: configurable trigger phrases, intersected with content types
  that actually exist in the loaded corpus.
- Foundational: a cue phrase → `is_foundational=True`.

Every fired rule records a human-readable `reason` string
(`QueryAnalysis.reasons`) for logging/debugging/tests.

## Why

- **Corpus-swap safety.** The assignment explicitly frames the corpus as "a
  variable, not a constant." A vocabulary derived from the *current*
  manifest means filters can only ever reference tags/content-types that
  exist right now — swap the corpus and the vocabulary updates itself; no
  code change, no risk of matching against a stale taxonomy.
- **Determinism + testability.** `analyze()` is unit-tested exhaustively
  (recency/explicit-year/last-N/tags/content-type/foundational/compare/plain/
  disabled) with zero network calls and zero flakiness. An LLM classifier
  would need either a fixed seed and cached responses (fragile) or accepted
  nondeterminism in test assertions.
- **Zero marginal cost and zero added latency** on the hot path — every
  `search`/`answer`/`ask` call performs filter inference for free, vs. an
  LLM round-trip purely to extract structured filters that a handful of
  deterministic rules already extract correctly for the phrasing this
  corpus's queries actually use.
- **Consistent with the M3 router's own hybrid philosophy** (ADR-0007):
  rules first, LLM only where rules genuinely can't decide. This ADR is the
  earlier instance of that same tradeoff, at the retrieval layer.

## Alternatives rejected

- **LLM-based filter extraction**: more flexible on genuinely novel phrasing,
  but non-deterministic, adds latency + cost to every single query (not just
  ambiguous ones), and is harder to unit-test without either mocking away
  the exact behavior being tested or accepting flaky assertions.
- **A fixed, hardcoded tag taxonomy**: violates "treat the corpus as a
  variable" directly — would silently stop matching (or falsely match) once
  the corpus's actual tag vocabulary diverges from the hardcoded list.

## Consequences

- Filter inference is provably safe to call on the hot path of every query.
- Recall on filter inference is bounded by the rule set's coverage of
  real phrasing — a query using vocabulary the corpus doesn't tag for will
  simply infer no filter (not a wrong one), and unfiltered hybrid search
  still runs.
- The `analyze()` signature is intentionally the extension point: a future
  LLM pass could *augment* (not replace) these rules without changing the
  function's contract, if broader recall is ever needed.
