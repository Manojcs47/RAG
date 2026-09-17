# ADR-0008: M4 Evaluation Harness

- **Status:** Accepted
- **Date:** 2026-09-17
- **Milestone:** M4 (Session 8)
- **Supersedes / relates to:** ADR-0005 (retrieval), ADR-0006 (generation & citations), ADR-0007 (LangGraph agent)

## Context

M4 requires an offline, reproducible evaluation of the whole Navigator: a labelled
golden set spanning all six routes, retrieval precision/recall@k against expected source
documents, citation-faithfulness scoring, refusal correctness on held-out out-of-scope
questions, latency (p50/p95) and token-cost per query, and a one-page report comparing at
least two configurations. `make eval` must emit both machine-readable JSON and a
human-readable Markdown report. The engineering bar from earlier sessions still holds:
`mypy --strict` on `src/`, `ruff` clean, structlog everywhere, no hardcoding, and no
silent failure (no `except: pass`).

The corpus is a *variable*, not a constant (ADR-0001): the harness must not bake in the
current 50 documents.

## Decision

### 1. Depend on structural **ports**, not concrete M2/M3 classes

`eval` never imports the concrete Qdrant retriever, the M2 generator, the agent graph, or
the OpenAI client at module load. It depends on four `typing.Protocol` "ports" in
`eval/ports.py` (`Retriever`, `Agent`, `SupportsComplete`, and the read-only
`RetrievedChunk` / `CitedSource` / `AgentOutcome` views). This mirrors the S7 `Generating`
protocol. Concrete classes are wrapped by thin adapters (`eval/adapters.py`) and wired only
in `eval/factory.py`.

**Consequence:** every module except `factory.py` imports, runs and unit-tests with zero
live dependencies. The test doubles (`Fake*`) satisfy the very same protocols, which keeps
the suite hermetic (24 tests, no network, no Qdrant). `factory.py` is the single seam that
needs `OPENAI_API_KEY` + a running Qdrant + FastEmbed egress; its concrete field/method
names (`use_sparse`, `build_retriever`, `AgentResult.answer`, …) are reconciled against the
real repo at wiring time and are the only thing a corpus/stack change can break.

### 2. Retrieval P/R@k is scored at the **document level**

Retrieval returns ranked *chunks*, but golden labels are *documents* (`doc_id`). Before
scoring we de-duplicate the ranked chunk list to its ordered unique `doc_id`s, then compute
precision/recall@k against the expected-source set. Precision's denominator is the number
of documents actually returned in the top-k (not k itself), so a small corpus or a query
with few relevant docs is not unfairly penalised. `k_values` defaults to `(3, 5, 8)` with a
`primary_k` of 5 for the headline number; all are settings.

### 3. `tiktoken` is **optional**; cost is an honest estimate

`tiktoken` gives exact OpenAI token counts but lazily downloads its BPE vocabulary on first
use, which fails in an air-gapped or sandboxed run. The token counter therefore treats
`tiktoken` as optional: it is used when it loads and, on failure, falls back **once**
(logged, never silently) to a `len/4` heuristic. The report states plainly whether counts
are `exact (tiktoken)` or `approximate (heuristic)`. Because the harness does not observe
the exact generation prompt, token/cost figures are labelled indicative, not
billing-accurate. Prices are settings (`price_per_1k_input_usd` / `_output_usd`) defaulting
to published gpt-4o-mini rates, marked "verify" here because vendor prices drift.

### 4. Citation faithfulness uses an **LLM judge with an explicit written rubric**

The rubric lives verbatim in `eval/judge.py` as `CITATION_JUDGE_RUBRIC` and is reproduced
below so the grading criteria are transparent. The judge scores two axes and returns a
single JSON object:

- **Grounding** — does every *cited* claim follow from the text of the source it cites?
- **Attribution** — is every *substantive* claim actually cited?

`score = 0.5 * grounding + 0.5 * attribution`, where `grounding = supported / cited` and
`attribution = cited / (cited + uncited)`. An answer is reported *faithful* iff the model's
boolean holds **and** `score >= judge_min_score` (default 0.8). Parsing is defensive: a
fenced ```json block is tolerated, a bare object is extracted as a fallback, and an
unparseable or erroring reply is recorded as `parse_ok=False` and counted as non-faithful —
surfaced and logged, never dropped. An answer that makes claims but cites nothing is scored
maximally unfaithful without calling the model.

### 5. Compare **hybrid vs dense-only** retrieval as the two required configurations

`default_configs()` yields `hybrid` (dense + sparse) and `dense_only`. A config is just a
name plus an overrides dict (`retrieval_mode`), applied via pydantic `model_copy(update=…)`
onto a fresh retriever+agent per config so overrides actually take effect. Adding a third
config (e.g. metadata-filtered, or a different `primary_k`) is a one-line change and needs
no new code. The runner asserts the golden set covers at least three routes before running,
so an accidentally truncated set fails fast rather than reporting a misleading score.

## Alternatives considered

- **Import the concrete stack directly.** Rejected: couples `eval` to Qdrant/FastEmbed/OpenAI,
  makes the harness un-runnable offline and the suite non-hermetic.
- **Chunk-level retrieval scoring.** Rejected: labels are documents; chunk-level numbers
  would depend on the chunking config and mislead across a corpus swap.
- **Hard-depend on `tiktoken`.** Rejected: breaks air-gapped/sandboxed runs; exactness is not
  worth a hard failure when the goal is *comparing* configurations.
- **String-overlap / ROUGE faithfulness.** Rejected: cannot tell a grounded citation from a
  fluent-but-unsupported one; the rubric-driven judge targets exactly that.

## Consequences

- The whole harness is offline-first and hermetically testable; only `factory.py` touches the
  live stack. Coverage of the `eval` package is 91% (factory.py is the uncovered seam by
  design).
- `make eval` runs the live sweep and writes `eval/report.json` + `eval/report.md`;
  `make eval-dry-run` runs the identical flow on in-repo fakes with no network, which is what
  CI and this ADR's numbers use.
- Cost/faithfulness numbers are honest proxies with their limitations printed in the report,
  not silently presented as ground truth.

## The rubric (verbatim)

```
You are a strict evaluator of citation faithfulness for a retrieval-augmented answer.
You are given a QUESTION, an ANSWER whose sentences carry inline markers like [1], [2],
and the SOURCES those markers refer to (each with its exact retrieved text).

Judge the answer on two axes:
1. GROUNDING: for every claim that carries a citation marker, does that claim actually
   follow from the text of the cited source? A claim is UNSUPPORTED if the cited source
   does not state or clearly imply it, or if it cites the wrong source.
2. ATTRIBUTION: does every substantive factual claim carry a citation marker? A
   substantive claim stated with no marker is an UNCITED claim. (Generic transitions,
   restatements of the question, and hedging are not substantive.)

Do not use outside knowledge; judge only against the provided SOURCES.

Reply with a SINGLE JSON object and nothing else, with exactly these keys:
{
  "supported_claims": <int>,      // cited claims that follow from their source
  "unsupported_claims": <int>,    // cited claims that do NOT follow from their source
  "uncited_claims": <int>,        // substantive claims with no citation marker
  "faithful": <bool>,             // true iff unsupported_claims and uncited_claims are 0
  "rationale": "<one or two sentences>"
}
```
