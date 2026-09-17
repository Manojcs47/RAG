# ADR-0007: M3 agent — hybrid router, deterministic routes, serializable state

Status: Accepted (Session 7)

## Context

M3 requires a LangGraph state machine: a router plus six route nodes
(`concept_explanation`, `paper_deep_dive`, `compare_approaches`, `recent_developments`,
`find_papers`, `out_of_scope`), at least one tool call, serializable state, and a rendered
graph. It must reuse the M2 Retriever + Generator inside the route nodes and preserve the
project's anti-fabrication and no-silent-failure guarantees.

## Decisions

### 1. Hybrid router: deterministic rules first, LLM fallback
`Router.route()` runs keyword/pattern rules first (priority order
`find_papers > compare_approaches > recent_developments > paper_deep_dive >
concept_explanation`); only genuinely ambiguous queries fall through to an LLM classifier,
and if that is disabled or unparseable, the router defaults to `concept_explanation`.

- **Why:** mirrors the S5 rules-first philosophy — cheap, deterministic, testable, and
  offline on the common path. The LLM is a fallback, not the hot path.
- **Safety net:** a mis-route still hits M2's retrieval cosine gate + LLM sentinel +
  empty-citation guard, so the worst case is a *graceful refusal*, never fabrication.
- **Alternative rejected:** pure-rule `out_of_scope` detection. Off-domain queries with no
  lexical cues ("how do I bake sourdough?") can't be caught by keywords without brittle
  denylists, so OOS classification for cue-free queries is delegated to the LLM fallback.

### 2. `find_papers` is deterministic — no LLM
The `find_papers` node answers straight from the corpus manifest via the `CorpusIndex`
tool (tag/field query, ordered, limited). The generator is never called.

- **Why:** "which papers do you have on X" is a metadata lookup, not a generation task.
  Answering it from the manifest gives **zero hallucination** of paper titles/authors/years
  by construction. An empty match refuses rather than inventing.

### 3. Serializable state = `TypedDict(total=False)` of JSON primitives
`AgentState` holds only JSON-friendly values; `total=False` lets each node return a partial
update that LangGraph merges. Tool invocations are recorded in `state["tool_calls"]` as
plain dicts.

- **Why:** satisfies the "serializable/checkpointable state" requirement, keeps the state
  inspectable, and is exactly what the B1 streaming-UI bonus (visible route + trace) will
  read. A `test_agents.py` case round-trips the full state through `json.dumps`.

### 4. `recent_developments` carries the required tool call as pure date-math
`recency_cutoff(now_year, window)` computes the recency window with no I/O and no model
call; the node also queries the `CorpusIndex` for the actual latest-year window and emits a
manifest-grounded preamble before recency-biased RAG. Both calls are logged to
`tool_calls`.

- **Why:** gives a genuine, unit-testable tool call (date-math + corpus lookup) whose
  output is grounded in the manifest, not asserted by the model.

### 5. Scoped `call-overload` mypy suppression on the wiring module only
`StateGraph.add_node` is an 8-way overload with a bounded `NodeInputT` TypeVar that mypy
cannot infer when a node is passed as a factory-produced `Callable`. We suppress
`call-overload` for `research_navigator.agents.graph` **only**.

- **Why acceptable:** `AgentState` is a real TypedDict, `compile()` validates the whole
  graph at build time (mis-wire fails loudly), and the tests invoke all six routes. This
  mirrors the existing thin `cli.py` (Typer) carve-out. The related `type-arg` friction is
  fixed *in code* via one parametrized `CompiledAgentGraph` alias, not by suppression.

## Consequences

- Routing is deterministic and offline for the vast majority of queries; LLM cost is
  incurred only on ambiguous routing and on the generative routes.
- Every route degrades to a grounded refusal instead of fabricating.
- The graph renders offline to `docs/agent_graph.mmd` via `make graph`.
- One narrowly-scoped, documented type-check suppression; the rest of the package is clean
  under `mypy --strict`.
