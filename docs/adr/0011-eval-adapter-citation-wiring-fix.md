# ADR-0011: Fix the eval-harness ↔ real-answer citation wiring (two bugs)

Status: Accepted (Session 9)

## Context

Switching `make eval` to a local model (ADR-0010) to escape a cloud quota
wall let the harness run long enough, on real (not fake) LLM calls, to expose
two bugs in `eval/adapters.py` that had silently made the M4
citation-faithfulness judge inert since it was written in Session 8 — it had
never once scored a real answer, on any backend, including the earlier
cloud runs that appeared to "work" (they just returned 0 questions with
citations, which looks identical in the report to "everything refused").

**Bug 1 — `getattr` on a dict never fails, it just returns the default.**
`AgentResult.answer` is `dict[str, Any] | None`: `agents/nodes.py` calls
`dataclasses.asdict(ans)` on the M2 `Answer` dataclass so the LangGraph state
stays JSON-serializable (ADR-0007). `outcome_from_agent_result` read
`getattr(answer, "citations", ())`, which is correct for an *object* but
silently wrong for a *dict* — `getattr(some_dict, "citations", ())` is
always `()`, because a dict has no such attribute, and `getattr`'s whole
point is to not raise on a missing one. `outcome.citations` was therefore
always empty, on every question, regardless of whether the real answer had
citations.

**Bug 2 — the judge's model call didn't match the real interface.**
`judge_model_from_language_model` wrapped the real `LanguageModel` as
`lambda p: llm.complete(p)`, but `LanguageModel.complete` takes
`list[ChatMessage]`, not a bare string. Once bug 1 was fixed and the judge
actually started being invoked, every call crashed with
`'str' object has no attribute 'role'` (`OpenAIChatModel.complete` iterates
its argument expecting `.role`/`.content` per element).

**Why the existing tests missed both**: `test_outcome_adapter_maps_duck_typed_result`
and `test_judge_model_adapter_variants` used hand-built fakes shaped like
plausible *objects*, not like what the real M3/generation code actually
produces (a dict, and a `list[ChatMessage]`-taking method, respectively). A
hermetic test only catches a wiring bug if its fake matches the real shape —
matching *a* shape isn't the same as matching *the* shape.

## Decision

1. `adapters.py::_field(obj, key, default)` — a small helper that reads `key`
   via `.get()` if `obj` is a dict, else via `getattr()`. Used for `answer`
   and every citation dict/object uniformly, so the code is correct for both
   the real (dict) shape and any future object-shaped test double.
2. `judge_model_from_language_model` now wraps the prompt in
   `[ChatMessage(role="user", content=p)]` before calling `.complete()`,
   with a fallback to the bare-string call (caught via `TypeError`) so
   existing simple test doubles keep working unchanged.
3. `RetrievedChunk` (port) and `_Hit` (adapter) gained a `.text` field, and
   `EvalRunner._score_retrieval` now also returns a `{doc_id: text}` map;
   `_with_grounding_text` backfills each citation's `.text` from the
   matching retrieved chunk when the citation itself carries none (`Citation`
   is a rendering-only model — it never carried source text, by design, per
   `CitedSource`'s own port docstring). Without this, the judge would
   structurally run but see an empty SOURCES block and have nothing to check
   grounding against.
4. Added regression tests using the *real* shapes: a dict-shaped `answer`
   with dict-shaped citations (matching exactly what `dataclasses.asdict`
   produces), and a `LanguageModel`-shaped fake whose `.complete` requires
   `list[ChatMessage]` and records what it received.

## Why

- **Fix at the boundary, not by changing the real types.** `AgentResult`
  staying a serializable dict (ADR-0007) and `LanguageModel.complete` taking
  structured messages (not a bare string) are both correct, deliberate
  choices elsewhere in the codebase. The adapter's job — per its own module
  docstring — is to isolate `eval` from exactly this kind of concrete-shape
  detail; the fix belongs entirely in `eval/adapters.py`/`eval/runner.py`,
  and nowhere else changed.
- **Grounding text via backfill, not by extending `Citation`.** Adding a
  `text` field to the display-only `Citation` model would conflate two
  different concerns (what's shown to the end user vs. what the eval judge
  needs to check grounding) for the benefit of eval code alone. Pulling it
  from the retrieval hits the runner already fetches for P/R@k scoring reuses
  data that exists anyway, at the cost of one small helper.
- **Regression tests must mirror the real shape, deliberately.** The new
  tests are written to fail against the *old* code specifically because they
  use the dict/`ChatMessage` shapes the real system produces — closing the
  exact gap that let both bugs ship silently in Session 8.

## Consequences

- The M4 citation-faithfulness judge has, for the first time, scored real
  (non-fake) generated answers — see ADR-0010's "Outcome" section and
  `docs/OBSERVATIONS.md` §6 for the numbers.
- `eval/adapters.py`'s and `eval/runner.py`'s docstrings were updated in
  place to state the real shapes explicitly, so a future contributor changing
  `AgentResult.answer`'s serialization or `LanguageModel.complete`'s
  signature has a specific, named contract to check against instead of a
  generic "adjust as needed" note.
- This is a caution against reading "0 errors" as "correct": the harness ran
  clean, exit 0, on every prior invocation *while a required metric was
  structurally inert the whole time*. Honest evaluation (a project ground
  rule) means checking that a metric is actually being computed, not just
  that the run didn't crash.
