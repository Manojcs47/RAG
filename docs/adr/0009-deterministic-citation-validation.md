# ADR-0009: Deterministic citation validation + source-level dedup

Status: Accepted (Session 6)

## Context

The assignment's core requirement is that every substantive claim carries an
inline `[n]` citation, and that no citation is fabricated — "every cited
chunk traces back to a real retrieved chunk." An LLM can be *prompted* to
only cite real sources, but a prompt is not a guarantee: the model can still
emit `[7]` when only 3 sources were given, or cite the same document three
times as if it were three distinct sources. Trusting the model's own claim
of correctness would violate the "no silent failure" ground rule if that
claim were ever wrong.

## Decision

Two independent, code-enforced mechanisms in `generate/`:

1. **Anti-fabrication marker validation** (`markers.py::render_citations`):
   parse every `[n]` the model emitted; any index that doesn't map to a
   numbered source in `Answer.citations` is **dropped**, not trusted;
   surviving citations are **renumbered contiguously** by first appearance
   (so the rendered text never shows a gap like `[1]...[4]` when `[2]`/`[3]`
   were fabricated); the sentence a dropped marker was attached to is
   flagged in `uncited_sentences` as a diagnostic (not deleted — mangling
   the model's prose is worse than flagging it).
2. **Source-level dedup** (`sources.py::build_sources`): multiple retrieved
   chunks from the same `doc_id` collapse into **one** numbered `Source`,
   pointing at the section of that document's highest-scoring chunk, with
   context built from the top chunks for that doc (character-budgeted).
   Sources are numbered 1..N by best score, capped at `max_sources`.

Both run as plain, unit-tested Python — no second LLM call, no "ask the
model to check itself."

## Why

- **Determinism is stronger than a good prompt.** A prompt reduces the
  *rate* of fabrication; code that structurally cannot render an
  out-of-range citation eliminates it. The rubric's "no citation is
  fabricated" acceptance criterion becomes a property that's *true by
  construction* of `render_citations`, checked by
  `tests/test_generate.py`'s fabricated-marker-drop and noncontiguous-
  renumber cases, rather than an emergent property of prompt phrasing.
- **Dedup at the source level, not the chunk level**, matches how a reader
  actually wants citations: "see source 3" should mean one entry per
  document, not three near-duplicate entries because three chunks from the
  same paper happened to be retrieved. Picking the *highest-scoring* chunk's
  section for that entry keeps the citation pointing at the most relevant
  part of the document, not an arbitrary one.
- **Diagnostic, not corrective, handling of uncited sentences.** Silently
  deleting a sentence the model failed to cite would mangle the model's
  actual output and could remove a true, just-uncited claim. Flagging it
  (`Answer.uncited_sentences`) keeps the text intact while making the gap
  visible — to `make eval`'s LLM-judge faithfulness check (M4) and to any
  future stricter mode.

## Alternatives rejected

- **Trust the model's citations as-is**: directly risks the exact failure
  mode ("hallucinate citations... offer no way to verify a claim") the
  assignment's problem statement names as the thing generic chatbots get
  wrong.
- **A second LLM call to "verify" the first LLM's citations**: adds cost and
  latency for a check that a deterministic parser does instantly and
  exactly, with no risk of the verifier itself being wrong.
- **Silently dropping uncited sentences from the rendered answer**: trades a
  visible gap (a flagged, uncited claim) for an invisible one (a deleted
  claim the reader never knows was there) — worse for the "respect sources,
  never fabricate" ground rule, not better.

## Consequences

- `render_citations` is a pure function of (model text, numbered sources),
  fully covered by unit tests with no network/LLM dependency.
- Three independent refusal/safety layers now exist end-to-end: the
  retrieval cosine gate (ADR-0006), the LLM's own sentinel abstention
  (`INSUFFICIENT_CONTEXT`), and this citation-validation/empty-citation
  guard — any one of them failing degrades to a graceful decline, never a
  fabricated answer.
- A citation index gap in the model's raw output is invisible in the final
  render (renumbering hides it) but visible in `uncited_sentences` and in
  structured logs (`generator.py` emits `generate.dropped_fabricated_markers`
  with the dropped indices whenever any marker is discarded) — satisfying
  "no silent failure" without surfacing internal bookkeeping to the end
  user.
