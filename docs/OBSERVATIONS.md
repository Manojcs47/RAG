# Corpus & Pipeline Observations

Honest, numbers-backed notes on the actual corpus and pipeline behavior —
per the ground rule "report what doesn't work." Numbers below were pulled
directly from `data/parsed/*.json` (50 docs) and a live `make eval` run;
regenerate with `make prepare` + the snippets inline below if the corpus
changes.

## 1. Corpus composition (as parsed)

| content_type | docs | docs with parse warnings |
|---|---|---|
| `arxiv_paper` | 30 | 28 |
| `course_chapter` | 12 | 8 |
| `survey_blog` | 5 | 5 |
| `lab_blog_post` | 3 | 1 |

- 0 of 50 documents failed to parse or collapsed to a single section — the
  heading detector always found *some* structure.
- 14 of 50 documents contain at least one code block (mostly `course_chapter`).
- All 30 arXiv papers have a detected `references` section (retained for
  citation lookup, excluded from retrieval). The other 20 documents
  (course chapters, surveys, blog posts) legitimately have none — they don't
  carry a bibliography.

## 2. Parse warnings, by category

- **Images skipped** (arXiv, figure/diagram content): the dominant warning
  category, ranging from 1 to 83 skipped images per paper. Figures are never
  retrievable text today — a diagram-heavy paper (e.g. architecture figures)
  loses that content entirely. Captions *are* kept (tagged as `caption`
  blocks) when they appear as text near the figure.
- **Navigation/link-only blocks dropped** (`course_chapter`, `survey_blog`):
  3–8 blocks per document — sidebar links, "next chapter" navigation, share
  buttons carried over from the original HTML→Markdown conversion
  (`corpus/README.md` §2 flags this as expected noise).

## 3. Known data-quality issue: heading false positives on arXiv PDFs

**This is the most significant limitation found in this pass.** The PDF
heading heuristic (`parse/pdf.py::_is_heading`: font size ≥1.12× body text,
≤12 words, no trailing period) also fires on:
- arXiv running headers, e.g. `arXiv:1706.03762v7  [cs.CL]  2 Aug 2023`
- numeric table/figure data rows rendered in a larger/bold font, e.g.
  `10 20 30 40 50 K Retrieved Docs` or a row of hyperparameter values

Measured across the corpus: **481 of 1,808 arXiv sections (≈27%) have a
garbled, non-semantic `section_title`**, and all 30 arXiv papers are affected
to some degree — including foundational papers (Attention Is All You Need,
BERT, the RAG paper, GPT-3, LoRA). This was directly observed in a live
`search` query, where a real hit's citation section printed as
`§0.5 23.16 22.62 22.52 21.93 21.14 18.55 0.6 21.65 ...` instead of a real
heading.

**Impact**: the retrievable *text* of the affected chunk is unaffected (the
heading only mis-labels the chunk's `section_title` payload field) — but the
citation rendered to the user (`generate/citations.py`) shows this garbled
label as the "section," which is a real, user-visible quality gap for the
"section" field the assignment's citation spec asks for. Not fixed in this
pass; the honest options are (a) filter candidate headings by a stricter
character-class check (reject lines that are mostly digits/whitespace), or
(b) fall back to the nearest preceding *good* heading when a candidate looks
like a running-header/data-row pattern.

## 4. Chunking implications (confirmed, not hypothesized)

- 4,777 total chunks across 50 documents (deterministic — confirmed by
  regenerating `data/chunks/` from scratch via `make prepare` and diffing
  against the live Qdrant collection's `total_points`, which matched
  exactly).
- Global `chunk_index` is assigned per-document in original order; an edit
  early in a long document shifts every later chunk's index, and therefore
  its `chunk_uid`/point id — a full-document re-embed, not a targeted one.
  Acceptable for a static corpus; would need content-local (not positional)
  chunk identity for a frequently-edited corpus.
- Oversized code blocks (rare — 14/50 docs have code at all) get their own
  chunk and are truncated at embed time by the tokenizer's max length; this
  is logged as a warning, never silently dropped.
- The sentence splitter used for prose overlap trimming is regex-heuristic,
  not a proper sentence-boundary model — acceptable for English prose,
  untested on the corpus's occasional inline LaTeX/code fragments.

## 5. Retrieval: filter relaxation is all-or-nothing

If a query's inferred filters (e.g. `year_gte=2024` AND `tags=[...]`) return
zero hits, the retriever currently retries **fully unfiltered** rather than
relaxing constraints incrementally (e.g. drop recency before dropping topic
tags). A query that's simultaneously "recent" and "foundational" — a
contradictory combination the corpus may not satisfy — falls back to
unfiltered search rather than partial relaxation. Logged loudly as a warning
either way; never a silent empty result.

## 6. Live evaluation: quota, then two real bugs, now fixed

A live `make eval` run against the originally-configured cloud backend
(Gemini 3.6 Flash, **free tier, 20 requests/day**) exhausted its quota
partway through both sweeps: 31/40 (hybrid) and 35/40 (dense_only) questions
errored with `RESOURCE_EXHAUSTED`. The harness behaved correctly under that
load — every failure was logged with the question id and error, the run
exited 0, and a report was still written — but its metrics were computed
over the ~5–9 surviving questions per config, not representative.
Switched to a local Ollama model (`llama3.2:3b`, no-root install — see
ADR-0010) to remove the quota ceiling entirely.

**That switch surfaced two real, previously-hidden bugs in `eval/adapters.py`
that no prior run had ever exercised**, both now fixed with regression tests
(`tests/test_eval.py`):

1. `AgentResult.answer` is a plain `dict` (`dataclasses.asdict`'d by
   `agents/nodes.py` for JSON-serializable state), but
   `outcome_from_agent_result` read citations off it with `getattr()` —
   which never raises on a dict, it just silently returns the default every
   time. `outcome.citations` was **always empty**, on every question, on
   every past eval run (including the Gemini one above) — the M4
   citation-faithfulness judge had never evaluated a single real answer,
   and the report's "not evaluated (judge disabled or no cited answers)"
   line was masking that, not describing an intentional skip.
2. After fixing (1), the judge started being called but crashed every time
   with `'str' object has no attribute 'role'`: the adapter passed the raw
   prompt string straight into the real `LanguageModel.complete()`, which
   actually takes `list[ChatMessage]`. Iterating a string yields characters,
   not messages.

Neither bug was caught by the existing test suite because the fakes used in
`test_judge_model_adapter_variants`/`test_outcome_adapter_maps_duck_typed_result`
were shaped differently from what the real M3/generation code actually
produces — a lesson in itself: a hermetic test only catches a wiring bug if
its fakes match the real shape, not just *a* plausible shape.

**With both fixed**, a real 18-question run (3 per route — the harness's own
minimum-coverage floor; run against reduced retrieval/generation context
sizes — `top_k`/`max_sources`/`source_char_budget`/`max_tokens` all turned
down — purely to keep CPU-only 3B-model latency tractable, ~20–30s/call
instead of ~90s) completed with **zero errors** and real citation-faithfulness
numbers: **0.62 (hybrid) / 0.57 (dense_only)** mean judge score, over
13–14 judged answers per config (4 judge-JSON parse failures per config —
the 3B model doesn't reliably follow the "reply with only JSON" instruction
under a tight `max_tokens`). This is now `eval/report.md`'s shipped content.
A full 40-question sweep at these settings would take roughly 2x as long
(~1–2 hours on this CPU-only machine) and was not run in this pass; the
18-question run already exercises every route and the full judge pipeline
end-to-end, which was the open item this pass closed.

Citation faithfulness scores here reflect a 3B CPU-inference model's ability
to produce citeable, judgeable answers, not a frontier model's — read them
as "the pipeline correctly computes and gates faithfulness," not as a
benchmark ceiling.

## 7. Documentation/code drift: refusal threshold

`PROJECT_LOG.md` states the refusal threshold as `0.35` in every session from
S1 onward, but the actual default in `retrieve/settings.py::RetrieveSettings`
is **`0.25`**. Whether this was a deliberate retuning that never made it into
the log, or drift, wasn't determinable from history — flagging it here so it
isn't silently trusted from stale prose. The code (0.25) is authoritative;
ADR-0006 documents the current value.

## 8. Manifest caveats (resolved)

Two arXiv IDs flagged at corpus curation time as "verify before use"
(`corpus/README.md` §5) were checked against the live manifest and resolve
correctly:
- `arxiv-2408.00118` → "Gemma 2: Improving Open Language Models at a
  Practical Size" ✅
- `arxiv-2501.12948` → "DeepSeek-R1: Incentivizing Reasoning Capability in
  LLMs via Reinforcement Learning" ✅
