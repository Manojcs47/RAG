# AI Research Navigator — Evaluation Report

_Generated 2026-09-17T11:30:31+00:00 · golden set: 40 questions · primary k = 5 · token counts approximate (heuristic)._

## Configuration comparison

| Config | Routing acc | P@5 | R@5 | F1@5 | Faithfulness | Refusal acc | p50 ms | p95 ms | $/query | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.00001 | 0 |
| dense_only | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.00001 | 0 |

## hybrid

- **Questions:** 40 (0 errored)
- **Retrieval (mean over retrieval subset):** P@3=1.00/R@3=0.96, P@5=1.00/R@5=0.99, P@8=1.00/R@8=1.00
- **Routing accuracy:** 1.00 overall; per route: compare_approaches=1.00, concept_explanation=1.00, find_papers=1.00, out_of_scope=1.00, paper_deep_dive=1.00, recent_developments=1.00
- **Refusal:** acc=1.00, recall=1.00 (refuses when it should), precision=1.00 (does not over-refuse), n=40
- **Citation faithfulness (LLM judge):** rate=1.00, mean score=1.00 over n=32 answers (0 judge parse failures)
- **Latency:** mean=0ms, p50=0ms, p95=0ms, max=0ms
- **Cost:** total=$0.0002, mean=$0.00001/query, tokens in/out=858/192

## dense_only

- **Questions:** 40 (0 errored)
- **Retrieval (mean over retrieval subset):** P@3=1.00/R@3=0.96, P@5=1.00/R@5=0.99, P@8=1.00/R@8=1.00
- **Routing accuracy:** 1.00 overall; per route: compare_approaches=1.00, concept_explanation=1.00, find_papers=1.00, out_of_scope=1.00, paper_deep_dive=1.00, recent_developments=1.00
- **Refusal:** acc=1.00, recall=1.00 (refuses when it should), precision=1.00 (does not over-refuse), n=40
- **Citation faithfulness (LLM judge):** rate=1.00, mean score=1.00 over n=32 answers (0 judge parse failures)
- **Latency:** mean=0ms, p50=0ms, p95=0ms, max=0ms
- **Cost:** total=$0.0002, mean=$0.00001/query, tokens in/out=858/192

## Limitations & honest notes

- Retrieval P/R is scored at the document level against a hand-labelled golden set; expected-source labels are best-effort and may under-count valid sources.
- Token/cost figures are estimated from the query, cited source text and the answer (the exact generation prompt is not observed by the harness) and are indicative, not billing-accurate.
- Citation faithfulness uses a single LLM judge; it is a proxy and inherits the judge model's biases. Parse failures are counted as non-faithful.
