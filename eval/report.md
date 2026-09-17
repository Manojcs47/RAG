# AI Research Navigator — Evaluation Report

_Generated 2026-09-17T18:57:46+00:00 · golden set: 18 questions · primary k = 5 · token counts approximate (heuristic)._

## Configuration comparison

| Config | Routing acc | P@5 | R@5 | F1@5 | Faithfulness | Refusal acc | p50 ms | p95 ms | $/query | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 0.78 | 0.68 | 0.70 | 0.69 | 0.62 | 0.89 | 23525 | 48374 | 0.00021 | 0 |
| dense_only | 0.78 | 0.64 | 0.69 | 0.66 | 0.57 | 0.94 | 27730 | 42711 | 0.00023 | 0 |

## hybrid

- **Questions:** 18 (0 errored)
- **Retrieval (mean over retrieval subset):** P@3=0.68/R@3=0.63, P@5=0.68/R@5=0.70, P@8=0.68/R@8=0.71
- **Routing accuracy:** 0.78 overall; per route: compare_approaches=0.67, concept_explanation=1.00, find_papers=0.67, out_of_scope=0.33, paper_deep_dive=1.00, recent_developments=1.00
- **Refusal:** acc=0.89, recall=0.67 (refuses when it should), precision=0.67 (does not over-refuse), n=18
- **Citation faithfulness (LLM judge):** rate=0.62, mean score=0.62 over n=13 answers (4 judge parse failures)
- **Latency:** mean=22497ms, p50=23525ms, p95=48374ms, max=58797ms
- **Cost:** total=$0.0038, mean=$0.00021/query, tokens in/out=7505/4516

## dense_only

- **Questions:** 18 (0 errored)
- **Retrieval (mean over retrieval subset):** P@3=0.66/R@3=0.68, P@5=0.64/R@5=0.69, P@8=0.64/R@8=0.69
- **Routing accuracy:** 0.78 overall; per route: compare_approaches=0.67, concept_explanation=1.00, find_papers=0.67, out_of_scope=0.33, paper_deep_dive=1.00, recent_developments=1.00
- **Refusal:** acc=0.94, recall=0.67 (refuses when it should), precision=1.00 (does not over-refuse), n=18
- **Citation faithfulness (LLM judge):** rate=0.57, mean score=0.61 over n=14 answers (4 judge parse failures)
- **Latency:** mean=25370ms, p50=27730ms, p95=42711ms, max=62581ms
- **Cost:** total=$0.0041, mean=$0.00023/query, tokens in/out=6977/5162

## Limitations & honest notes

- Retrieval P/R is scored at the document level against a hand-labelled golden set; expected-source labels are best-effort and may under-count valid sources.
- Token/cost figures are estimated from the query, cited source text and the answer (the exact generation prompt is not observed by the harness) and are indicative, not billing-accurate.
- Citation faithfulness uses a single LLM judge; it is a proxy and inherits the judge model's biases. Parse failures are counted as non-faithful.
