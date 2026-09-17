# ADR-0006: RRF fusion by default + a separate dense-cosine refusal probe

Status: Accepted (Session 5)

## Context

M2 needs (a) a way to combine dense and sparse rankings into one result, and
(b) a "low-confidence refusal path" — a similarity threshold below which the
system declines rather than answers. Qdrant's Query API supports fusing
`Prefetch` branches via `FusionQuery(fusion=RRF | DBSF)`. The trap: fused
scores from RRF are **rank-based** (small numbers with no fixed scale) and
are not comparable to a fixed cosine-similarity threshold — thresholding the
fused score directly would either never refuse or always refuse depending on
`top_k`/`prefetch_limit`, not on actual relevance.

## Decision

- **Fusion**: RRF (`fusion="rrf"`) is the default; DBSF and a dense-only mode
  (`dense_only=True`, dropping the sparse branch entirely) are config
  switches on `RetrieveSettings`, set up for the M4 hybrid-vs-dense-only
  comparison the assignment asks for.
- **Refusal**: a **separate dense-cosine probe** (`store.dense_query`) is run
  against the same candidate set purely to get a comparable, fixed-scale
  confidence number. `retriever.py`: `refused = confidence < refusal_threshold
  or not hits`, where `confidence` is the max cosine from that probe, never
  the fused rank score. Current tuned value: `refusal_threshold = 0.25`
  (COSINE space) — see `docs/OBSERVATIONS.md` for a note on drift between
  this value and an earlier-stated `0.35` in session logs.
- **RECENT intent** re-sorts the final chunk list chronologically after
  fusion (fusion picks relevance; the recency-biased route then re-orders
  for recency without discarding the relevance ranking that got it there).

## Why

- **Fusion and confidence answer different questions.** Fusion ranks
  candidates *relative to each other*; refusal needs an *absolute* signal
  ("is anything actually relevant"). Conflating them — thresholding the
  fused score — was identified and explicitly avoided as "a critical trap"
  during implementation (verified in a live `:memory:` demo: an off-corpus
  query returns `confidence=0.0` → refused, while the fused ranking of
  those same, irrelevant chunks still "looks" ordered).
- **RRF as the default** is a standard, parameter-light way to combine
  heterogeneous score distributions (cosine similarity vs. BM25-like sparse
  scores) without needing to calibrate their relative scales against each
  other.
- **Config switches, not hardcoded fusion**, directly enable the M4
  acceptance criterion ("compare at least two configurations — e.g.
  dense-only vs. hybrid") without duplicating retrieval code per
  configuration.

## Alternatives rejected

- **Thresholding the fused RRF score directly**: scale depends on
  `prefetch_limit`/rank position, not on relevance — would refuse or accept
  inconsistently as those parameters change, with no principled way to pick
  a threshold.
- **DBSF as the default**: also rank-based in a different way; RRF was
  chosen as the more standard/simpler default, with DBSF available as an
  explicit opt-in for comparison, not because DBSF was found worse.

## Consequences

- Every retrieval performs one extra dense-only query (the confidence probe)
  beyond the fused hybrid query — a deliberate cost for a correct refusal
  signal, not an oversight.
- The refusal threshold is a single tunable float in `RetrieveSettings`;
  retuning it does not require touching fusion logic at all.
- `search`/`answer`/`ask` all surface `confidence` and `fusion` in their
  output specifically so a mis-calibrated threshold is visible during manual
  testing, not just in aggregate eval metrics.
