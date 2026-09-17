# ADR-0004: Content-addressed point IDs + per-document diff upsert

Status: Accepted (Session 4)

## Context

The assignment requires idempotent upserts: re-ingesting an unchanged corpus
must be a no-op, and editing one document must trigger targeted updates only
to its own chunks (§1d). Qdrant point ids must be an unsigned int or a UUID,
so a chunk's natural key (`doc_id`, `chunk_index`, `content_hash`) needs to be
mapped onto one deterministically.

## Decision

- `chunk_uid = f"{doc_id}:{chunk_index}:{content_hash[:16]}"`
  (`chunk/hashing.py`), and `point_id = uuid5(RN_NAMESPACE, chunk_uid)`
  (`ingest/ids.py`), where `RN_NAMESPACE` is a fixed, never-changing
  `uuid5` constant for the whole project.
- `content_hash = sha256(normalize_for_hash(text))` — normalization
  collapses only cosmetic whitespace, so the hash is stable across trivial
  re-parses but changes on any real text edit.
- Ingestion (`ingest/pipeline.py::_ingest_chunks`) is a **per-document diff**:
  compute the *desired* id set from the current chunk cache, read the
  *existing* id set back from Qdrant (scrolled, filtered by `doc_id`), then
  `to_add = desired − existing`, `to_delete = existing − desired`,
  `unchanged = desired ∩ existing`. Only `to_add` is embedded; only
  `to_delete` is removed; `unchanged` is never touched.

## Why

- **Idempotence falls out of the id function, not out of extra bookkeeping.**
  Because the id is a pure function of `(doc_id, chunk_index, content_hash)`,
  re-running ingestion on an unchanged corpus recomputes the exact same
  desired-id set every time — `to_add`/`to_delete` are both empty by
  construction, with no separate "have I seen this before" ledger to keep in
  sync.
- **A content edit is a natural add+delete, not a special case.** Editing one
  chunk's text changes its `content_hash`, so its id changes: the old id
  falls out of `desired` (→ deleted) and a new id appears (→ added). No
  explicit "diff the old value against the new value" logic is needed.
- **Embeddings — the expensive step — are computed only for what's actually
  new.** `_records_for` embeds exactly `to_add`, never the full corpus, which
  is what makes re-ingestion of a 4,777-chunk corpus a sub-second, zero-write
  operation (confirmed live this session).

## Alternatives rejected

- **Sequential/counter-based ids**: not reproducible across machines or runs
  without a shared counter service; doesn't naturally express "this chunk's
  content changed."
- **Re-upserting everything every run** (Qdrant upsert is itself idempotent
  per-point): correct but wasteful — it would re-embed and re-write the
  entire corpus on every ingest, defeating the "unchanged corpus = 0 writes"
  acceptance criterion.

## Consequences

- `RN_NAMESPACE` must never change for an existing collection — doing so
  remaps every point id and forces a full re-ingest (documented inline in
  `ingest/ids.py`).
- Point ids are opaque UUID strings; correlating a Qdrant point back to its
  source chunk always goes through the payload (`doc_id`, `chunk_index`), not
  the id itself.
- `chunk_index` is *positional* within a document (§ARCHITECTURE.md's chunk
  identity note): editing text early in a long document shifts every later
  chunk's index, and therefore its id — a full-document re-embed, not a
  single-chunk one. Acceptable for a mostly-static corpus; noted as a
  limitation in `docs/OBSERVATIONS.md`.
