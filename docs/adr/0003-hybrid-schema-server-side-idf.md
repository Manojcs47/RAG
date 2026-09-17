# ADR-0003: Hybrid Qdrant schema — one point, two named vectors, server-side IDF

Status: Accepted (Session 4)

## Context

Hybrid retrieval (M2) needs both a dense (semantic) and a sparse
(lexical/BM25) similarity search over the same corpus, fused into one ranking.
Qdrant supports this via **named vectors** on a single point, or via two
separate collections joined at query time. BM25-style sparse scoring also
needs IDF (inverse document frequency) weighting, which can be computed
client-side once and baked into the sparse vector, or server-side by Qdrant
from live collection statistics.

## Decision

- **One collection, one point per chunk, two named vectors**:
  `dense` (`VectorParams(size=dense_dim, distance=COSINE)`) and `bm25`
  (`SparseVectorParams(modifier=Modifier.IDF)`), defined in
  `ingest/schema.py`.
- **Server-side IDF**, set via `Modifier.IDF` **at collection-creation time**
  (Qdrant does not allow adding this to an existing collection without
  recreating it — documented inline in `schema.py` and enforced by always
  going through `store.create()`/`store.recreate()`).
- **Payload indexes** on every field actually used as a filter:
  `doc_id`, `content_type`, `primary_category`, `tags` (keyword, list
  membership), `is_foundational` (bool), `year` (integer).

## Why

- **One id, one payload, one ingest path.** Two vectors on the same point
  means there is exactly one place a chunk's identity and metadata can drift
  out of sync between the dense and sparse "sides" — there is no sync
  problem because there's only one write path (`ingest/pipeline.py`).
- **Server-side IDF stays current automatically.** Baking a fixed IDF into
  the sparse vector at embed time would freeze term statistics to whatever
  the corpus looked like at that moment; letting Qdrant compute it from live
  collection stats means every ingested/removed document keeps IDF accurate
  without recomputing any vectors.
- **Payload indexes match the actual query patterns**, not a generic
  "index everything": `doc_id` is indexed because the idempotent-upsert diff
  (ADR-0004) filters existing points by `doc_id` on *every* ingest call, and
  the rest are indexed because `retrieve/filters.py` builds `Filter` clauses
  on exactly these fields and no others.

## Alternatives rejected

- **Two collections (dense-only + sparse-only), joined client-side**: doubles
  the ingest path, doubles the place metadata can drift, and pushes fusion
  logic into application code instead of using Qdrant's native
  `Prefetch`+`FusionQuery` (RRF/DBSF) Query API.
- **Client-side/precomputed IDF**: would need re-embedding the entire corpus
  every time term frequencies shift meaningfully (e.g. corpus growth),
  instead of getting current statistics for free from the server.

## Consequences

- Qdrant's local (`:memory:`) mode does not enforce payload indexes (a
  `UserWarning` fires in tests: "Payload indexes have no effect in the local
  Qdrant"); the index *definitions* are still exercised by hermetic tests,
  but their actual query-acceleration effect is only real on the Docker
  server — flagged as unverified-in-CI in `PROJECT_LOG.md` and confirmed live
  in this session (`stats`/`validate`/`search` all ran against the real
  Docker Qdrant and returned correct, indexed-filter results).
- Changing which fields are filterable means editing `PAYLOAD_INDEXES` and
  running `reindex` (payload indexes, like the IDF modifier, are tied to
  collection creation).
