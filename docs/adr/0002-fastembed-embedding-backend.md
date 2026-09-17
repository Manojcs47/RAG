# ADR-0002: Embedding backend = FastEmbed (dense + sparse, one dependency)

Status: Accepted (Session 4)

## Context

M1c–f needs both a dense embedding (for semantic similarity) and a sparse
embedding (for lexical/BM25-style matching) to power hybrid retrieval (M2).
The assignment mandates Qdrant and says "use off-the-shelf models throughout"
(no fine-tuning). Candidates considered: `sentence-transformers` + a separate
BM25 library (two dependencies, two code paths); a hosted embedding API
(OpenAI/Cohere embeddings — network dependency + cost per chunk, and OSS-first
is a ground rule); FastEmbed (Qdrant's own embedding library, ONNX runtime,
no PyTorch).

## Decision

Use **FastEmbed** for both vectors, behind an `Embedder` protocol
(`ingest/embedder.py`):
- Dense: `BAAI/bge-small-en-v1.5` (384-dim).
- Sparse: `Qdrant/bm25` — emits raw term frequencies; Qdrant applies IDF
  server-side at query time (see ADR-0003), so the sparse vector itself is
  corpus-independent and doesn't need recomputing when the corpus grows.
- `build_embedder` fails loud (raises, no fake fallback) if construction
  fails — never silently returns a degraded embedder.
- Dimension is *probed* from the model at construction (`embedder.dense_dim`),
  never hardcoded, so the Qdrant collection schema always matches the actual
  model in use.

## Why

- **One dependency, two vector types.** `fastembed` ships both the dense
  bi-encoder and a BM25-compatible sparse encoder via ONNX, avoiding a second
  library (and a second failure mode) just for sparse vectors.
- **No PyTorch.** ONNX runtime is lighter to install and run than a full
  `sentence-transformers` + `torch` stack, and this project runs no GPU
  training — inference-only workloads don't need it.
- **OSS-first, no network cost per query.** Both models run locally after
  their one-time download; there's no per-embedding API cost or added
  latency from a hosted embedding service, and no OSS-first justification is
  needed for a proprietary alternative because none was used.
- **Protocol boundary.** Every embedder call in the codebase goes through
  `Embedder` (dense + sparse `embed_passages`/`embed_query`), so tests inject
  a fake embedder and the backend is swappable without touching `retrieve/`
  or `generate/`.

## Alternatives rejected

- **`sentence-transformers` + a standalone BM25 library**: two dependencies,
  two failure surfaces, no shared tokenizer guarantee between the "length
  budget" used at chunk time and the model actually embedding at ingest time.
- **Hosted embedding API**: violates OSS-first without a compelling reason;
  every ingest/query would incur network latency and per-call cost purely for
  embedding, on top of the LLM cost already incurred for generation.

## Consequences

- FastEmbed's ONNX models are downloaded once (from Hugging Face) and cached
  locally; the *first* `ingest`/`search`/`parse+chunk→ingest` run on a fresh
  machine needs that one-time network access, everything after is offline.
- Swapping the dense model requires recreating the Qdrant collection (the
  vector dimension is part of the schema) — `reindex` exists for exactly this.
- `ingest/embedder.py` carries a scoped `ignore_missing_imports` mypy override
  for `fastembed.*`/`tokenizers.*` (both ship without complete type stubs).
