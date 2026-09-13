=== CHECKPOINT: Session 1 complete (M0) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
MANDATED STACK: Qdrant (vector store, metadata filtering), LangGraph (agent), Python 3.12.

COMPLETED THIS SESSION:
- Repo skeleton with src/ layout: packages parse, chunk, ingest, retrieve, generate, agents, eval, common.
- uv-managed deps + uv.lock. Dev group: ruff, mypy, pytest, pytest-cov, pre-commit, detect-secrets.
- pre-commit hooks: ruff (lint+format), mypy --strict on src/, detect-secrets, basic hygiene hooks.
- docker-compose.yml running Qdrant pinned at v1.18.2 with a healthcheck + named volume.
- config.py (pydantic-settings, RN_ prefix) holds ALL tunables incl. refusal_similarity_threshold=0.35,
  retrieval_top_k=8, dense model BAAI/bge-small-en-v1.5, sparse model Qdrant/bm25, llm gpt-4o-mini.
- logging.py (structlog, console+JSON). common/qdrant.py (make_client + check_health, logs loudly).
- Thin Typer CLI (cli.py) with `healthcheck` command; entrypoint `research-navigator`.
- Tests: config, logging (unit); qdrant health (marked integration, deselected by default).
- Makefile: setup/lint/format/type/test/test-integration/up/down/healthcheck/clean.
- scripts/explore_corpus.py + docs/OBSERVATIONS.md template + docs/adr/0001 (meta-ADR).

KEY DECISIONS (keep consistent):
- uv over poetry/pip (speed, lockfile). Python 3.12. src/ layout (mandated).
- ruff for lint+format; mypy strict scoped to src/ only (tests untyped-friendly).
- detect-secrets for scanning. pydantic-settings for config. structlog for logging.
- make as task runner (matches grader's `make ...` acceptance vocabulary).
- Qdrant local Docker (OSS-first); :memory: reserved for unit tests later.
- typer for CLI (thin entrypoints). pytest+cov; integration tests opt-in via marker.
- Ground rules internalized: no silent failure (log + fail loudly), OSS-first, document assumptions.

CURRENT STATE:
- `make setup && make lint && make test` green on fresh clone.
- Qdrant reachable; `make healthcheck` OK. Corpus NOT yet ingested (no logic yet).
- corpus/ and documents/ are git-ignored; sealed corpus to be placed under corpus/manifest.json + documents/.

STOPPED AT: end of M0. No parsing/chunking/retrieval code exists yet.

NEXT STEP (Session 2 = M1a Parsing):
Build src/research_navigator/parse/ to convert each raw file into a normalized
document IR with section boundaries preserved. Handle arXiv PDF realities
(multi-column, footnotes, references, figures/tables) and Markdown (HF/Lil'Log/lab).
Decide the PDF library (pymupdf vs pdfplumber vs unstructured), define the IR schema,
and log every drop decision. Acceptance: parse produces a structured object for all 50 docs.
=== END CHECKPOINT ===
