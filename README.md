# AI Research Navigator

A citation-grounded Retrieval-Augmented Generation (RAG) system with an agentic
routing layer, built over a curated 50-document AI/ML corpus (arXiv papers,
Hugging Face Learn chapters, Lil'Log surveys, lab blog posts). Every
substantive claim in an answer carries an inline `[n]` citation back to a real
retrieved chunk, or the system refuses rather than fabricating one.

**Stack:** Qdrant (hybrid vector store, metadata filtering) · LangGraph (agent
orchestration) · Python 3.12 · [uv](https://docs.astral.sh/uv/) · FastEmbed
(dense + sparse embeddings) · an OpenAI-compatible LLM for generation.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design and
[docs/adr/](docs/adr/) for individual decision records.

## Quickstart (clone → working demo)

### 0. Prerequisites
- Python 3.12, [`uv`](https://docs.astral.sh/uv/getting-started/installation/), Docker + Docker Compose.
- An LLM for generation (`answer`/`ask`/`eval`) — see [Configuration](#configuration).
  Everything else (`healthcheck`, `parse`, `chunk`, `ingest`, `validate`, `stats`,
  `analyze`, `search`, `route`) works with **no LLM at all**.
  Recommended: a local model via Ollama (no API key, no quota, no cost —
  see [docs/adr/0010-local-llm-for-eval.md](docs/adr/0010-local-llm-for-eval.md)
  for the no-root install used in this environment) rather than a cloud key;
  a free-tier cloud key will not sustain `make eval`'s call volume.

### 1. Install + start Qdrant
```bash
make setup   # uv sync, pre-commit install, secrets baseline
make up      # docker compose up -d (Qdrant v1.18.2)
make healthcheck
```

### 2. Get the corpus
The 50-document corpus is a sealed package (`corpus/`). Hugging Face Learn and
Lil'Log markdown are committed to this repo; the 30 arXiv PDFs and 3 lab blog
posts are fetched from the public internet on first use (they're large
binaries and aren't committed). Run once:
```bash
cd corpus && pip install requests trafilatura html2text && python3 complete_corpus.py && cd ..
```
See [corpus/README.md](corpus/README.md) for what this script does and how to
verify the corpus is complete (`Missing: 0`).

### 3. Parse, chunk, ingest
```bash
make prepare   # research-navigator parse && chunk -> data/parsed/, data/chunks/
make ingest    # embeds + upserts into Qdrant (idempotent: re-running is a no-op)
make validate  # exit 0 iff the collection matches the corpus, no drift
```
`make prepare` needs no network beyond the corpus step above (FastEmbed's ONNX
models are downloaded from Hugging Face on first run, then cached locally).

### 4. Ask it something
```bash
uv run research-navigator search "what is retrieval augmented generation"   # M2 retrieval, no LLM
uv run research-navigator answer "what is retrieval augmented generation"   # M2 + citation, needs an LLM key
uv run research-navigator ask "compare LoRA and full fine-tuning"           # M3 agent routing + M2
uv run research-navigator route "compare LoRA and full fine-tuning"        # routing decision only, no LLM needed on the rule path
```

### 5. Evaluate
```bash
make eval-dry-run   # offline, in-repo fakes — always works, no external services
make eval           # live: 40-question golden set x {hybrid, dense_only} -> eval/report.md
```
**Note:** the live sweep makes ~80+ LLM calls. A free-tier key (e.g. Gemini's
20 requests/day) will exhaust its quota partway through and most questions
will error out with `RESOURCE_EXHAUSTED` — this is a quota limit, not a bug;
the harness logs each failure and still emits a report (see
[docs/OBSERVATIONS.md](docs/OBSERVATIONS.md)). Use a paid key or a local
OSS-compatible server for a meaningful live run.

## CLI reference

| Command | Milestone | Needs Qdrant? | Needs LLM? | Purpose |
|---|---|---|---|---|
| `healthcheck` | M0 | ✅ | — | Verify Qdrant is reachable |
| `parse` | M1a | — | — | Corpus → `data/parsed/*.json` (IR) |
| `chunk` | M1b | — | — | `data/parsed/` → `data/chunks/*.json` |
| `ingest [--doc ID]` | M1c–f | ✅ | — | Chunks → Qdrant (idempotent upsert) |
| `validate` | M1c–f | ✅ | — | Detect drift; exit ≠0 on mismatch |
| `reindex [--yes]` | M1c–f | ✅ | — | Drop + rebuild the collection |
| `stats` | M1c–f | ✅ | — | Point counts by type/year/tags |
| `analyze QUERY` | M2 | — | — | Show inferred intent + filters only |
| `search QUERY` | M2 | ✅ | — | Ranked chunks or a refusal |
| `answer QUERY` | M2 | ✅ | ✅ | Full retrieve → cited answer |
| `route QUERY` | M3 | — | maybe* | Routing decision only |
| `ask QUERY` | M3 | ✅ | ✅ | Full agent: route → retrieve → cited answer |
| `eval [--dry-run]` | M4 | ✅** | ✅** | Golden-set evaluation → `eval/report.{json,md}` |

\* `route` only calls the LLM if no deterministic rule fires (ambiguous / cue-free queries).
\*\* not with `--dry-run`.

Run `uv run research-navigator --help` or `... COMMAND --help` for full option lists.

## Configuration

All tunables live in `src/research_navigator/config.py` (pydantic-settings,
prefix `RN_`, **double-underscore** nesting — see comments in
[.env.example](.env.example)). Copy it to `.env` and fill in an LLM key:
```bash
cp .env.example .env
```
Nothing is hardcoded: chunk sizes, retrieval thresholds, router cues, eval
settings, etc. are all fields on `Settings` and overridable via env vars.

## Testing & quality gates

```bash
make lint   # ruff check + ruff format --check + mypy --strict (src/)
make test   # pytest, hermetic (:memory: Qdrant / fakes), ~90% coverage
make test-integration   # the one test that needs a live Qdrant (make up first)
```
No test above requires an LLM key or network egress; FastEmbed and Qdrant are
faked or run against `:memory:`.

## Project layout

```
src/research_navigator/
├── parse/      M1a  PDF (PyMuPDF) + Markdown (markdown-it-py) -> typed IR
├── chunk/      M1b  content-type-aware chunking, deterministic content hashing
├── ingest/     M1c-f  FastEmbed embeddings, Qdrant hybrid schema, idempotent upsert
├── retrieve/   M2   query understanding (rules), hybrid RRF/DBSF fusion, refusal gate
├── generate/   M2   citation-grounded generation, anti-fabrication marker validation
├── agents/     M3   LangGraph router + 6 route nodes, tool calls, serializable state
├── eval/       M4   golden-set harness: P/R@k, routing acc, refusal, LLM-judge faithfulness
├── common/     manifest loading, Qdrant health, shared types
└── cli.py            thin Typer entrypoints only — no business logic
```

## Known limitations

See [docs/OBSERVATIONS.md](docs/OBSERVATIONS.md) for the full, honest list
(heading-detection false positives on ~27% of arXiv PDF sections, live-eval
LLM quota constraints, filter-relaxation being all-or-nothing, etc.) and
[eval/report.md](eval/report.md) for the latest evaluation numbers and their
own limitations section.
