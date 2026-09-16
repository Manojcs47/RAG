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

=== CHECKPOINT: Session 2 complete (M1a — Parsing) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
STACK: Qdrant, LangGraph, Python 3.12. (Session 1/M0 done: repo skeleton, uv, ruff/mypy strict,
docker Qdrant v1.18.2, pydantic-settings config, structlog, typer CLI, tests green.)

COMPLETED THIS SESSION (M1a):
- Defined the document IR (the contract everything downstream depends on):
  parse/models.py -> ParsedDocument -> Section (index,title,level,kind,blocks) -> Block(type,text,language).
  SectionKind = abstract|body|references|appendix ; BlockType = paragraph|code|list|caption.
  ParsedDocument.retrievable_sections excludes references (references RETAINED on doc for citation lookup).
- common/types.py: ContentType enum. common/manifest.py: typed Manifest/ManifestEntry (month: int|None),
  load_manifest(). Reused by ingestion in Session 4.
- parse/pdf.py (PyMuPDF/fitz): column-aware ordering (midpoint split + 20% density check),
  modal-body-font detection, hybrid heading detection (section-number/Abstract/References regex OR
  font >= 1.12x body & short & no trailing period), abstract/references tagging, figure images skipped
  (counted), table/figure caption lines tagged as caption blocks, footnotes left inline.
- parse/markdown.py (markdown-it-py): frontmatter strip, heading-delimited sections, code fences preserved
  with language, list items tagged, Lil'Log nav/share/link-only noise dropped (counted in warnings).
- parse/dispatch.py: parse_document routes by content_type; parse_corpus reads manifest, resolves paths
  (corpus_dir/local_path with repo-root fallback), logs loudly + skips on missing/failed docs,
  caches IR to data/parsed/{doc_id}.json.
- config.py: added parsed_dir=Path("data/parsed"). cli.py: added thin `parse` command.
- Tests: markdown (kinds+code), pdf (synthetic in-test PDF, abstract/references kinds), dispatch routing.
- deps added: pymupdf, markdown-it-py. .gitignore: added data/.

KEY DECISIONS (keep consistent):
- IR is a typed Pydantic tree (not flat text / not tuples) so chunker reads structure as flags.
- PDF: PyMuPDF over GROBID (GROBID better for academic refs but needs a Java service; statement says
  "pymupdf or equivalent") / pdfplumber / unstructured / pypdf. Documented GROBID as declined option (ADR).
- Headings via font-size+regex hybrid (deterministic). Columns via midpoint split w/ density check.
- Figures skipped; table internals linearized (cell structure lost); references excluded-from-retrieval
  but retained; footnotes inline. All drops logged as warnings.
- Markdown via markdown-it-py (CommonMark token stream) over regex/python-markdown/mistune.
- Manifest typed via Pydantic (no dict[str,Any] in interfaces).

CURRENT STATE:
- `research-navigator parse` produces data/parsed/*.json for all present docs; make lint/test green.
- Corpus completed via corpus/complete_corpus.py. Note: month is null for course_chapter + Lil'Log entries.
- IR NOT yet chunked; nothing embedded or in Qdrant yet.

OPEN ITEMS / TODO carried forward:
- Record MANIFEST caveats: arxiv-2408.00118 (Gemma 2) and arxiv-2501.12948 (DeepSeek-R1) IDs to verify.
- Note any PDFs with "no headings"/"no references"/residual column interleaving in docs/OBSERVATIONS.md.

STOPPED AT: end of M1a. IR exists + cached.

NEXT STEP (Session 3 = M1b + payload schema): Design content-type-aware CHUNKING over the IR.
Treat abstracts distinctly; respect section boundaries; never split code blocks; exclude references
from retrieval (retain for citation lookup). Add content_hash for change detection. Define the full
per-chunk payload (section_title, section_index, chunk_index, content_hash + all manifest fields).
Deterministic, unit-tested chunks. Decide chunk sizing/overlap per content type and the tokenizer for
length measurement. Acceptance: deterministic chunks + complete metadata; unit tests cover chunking + hashing.
=== END CHECKPOINT ===

=== CHECKPOINT: Session 3 complete (M1b — Chunking + payload schema) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
STACK: Qdrant, LangGraph, Python 3.12. (M0 done; M1a done: PDF+MD -> typed IR cached to data/parsed/*.json.)

COMPLETED THIS SESSION (M1b + payload):
- New package src/research_navigator/chunk/:
  - settings.py: ChunkingSettings + ChunkPolicy(target/max/overlap/min tokens), per-content-type map
    (arxiv 320/512/64, survey_blog & lab_blog_post 384/512/64, course_chapter 320/512/48), abstract_max=512.
    All sizes in config; nothing hardcoded. Nested under Settings.chunking (env RN_CHUNKING__...).
  - tokenizer.py: Tokenizer Protocol; HFTokenizer (tokenizers lib on the dense model — length measured in
    the SAME subword unit that governs the 512 limit); WhitespaceTokenizer (deterministic, offline) used in
    tests + as a LOUD fallback (build_tokenizer logs + degrades, never silent).
  - hashing.py: normalize_for_hash (collapse cosmetic whitespace only), content_hash=sha256(hex),
    chunk_uid = f"{doc_id}:{chunk_index}:{hash[:16]}".
  - models.py: Chunk(ManifestEntry) -> inherits EVERY manifest field + adds section_index, section_title,
    section_kind, chunk_index (global 0..n-1), text, token_count, content_hash, is_abstract. .uid + .payload().
  - chunker.py: chunk_document(doc, entry, tokenizer, settings). Section-bounded greedy packing; prose split
    at sentence granularity, code atomic; abstract emitted as one flagged chunk; references excluded via
    retrievable_sections; token-bounded prose overlap trimmed so prose never exceeds max; long-sentence
    word-window fallback; below-min tail merge-back. Oversized code -> own chunk + logged warning.
  - dispatch.py: chunk_corpus reads data/parsed/{doc_id}.json, chunks, caches to data/chunks/{doc_id}.json;
    skips+logs missing IR; logs per-doc + total counts.
- config.py: added chunk_dir=data/chunks, chunk_tokenizer_model="BAAI/bge-small-en-v1.5", chunking=ChunkingSettings().
- cli.py: added thin `chunk` command.
- Tests: test_hashing.py (normalization/stability/uid) + test_chunker.py (determinism, abstract-distinct,
  code-never-split, references-excluded, section boundaries, chunk_index monotonic, size<=max, overlap,
  full manifest carry-through, empty doc, all content types have a policy). 17 tests, ruff+mypy(strict) green.
- deps added: tokenizers.

KEY DECISIONS (keep consistent):
- Length unit = embedding model's subword tokenizer (truncation-faithful); injected via Protocol.
- Payload = Chunk subclasses ManifestEntry so manifest carry-through is guaranteed by construction.
- content_hash over normalized text; chunk identity = (doc_id, chunk_index, content_hash) for S4 upserts.
- Overlap bounded by overlap_tokens AND clamped to max_tokens; code never duplicated/split.
- References excluded from retrieval, retained in data/parsed IR for citation lookup.

CURRENT STATE:
- `research-navigator chunk` produces data/chunks/*.json (full payloads) from cached IR. lint/test green.
- Chunks are deterministic per fixed tokenizer. NOT yet embedded; nothing in Qdrant.

OPEN ITEMS / TODO carried forward:
- Verify real bge tokenizer loads in your env (sandbox had no HF egress); spot-check arXiv chunks <=512 tokens.
- Carried from M1a: verify arxiv-2408.00118 (Gemma 2) & arxiv-2501.12948 (DeepSeek-R1) manifest IDs.
- Record in docs/OBSERVATIONS.md: global chunk_index shifts later uids on edits (design S4 diff accordingly);
  oversized code chunks truncate at embed; sentence splitter is regex-heuristic.
- Write ADR "chunking strategy per content type" (D1–D6 above).

STOPPED AT: end of M1b. Deterministic chunks + full payload exist + cached.

NEXT STEP (Session 4 = M1c–M1f): Dense+sparse embeddings, Qdrant collection schema, payload indexes on
content_type/year/tags/primary_category/is_foundational, deterministic point IDs from chunk_uid (uuid5),
idempotent upserts (re-ingest unchanged = 0 writes), ingest/validate/reindex/stats CLI.
** Per ROADMAP note: re-verify current Qdrant Query API / hybrid signatures before writing S4 code. **
=== END CHECKPOINT ===
