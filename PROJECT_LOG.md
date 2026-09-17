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

=== CHECKPOINT: Session 4 complete (M1c–M1f — Embeddings + Qdrant ingest) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
STACK: Qdrant (v1.18.2 pinned), LangGraph, Python 3.12.
DONE: M0 setup; M1a parse -> data/parsed/*.json; M1b chunk -> data/chunks/*.json (full payloads);
      M1c–M1f ingest -> hybrid Qdrant collection.

COMPLETED THIS SESSION (M1c–M1f):
- New package src/research_navigator/ingest/:
  - embedder.py: Embedder Protocol + SparseVec; FastEmbedEmbedder (dense BAAI/bge-small-en-v1.5 384-dim
    + sparse Qdrant/bm25, ONNX/no-torch); build_embedder fails LOUD (no fake fallback). Dim probed, not
    hardcoded. BM25 emits raw TF; IDF is applied server-side by the collection.
  - schema.py: named vectors DENSE="dense", SPARSE="bm25"; sparse uses Modifier.IDF (must be set at
    creation). PAYLOAD_INDEXES: doc_id, content_type, primary_category, tags(keyword-list),
    is_foundational(bool), year(int).
  - ids.py: point_id = uuid5(RN_NAMESPACE, chunk_uid). chunk_uid embeds content_hash -> content-addressed
    ids; text edit => new id + old id stale.
  - store.py: thin Qdrant wrapper (all client calls isolated) — exists/create(+indexes)/drop/recreate,
    existing_ids_for_doc (paginated scroll), upsert/delete (batched), count/count_where, vector_names.
  - pipeline.py: ingest_doc (per-doc diff: add desired−existing, delete existing−desired, skip
    intersection; embed only what's written), ingest_corpus (ensures collection), collection_stats,
    validate_corpus (missing/stale/orphan/count drift).
  - models.py: DocIngestResult / IngestReport (added/deleted/unchanged/writes) / CollectionStats /
    ValidationReport. settings.py: IngestSettings. factory.py: build_client/build_store.
- cli.py: thin `ingest [--doc]`, `validate` (exit!=0 on drift), `reindex [--yes]`, `stats`.
- config.py: added `ingest: IngestSettings`. pyproject: mypy override ignore_missing_imports for
  fastembed.*, tokenizers.*. deps: fastembed, qdrant-client.
- tests/test_ingest.py (9): deterministic ids, named-vector collection, re-ingest = 0 writes, edit
  replaces point, remove deletes stale, full payload carry-through, stats counts, validate ok+drift,
  missing-collection. Hermetic via QdrantClient(":memory:") + fake embedder.

VERIFIED: ruff + ruff format + mypy --strict (24 files) clean; 26 tests pass (17 S3 + 9 S4).
  E2E smoke: ingest#1 writes=3; ingest#2 (unchanged) added=0 deleted=0 writes=0; stats by type/year
  /foundational correct; validate ok expected==actual==3.

KEY DECISIONS (keep consistent):
- FastEmbed backend (dense+sparse from one dep), behind Embedder protocol, fail-loud.
- One point, two named vectors (dense+bm25) + full payload; sparse Modifier.IDF (server-side, day-one).
- Content-addressed point ids (uuid5 over chunk_uid) => idempotent diff-based upsert.
- Embed only chunks being written; unchanged corpus = 0 embeds, 0 writes.

CURRENT STATE:
- `research-navigator ingest` populates the collection from data/chunks/*.json; validate/stats/reindex work.
- NOT yet retrieved from: no query understanding / fusion at query time yet.

OPEN ITEMS / TODO carried forward:
- Run real `ingest` in your env (sandbox had no HF egress); confirm dense_dim==384.
- Confirm on the Docker server (not :memory:) that payload indexes + IDF actually take effect (stats +
  a filtered query).
- Carried from M1a: verify arxiv-2408.00118 (Gemma 2) & arxiv-2501.12948 (DeepSeek-R1) manifest ids.
- docs/OBSERVATIONS.md: global chunk_index re-embeds trailing chunks on early edits; local-mode index
  no-op caveat; PDF no-headings/no-refs/column-interleaving notes.
- Write ADRs: "embedding backend = FastEmbed", "hybrid collection schema + server-side IDF",
  "content-addressed ids + idempotent upsert" (rubric wants ≥5).

STOPPED AT: end of M1f. Corpus is embeddable + idempotently indexed with full metadata.

NEXT STEP (Session 5 = M2 retrieval): query understanding (intent + filter inference), hybrid
retrieval via the Query API (dense + bm25 prefetch -> RRF FusionQuery), Qdrant-native metadata
filtering, top_k + refusal threshold from config.
** Verified this session for S5: query_points(prefetch=[Prefetch(using="dense"...), Prefetch(using="bm25",
   query=SparseVector...)], query=FusionQuery(fusion=RRF)) works on this collection; prefetch limit must
   be >= final limit. Re-verify LangGraph API at Session 7. **
=== END CHECKPOINT ===
=== CHECKPOINT: Session 5 complete (M2 retrieval — query understanding + hybrid retrieval) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
STACK: Qdrant (v1.18.2 pinned; client 1.19.1 verified), LangGraph, Python 3.12.
DONE: M0 setup; M1a parse; M1b chunk; M1c–M1f ingest (hybrid Qdrant collection);
      M2 retrieval (this session) -> filtered, fused, ranked chunks + refusal signal.

COMPLETED THIS SESSION (M2 retrieval):
- New package src/research_navigator/retrieve/:
  - settings.py: RetrieveSettings (top_k=8, prefetch_limit=40 forced>=top_k, fusion=rrf,
    dense_only=False, refusal_threshold=0.35 [COSINE], recency_window_years=2, infer_filters,
    plus configurable content_type/recency/foundational trigger phrases — no hardcoding).
  - vocab.py: QueryCatalog.from_manifest_rows -> tag vocab + content types DERIVED FROM MANIFEST
    (corpus-swap safe). match_tags: normalized whole-token/phrase match; canonical casing preserved
    (RAG, long_context). No LLM.
  - query_understanding.py: analyze() PURE fn -> QueryAnalysis(intent, InferredFilters, reasons[]).
    Deterministic rules: recency cue|explicit year|"last N years/months" -> year_gte; tag match;
    content_type triggers (intersected w/ corpus); foundational cue -> is_foundational=True.
    Coarse advisory intent (CONCEPT/RECENT/COMPARE/PAPER_SPECIFIC/FIND_PAPERS) — NOT the M3 router.
  - filters.py: build_qdrant_filter -> models.Filter (must=AND across fields; MatchAny=OR within
    tags/content_types). None when empty. Applied SERVER-SIDE (in each prefetch branch), not in Python.
  - ports.py: QueryEmbedder + HybridSearcher Protocols (retriever decoupled from store/embedder).
  - retriever.py: analyze -> embed_query -> build filter -> store.hybrid_query (RRF/DBSF; dense-only
    drops sparse) -> store.dense_query for COSINE confidence -> refusal = confidence<threshold or no hits.
    RECENT intent re-sorts chronologically. Filter-relaxation fallback: filtered result empty ->
    retry unfiltered + log warning (no false refusal, no silent failure).
  - factory.py: load_catalog(manifest) + build_retriever. __init__.py exports public API.
- PATCHES to existing files (additive; see PATCHES.md):
  - ingest/store.py: + Hit dataclass, hybrid_query(), dense_query() (all qdrant calls stay isolated).
  - ingest/embedder.py: + QueryVectors, Embedder.embed_query (bge QUERY path + bm25 query encoder).
  - config.py: + retrieve: RetrieveSettings (RN_RETRIEVE__*). cli.py: + thin `analyze`, `search`.
- tests/test_retrieve.py (18): filter inference (recency/explicit-year/last-N/tags/content-type/
  foundational/compare/plain/disabled), filter building, e2e hybrid ranking, server-side filter
  exclusion, dense-only, cosine refusal on off-corpus, chronological RECENT order, relaxation recovery.
  Hermetic via QdrantClient(":memory:") + FakeEmbedder (S4 pattern).

VERIFIED: ruff + ruff format clean; mypy --strict clean (14 files, src-scoped); 18 tests pass.
  Live :memory: demo: intent+reasons trace correct; "recent" excludes 2017 paper; RRF ranks;
  off-corpus query -> confidence 0.0 -> refused=True (chunks still ranked, gate is cosine-only).
  Qdrant Query API re-verified on client 1.19.1: prefetch(dense)+prefetch(bm25 SparseVector)+
  FusionQuery(RRF) with per-branch Filter works; dense-only probe returns cosine; DBSF also works;
  prefetch limit forced >= final limit.

KEY DECISIONS THIS SESSION (S5 open items now resolved -> write as ADRs):
- Filter inference = DETERMINISTIC RULES, not LLM (testable per M5; manifest-derived vocab; example
  in spec is pattern-based). LLM pass can augment later behind same analyze() signature.
- Fusion = RRF default; DBSF + dense_only are config switches (sets up M4 dense-vs-hybrid comparison).
- Refusal/score-normalization = threshold on max DENSE COSINE via a dedicated dense probe; RRF fused
  scores are rank-based (~small) and NOT comparable to the 0.35 cosine threshold — critical trap avoided.
- Filters applied via Qdrant primitives inside each prefetch branch (server-side), never post-hoc.
- Store stays the only place with qdrant-client calls; retriever depends on Protocols.

CURRENT STATE:
- `research-navigator search "<q>"` -> ranked chunks or graceful refusal; `analyze "<q>"` -> filters.
- Retrieval returns RetrievalResult(analysis, chunks, confidence, refused, fusion, dense_only, top_k,
  filtered) — everything S6 generation needs to cite-or-refuse.
- NOT yet generated: no inline [n] citations / structured citation blocks / citation dedup yet.

OPEN ITEMS / TODO carried forward:
- Run real `search` in your env (needs HF egress for FastEmbed; confirm bge query-embed path).
- Confirm on Docker server (not :memory:) that payload indexes accelerate filtered queries.
- Filter relaxation is currently ALL-OR-NOTHING; consider INCREMENTAL relaxation (drop recency before
  foundational) — log to docs/OBSERVATIONS.md. Also note recent+foundational can be contradictory.
- Carried: verify arxiv-2408.00118 (Gemma 2) & arxiv-2501.12948 (DeepSeek-R1) manifest ids.
- ADRs to write (rubric wants >=5): (1) embedding backend=FastEmbed, (2) hybrid schema + server-side
  IDF, (3) content-addressed ids + idempotent upsert, (4) filter inference = rules-not-LLM,
  (5) fusion=RRF + cosine-based refusal.
- Re-verify LangGraph API at start of Session 7.

STOPPED AT: end of M2 retrieval. Corpus is queryable: filtered + fused + ranked, with a calibrated
refusal signal in cosine space.

NEXT STEP (Session 6 = M2 generation): inline [n] citations + structured citation blocks
(title, authors first-et-al for >=3, year, source, section, URL), citation dedup (collapse
same-doc chunks to most-relevant section), consume res.refused for graceful low-confidence decline.
20 held-out Qs each cite-or-refuse; no fabricated citations.
=== END CHECKPOINT ===
=== CHECKPOINT: Session 6 complete (M2 generation — cited answers + refusal) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
STACK: Qdrant (v1.18.2 pinned; client 1.19.1), LangGraph, Python 3.12, OpenAI gpt-4o-mini (swappable).
DONE: M0 setup; M1a parse; M1b chunk; M1c–M1f ingest; M2 retrieval (S5);
      M2 generation (this session) -> grounded answers w/ validated inline citations + refusal.

COMPLETED THIS SESSION (M2 generation):
- New package src/research_navigator/generate/:
  - settings.py: GenerateSettings (refusal_message, refusal_sentinel=INSUFFICIENT_CONTEXT,
    require_citations, max_chunks_per_doc=3, source_char_budget=1200, max_sources=8,
    authors_etal_threshold=3, source_labels map — no hardcoding).
  - models.py: Source (doc-collapsed context), Citation (index/title/authors/year/source/section/url
    + render()), Answer (text/citations/refused/reason/intent/confidence/uncited_sentences + render()).
  - citations.py: format_authors (>=3 -> "First et al.", 2 -> "A and B"), extract_arxiv_id,
    source_label (arxiv_paper -> "arXiv:<id>"; else map: Lil'Log / Hugging Face Learn / Lab Blog),
    to_citation. PURE, unit-tested.
  - sources.py: build_sources -> CITATION DEDUP: group chunks by doc_id, one Source per doc, section =
    highest-scoring chunk's section, context = top chunks concatenated (char-budgeted), ordered by
    best score, numbered 1..N (<= max_sources).
  - markers.py: parse_markers; render_citations = THE ANTI-FABRICATION GUARD -> drop out-of-range
    (fabricated) markers, renumber survivors contiguously by first appearance, return (clean_text,
    used_sources_new_index, dropped[]). uncited_sentences (diagnostic). PURE.
  - prompt.py: build_messages (system: cite every factual sentence, only listed numbers, else emit
    sentinel; user: question + numbered sources). llm.py: ChatMessage, LanguageModel Protocol,
    OpenAIChatModel (lazy import, base_url => OSS OpenAI-compatible servers), build_llm fail-loud.
  - generator.py: Generator.answer -> retrieve; if res.refused -> refuse(low_confidence) [LLM NOT
    called]; build_sources; if none -> refuse; LLM.complete; if sentinel -> refuse(model_insufficient);
    render_citations (log dropped fabricated markers, no silent failure); if require_citations and no
    valid citation -> refuse(no_valid_citations); else Answer with Citations. Retrieving Protocol
    decouples from concrete retriever. factory.py: build_generator. __init__ exports API.
- config.py: + generate: GenerateSettings; LLMSettings + api_key + base_url (RN_LLM__*).
  cli.py: + thin `answer` command (retrieve -> generate -> render).
- tests/test_generate.py (14): author formatting, arxiv-id extraction, source labels, same-doc dedup,
  marker parse (grouped), fabricated-marker drop + renumber, noncontiguous renumber, e2e cited answer,
  refusal (low_confidence / sentinel / only-fabricated), sources-block render, two-author citation.
  Hermetic: FakeRetriever(RetrievalResult) + FakeLLM (NO network, NO qdrant).

VERIFIED: ruff + ruff format clean; mypy --strict clean (25 src files); 32 tests pass (18 S5 + 14 S6);
  cli.py type-checks against real-signature S4 stubs. Live FakeLLM demo: 3 chunks/2 docs -> 2 deduped
  sources; model's fabricated [7] dropped+logged, its sentence flagged uncited; "Dettmers et al." +
  arXiv:2305.14314 + section + URL rendered; low-confidence path refuses WITHOUT calling the LLM.

KEY DECISIONS THIS SESSION (-> ADR material):
- No-fabricated-citations is DETERMINISTIC: enforced by render_citations (drop out-of-range, renumber),
  not by trusting the LLM. Every surviving [n] maps to a real retrieved chunk.
- Citation dedup at SOURCE level: one citation per doc_id, pointing at the most-relevant section.
- Two refusal layers: retrieval cosine gate (S5) AND LLM sentinel abstention (S6) AND empty-citation
  guard -> graceful decline, never fabrication.
- Generation backend behind LanguageModel Protocol; OpenAI default, base_url makes it OSS-swappable.
- Uncited sentences are FLAGGED (diagnostic), not deleted, to avoid mangling model text; M4 LLM judge
  measures per-claim faithfulness. (Optional future: drop_uncited_sentences toggle.)

CURRENT STATE:
- `research-navigator answer "<q>"` -> grounded cited answer or graceful refusal.
- Full M2 pipeline works end to end (retrieve + generate). Ready for M3 agent routing.
- NOT yet: LangGraph router/route nodes/tool call (M3); eval harness (M4).

OPEN ITEMS / TODO carried forward:
- Run real `answer` (needs OPENAI_API_KEY + HF egress for FastEmbed). Confirm citation quality on
  real gpt-4o-mini output; watch for markers like [1][2] adjacency (parser handles [1, 2] and separate).
- Curate the 20 held-out Qs for M2 acceptance (each must cite-or-refuse) — feeds M4 golden set.
- Consider drop_uncited_sentences toggle + a stricter "every sentence cited" mode.
- Carried from S5: incremental (not all-or-nothing) filter relaxation; :memory: index no-op caveat;
  verify Gemma2/DeepSeek-R1 manifest ids.
- ADRs (rubric >=5): FastEmbed backend; hybrid schema+IDF; content-addressed ids; filter inference=
  rules; fusion=RRF+cosine refusal; (NEW) deterministic citation validation + source-level dedup.
- Re-verify LangGraph API at START of Session 7.

STOPPED AT: end of M2 generation. Answers are grounded, cited, deduped, and refuse gracefully.

NEXT STEP (Session 7 = M3 agent): LangGraph state machine — Router -> {concept_explanation,
paper_deep_dive, compare_approaches, recent_developments, find_papers, out_of_scope}. Serializable
state; >=1 tool call (e.g. corpus-metadata lookup or date-math for recent_developments). Reuse
Retriever (M2) + Generator (M2) inside route nodes. Graph visualized for design notes. Each route
exercised by >=3 test queries. VERIFY current LangGraph API signatures first.
=== END CHECKPOINT ===
=== CHECKPOINT: Session 7 complete (M3 — LangGraph agent) ===

PROJECT: AI Research Navigator — citation-grounded RAG + LangGraph agent over 50 AI/ML docs.
STACK: Qdrant (v1.18.2 pinned; client 1.19.1), LangGraph (1.2.11 verified this session), Python 3.12.
DONE: M0 setup; M1a parse; M1b chunk; M1c–M1f ingest (hybrid Qdrant collection);
      M2 retrieval (query understanding + hybrid fusion); M2 generation (cited/refusing answers);
      M3 agent (this session) -> router + 6 route nodes + tool layer over the M2 Generator.

** Verified this session for S7: langgraph 1.2.11 (1.x). Stable API used ->
   from langgraph.graph import StateGraph, START, END; from langgraph.graph.state import
   CompiledStateGraph (generic over StateT, ContextT, InputT, OutputT). Nodes = plain
   callables returning partial-state dicts; builder.add_node/add_edge/add_conditional_edges(
   source, path_fn, mapping)/compile(); graph.invoke(); graph.get_graph().draw_mermaid()
   (offline) / draw_mermaid_png() (mermaid.ink network). Re-verify eval-relevant APIs at S8. **

COMPLETED THIS SESSION (M3 agent):
- New package src/research_navigator/agents/ (9 files):
  - state.py: Route(StrEnum) — the 6 official routes; values DOUBLE AS node names (single source
    of truth, so router + wiring can't drift). ToolCall TypedDict. AgentState(TypedDict, total=False):
    query, now_year, route, route_reason, route_confidence, tool_calls, answer, answer_text, refused
    — all JSON primitives (serializable/checkpointable). StateUpdate = AgentState (partial dicts valid).
    CompiledAgentGraph alias (TYPE_CHECKING) pins the 4-arg CompiledStateGraph generic in ONE place.
  - settings.py: AgentSettings(BaseModel), nested under Settings.agents (env RN_AGENTS__*). All router
    cue lists + confidences + recency window + find_papers_limit + OOS message here. NO hardcoding.
  - tools.py: recency_cutoff(now, window) [pure date-math]; DocMeta (frozen); CorpusIndex
    (from_manifest / from_rows over manifest rows -> corpus-swap safe). find(text/tags/content_types/
    year_gte/year_lte/is_foundational/limit; AND across, ANY within, year-desc then title), match_tags
    (infers filters from the corpus's OWN tag vocab), stats/latest_year. This is the ">=1 tool call".
  - router.py: classify_by_rules() — deterministic, priority order find_papers > compare >
    recent > deep_dive > concept; each hit carries a human-readable reason. Router.route() = rules
    first, LLM fallback ONLY for ambiguous (via generate.llm LanguageModel protocol), then
    concept_explanation default. No silent failure (unparseable LLM reply logged + falls through).
  - nodes.py: Generating protocol (structural = M2 Generator.answer); AgentDeps (generator/corpus/
    router/settings). Router node + 6 route nodes via factories. concept & compare = straight cited RAG;
    recent = date-math + corpus-window tool calls + manifest-grounded preamble + recency RAG;
    deep_dive = corpus-resolve tool + cited RAG; find_papers = DETERMINISTIC manifest query (NO LLM,
    zero hallucination; empty -> graceful refuse); out_of_scope = fixed decline (no retrieval/LLM).
    Tool calls recorded in state["tool_calls"].
  - graph.py: START -> router --conditional on state["route"]--> {6 route nodes} -> END. compile()
    validates every node/edge at build time. graph_mermaid(deps) renders offline.
  - agent.py: Agent facade (run(query, now_year) -> AgentResult; route_of(); mermaid()). AgentResult
    frozen dataclass (fully serializable) with .render() showing the route header.
  - factory.py: build_agent(*, generator, manifest_path, llm, settings) -> Agent.
  - __init__.py: package exports.
- tests/test_agents.py: 23 hermetic tests (no network/qdrant/LLM/disk). FakeGenerator, FakeLLM,
  ExplodingLLM (proves rules path never calls LLM), CorpusIndex.from_rows over 5 inline manifest rows.
  Each of the 6 routes exercised by >=3 queries (acceptance met). Covers: rules table, find_papers>recent
  priority, OOS via LLM fallback (cue-free), llm-disabled default, unparseable-reply fallback, all-6-invoke,
  cited answer, refusal propagation, recency math, recent tool-calls+preamble, find_papers deterministic
  (gen not called), find_papers empty-refuse, deep_dive resolve+generate, OOS skips generator, corpus
  stats/find, JSON serializability of final state, graph compiles + mermaid renders, result render.
- scripts/visualize_graph.py: writes docs/agent_graph.mmd (offline); --png optional (mermaid.ink network).
  Uses a NullGenerator (topology is static; nodes never run during rendering) -> no Qdrant/embedder/API key.
- docs/agent_graph.mmd: rendered graph artifact (START -> router -> 6 conditional routes -> END).
- docs/adr/0007-m3-langgraph-agent.md: the S7 ADR (5 decisions, see below).
- Patches to existing files (see patches/*.patch.md): pyproject.toml (+langgraph>=1.2,<2 dep + scoped
  mypy override for agents.graph), config.py (+agents: AgentSettings field), cli.py (+`ask` full-agent
  and +`route` routing-only commands), Makefile (+graph / graph-png targets).

KEY DECISIONS / LOCKED (keep consistent) — recorded in ADR-0007:
- Router = HYBRID (deterministic rules first, LLM fallback for ambiguous only), mirroring S5's
  rules-first philosophy. Mis-route degrades to the M2 retrieval refusal gate -> graceful decline,
  never fabrication. Cue-free out_of_scope relies on the LLM fallback (pure-rule OOS rejected as fragile).
- find_papers is DETERMINISTIC from the manifest (no LLM) -> zero hallucination of titles/authors/years.
  Never calls find() with no constraint (would return whole corpus); empty match refuses.
- Serializable state = TypedDict(total=False) of JSON primitives; tool calls are plain dicts in
  state["tool_calls"]. A test round-trips the final state through json.dumps.
- recent_developments carries the required tool call as pure date-math (recency_cutoff) + a corpus
  window count, with a manifest-grounded preamble (grounded, not model-asserted).
- One scoped mypy suppression: disable_error_code=["call-overload"] for module
  research_navigator.agents.graph ONLY (LangGraph's 8-way add_node overload with a bounded NodeInputT
  TypeVar can't be inferred through a factory-produced Callable). Mirrors the existing cli.py Typer
  carve-out; justified by compile() build-time validation + all-6-routes tests. The related type-arg
  friction is fixed IN CODE via the single CompiledAgentGraph alias, not by suppression.

CURRENT STATE:
- Quality gates GREEN on the agents package + tests + script: ruff (repo select E,F,I,B,UP,SIM,C4,RUF)
  clean; ruff format clean; mypy --strict clean (with the one scoped override in pyproject);
  pytest 23 passed. All validated against real langgraph 1.2.11.
- `make graph` renders docs/agent_graph.mmd offline. `research-navigator route "<q>"` works offline
  (set RN_AGENTS__USE_LLM_ROUTER=false for zero-network routing).
- Agent is wired but the FULL `ask` path still needs the live M2 stack: OPENAI_API_KEY + Qdrant up +
  HF egress (FastEmbed). Routing + find_papers + out_of_scope run without any of that.

OPEN ITEMS / TODO carried forward:
- Curate the 20 held-out Qs from M2 acceptance into the M4 golden set (~40 Qs). The >=3-per-route
  query tables already in test_agents.py (CONCEPT_Q/COMPARE_Q/RECENT_Q/DEEP_Q/FIND_Q/OOS_Q) are a
  ready seed for the routing portion of the golden set.
- Carried from M1a/M1b: verify manifest ids arxiv-2408.00118 (Gemma 2) & arxiv-2501.12948 (DeepSeek-R1).
- ADR count for the rubric (>=5): ADR-0007 (this session) added. Confirm the running total in docs/adr/.
- docs/OBSERVATIONS.md: prior notes (chunk_index re-embed on early edits; PDF no-headings/no-refs/column
  interleaving) still stand — no new observations this session.
- Run `make lint && make type && make test` in the real repo after applying the 4 patches (the sandbox
  verified the code, but pyproject/config/cli/Makefile edits land in your repo).

STOPPED AT: end of M3. The agent routes each query to one of 6 routes, reuses M2 Retriever+Generator in
the route nodes, makes >=1 tool call, keeps serializable state, and the graph is rendered. Each route is
covered by >=3 tests. Acceptance met.

NEXT STEP (Session 8 = M4 eval harness): Build src/research_navigator/eval/. Golden set (~40 Qs) with
expected route + relevance labels; retrieval P/R@k; citation-faithfulness LLM judge; refusal correctness
(does it refuse when it should, answer when it should); latency/cost capture; a config comparison
(e.g. fusion strategy / top_k / dense-only). `make eval` emits a JSON + Markdown report. Reuse the
per-route query tables from test_agents.py as the routing seed. Re-verify any eval-relevant LangGraph /
OpenAI APIs before writing code.
=== END CHECKPOINT ===
=== CHECKPOINT: Session 8 — M4 Evaluation Harness (2026-09-17) ===

DELIVERED
- research_navigator.eval package (11 modules): ports, settings, golden, metrics,
  cost, judge, runner, report, harness, adapters, factory.
- Golden set: data/golden_set.json — 40 Qs across all 6 routes
  (concept_explanation 7, paper_deep_dive 6, compare_approaches 6,
   recent_developments 7, find_papers 6, out_of_scope 8); 32 retrieval + 8 refusal.
- Metrics: document-level precision/recall@k (k=3,5,8; primary 5), F1, routing
  accuracy, refusal precision/recall/accuracy, latency p50/p95, token+cost/query.
- Citation faithfulness: LLM-as-judge with explicit rubric (grounding + attribution),
  defensive JSON parse, score >= judge_min_score (0.8) gate, no silent failure.
- Reports: EvalReport -> report.json + one-page report.md comparing hybrid vs
  dense_only; honest Limitations section. `make eval` / `rn eval` (+ `--dry-run`).
- ADR-0008 (ports decoupling, doc-level P/R@k, tiktoken-optional cost, judge rubric,
  hybrid-vs-dense config comparison) + verbatim rubric. ADR total now 8 (>= 5).
- Patches: config.py (Settings.eval), cli.py (eval cmd + offline fakes),
  Makefile (eval / eval-dry-run), pyproject.toml (tiktoken extra, mypy note).
- tests/test_eval.py — 24 hermetic tests; scripts/eval_dry_run.py (offline).

QUALITY GATES
- ruff check + ruff format --check: clean.
- mypy --strict (src/research_navigator/eval): Success, 12 files.
- pytest: 24 passed. Coverage research_navigator.eval: 91%
  (factory.py 0% — live-stack seam, by design).
- Golden set validated: 40 Qs, all 6 routes, minimum-coverage assertion passes.

INTEGRATION SEAM (reconcile at wiring time — only thing S8 couldn't verify live)
- adapters.py / factory.py reference concrete names: settings.retrieval.use_sparse,
  settings.corpus.manifest_path, build_retriever/build_generator/build_llm/build_agent,
  AgentResult.route/refused/answer.citations. Confirm against real M2/M3 modules.
  Offline --dry-run + test suite do NOT touch this seam.

OPEN ITEMS (carried)
- Apply the 5 patches (S8's 4 + any pending) in the real repo, then:
  make lint && make type && make test.
- `make eval` (live) needs OPENAI_API_KEY + running Qdrant + FastEmbed egress;
  offline dry-run + golden validation run without any of these.
- Verify manifest ids arxiv-2408.00118 (Gemma 2) & arxiv-2501.12948 (DeepSeek-R1)
  resolve to intended titles.
- Install tiktoken (uv sync --extra eval) for exact token counts; else heuristic.

NEXT: Session 9 — M5 (raise total coverage >= 70%, README/ARCHITECTURE/ADRs pass,
  seeds, `docker compose up`, demo script, submission-ready).
=== END CHECKPOINT ===
