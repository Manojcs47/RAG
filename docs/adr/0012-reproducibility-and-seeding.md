# ADR-0012: Reproducibility and single-point seeding

Status: Accepted (Session 9)

## Context

M5 requires "fixed seeds where randomness is involved" and a reproducible run.
Most of this system is already deterministic *by construction*: chunk IDs are
content-addressed (`chunk/hashing.py`), the router runs rules-first before any
LLM fallback (`agents/`), hybrid fusion uses RRF (order-stable), retrieval
filters are applied server-side by Qdrant, and generation runs at
`temperature=0.0`. So there is very little live randomness to pin — but "very
little" is not "none," and reproducibility that depends on *no one ever adding
a random component* is fragile.

## Decision

A single `common/seeds.py::seed_everything(seed)` seeds every RNG the process
can touch — Python's `random`, `PYTHONHASHSEED` (for child processes), and
numpy's legacy global RNG (imported defensively; its absence is logged, not
swallowed). The seed and an on/off switch live in config
(`Settings.repro.seed` / `.seed_on_startup`, env `RN_REPRO__*`), and a Typer
`@app.callback()` in `cli.py` calls `seed_everything` before *every* command.
Reproducibility is thus a property of the entry point, inherited automatically
by any future code path — not something each new feature has to remember.

## Why

- **One seed, one place.** A future contributor who introduces a stochastic
  step (a sampled reranker, a non-zero-temperature ablation) inherits the seed
  for free instead of silently breaking determinism.
- **Config, not a constant.** The seed is a `pydantic-settings` field like
  every other tunable (no magic numbers), overridable per run for ablations.
- **Honest about `PYTHONHASHSEED`.** CPython reads it only at interpreter
  startup, so setting it at runtime governs child processes and documents
  intent but does not re-randomize the running process. The pipeline never
  relies on `set`/`dict` iteration order for output, so this is a belt-and-
  braces measure, and the docstring says so rather than implying a guarantee
  it can't make.
- **No silent failure.** numpy is a transitive dependency, not a hard one, so
  the numpy seed is best-effort with a debug log — consistent with the
  project's "log clearly, then recover or fail loudly" ground rule.

## Consequences

- `make test` covers `seeds.py` via `tests/test_seeds.py` (reproducible
  streams, env set, numpy pinned).
- The determinism the ADR claims is verified elsewhere too: chunking
  determinism (`tests/test_chunk.py`), routing determinism
  (`tests/test_agents.py`), and metric purity (`tests/test_eval.py`).
- Locked dependency files (`uv.lock`) and the pinned Qdrant image
  (`qdrant/qdrant:v1.18.2`, ADR-0001/compose) complete the reproducibility
  story at the environment level.
