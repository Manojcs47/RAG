# ADR-0010: Local Ollama model for `answer`/`ask`/`eval`, in place of a free-tier cloud key

Status: Accepted (Session 9)

## Context

`docs/OBSERVATIONS.md` §6 documents that the environment's configured cloud
LLM (Gemini 3.6 Flash, free tier, 20 requests/day) exhausted its quota partway
through a single `make eval` run (31/40 and 35/40 questions errored with
`RESOURCE_EXHAUSTED`). The M4 acceptance criterion requires a working
citation-faithfulness LLM-judge pass over the golden set, which needs a
sustained volume of successful LLM calls (generation + judge, per cited
question, per config) that a 20-req/day quota structurally cannot deliver.

Two constraints on this machine: no passwordless `sudo` (ruling out Ollama's
standard root-installing script), and initially very little free disk
(9.5GB, since freed to ~40GB by clearing ~37GB of regenerable package/browser
caches — a one-time, unrelated cleanup, not part of this decision).

## Decision

Run Ollama **without root**, entirely in user space:
1. Download the portable release tarball directly
   (`https://github.com/ollama/ollama/releases/download/vX.Y.Z/ollama-linux-amd64.tar.zst`)
   instead of the installer script — this skips every step that needs `sudo`
   (system binary symlink, a dedicated `ollama` system user, a systemd unit).
2. Extract to `~/.local/ollama` (a plain user directory).
3. Run the server as a background user process, not a systemd service:
   `OLLAMA_MODELS=~/.local/ollama_models OLLAMA_HOST=127.0.0.1:11434
   ollama serve`.
4. Pull a small instruct model sized for CPU-only inference (no GPU detected
   on this machine beyond an integrated iGPU Ollama declines to use without
   an explicit opt-in): `llama3.2:3b` (~2GB).
5. Point `RN_LLM__BASE_URL` at Ollama's OpenAI-compatible endpoint
   (`http://127.0.0.1:11434/v1`), `RN_LLM__MODEL=llama3.2:3b`,
   `RN_LLM__API_KEY=ollama` (any non-empty string — Ollama doesn't check it).
   **No code change** — this is exactly what `generate/llm.py`'s `base_url`
   parameter exists for (ADR-0002's "OSS-first" principle, applied here to
   generation rather than embeddings).

## Why

- **No quota, ever.** A local model has no per-day request cap — the binding
  constraint becomes wall-clock time (CPU inference is slow: ~90s for one
  generation call with an 8-source, ~1200-char-per-source context on this
  machine's CPU), not an external rate limit that fails the run outright.
  Slow-but-completing is strictly better than fast-until-quota-exhausted for
  actually exercising the judge path.
- **No new secrets, no billing.** Consistent with the project's OSS-first
  ground rule and avoids depending on the user having (or paying for) a
  cloud key just to run the evaluation harness the assignment requires.
- **The no-root path is the only path available on this machine** (no sudo),
  and it's a completely standard, supported Ollama deployment mode — nothing
  Ollama-specific was patched or hacked around; only the *installation*
  mechanism differs from the documented default.
- **Fully offline after the one-time pull.** No network egress at query
  time, matching the "self-hosted... unless you can justify why a
  proprietary alternative is materially better" ground rule.

## Alternatives rejected

- **A paid/higher-quota cloud key**: would work, but isn't something this
  environment has, and isn't reproducible for someone else running this
  repo without also paying for a key.
- **The standard Ollama installer (`curl | sh`)**: requires `sudo`, which
  this environment doesn't have configured for passwordless use; asking for
  a password to run an installer is exactly the kind of action this
  project's own guidance says to avoid without explicit user action.
- **A larger/more capable local model**: would likely improve answer and
  judge quality, but this machine has no GPU and ~7GB of usable RAM for
  inference — a 3B model is close to the practical ceiling for CPU-only
  inference at tolerable latency. Documented as a real quality/speed
  tradeoff, not hidden.

## Consequences

- Generation and judge latency both increase substantially vs. a cloud API
  (seconds → ~90s/call observed) — `make eval`'s live sweep now takes
  tens of minutes to a few hours rather than minutes, but *completes*
  instead of erroring out on quota.
- Answer/judge quality is bounded by a 3B instruct model, not a frontier
  model — citation-faithfulness scores from this judge should be read as
  "the pipeline correctly gates on a real score," not as a benchmark of the
  best achievable faithfulness.
- This setup is machine-local and not committed to git (`~/.local/ollama*` is
  outside the repo); `.env.example` documents the exact env vars needed to
  point at any Ollama instance, so reproducing this on another machine is a
  three-command `ollama pull` away, no code changes.
- If this machine later gets a GPU or a paid cloud key, switching back is a
  pure `.env` edit (`RN_LLM__BASE_URL`/`RN_LLM__MODEL`/`RN_LLM__API_KEY`) —
  no code path is specific to Ollama.

## Outcome

Confirmed: with the local model wired in, this switch **surfaced two real,
previously-hidden bugs in `eval/adapters.py`** that no prior cloud-backed run
had ever exercised long enough to hit — see `docs/OBSERVATIONS.md` §6 for the
full account. In short: `AgentResult.answer`'s citations were read via
`getattr()` on what is actually a plain `dict`, so `outcome.citations` was
silently always empty; and after fixing that, the judge model adapter passed
a bare string into a `LanguageModel.complete()` that requires
`list[ChatMessage]`. Both are fixed (`eval/adapters.py`, `eval/runner.py`)
with regression tests. A real 18-question run (3/route, the harness's own
coverage floor) then completed with 0 errors and produced genuine
citation-faithfulness scores (0.62 hybrid / 0.57 dense_only mean judge
score) — the first time in this project's history the M4 LLM-judge metric
has scored a real, non-fake answer. That report is now the one shipped at
`eval/report.md`.

This is the concrete payoff the "no quota, ever" rationale above predicted:
a backend that completes slowly surfaces bugs that a backend which fails
fast (on quota) never gets far enough to expose.
