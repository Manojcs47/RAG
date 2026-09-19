#!/usr/bin/env bash
#
# scripts/demo.sh — guided end-to-end walkthrough of the AI Research Navigator.
#
# Backs the 10–15 min recorded demo (Deliverable §6): ingestion status, one query
# through EACH of the six agent routes, an explicit low-confidence refusal, and the
# evaluation harness. Every command is echoed before it runs so the recording shows
# exactly what is executed.
#
# Usage:
#   bash scripts/demo.sh            # run straight through
#   RN_DEMO_PAUSE=1 bash scripts/demo.sh   # pause for <Enter> between sections
#   RN_DEMO_LIVE_EVAL=1 bash scripts/demo.sh   # run the LIVE eval (needs LLM + Qdrant)
#
# Prerequisites: `make setup && make up && make prepare && make ingest` have been run
# (see README §1–3). `ask`/`answer`/`eval` also need an LLM configured (see README
# "Configuration"); routing, find_papers and out_of_scope run with no LLM at all.

set -euo pipefail

RN="uv run research-navigator"
PAUSE="${RN_DEMO_PAUSE:-0}"

hr()   { printf '\n\033[1;34m════════════════════════════════════════════════════════════\033[0m\n'; }
say()  { printf '\033[1;36m▶ %s\033[0m\n' "$1"; }
run()  { printf '\033[0;90m$ %s\033[0m\n' "$*"; "$@"; }
wait_for_enter() { [ "$PAUSE" = "1" ] && { printf '\033[0;33m(press Enter)\033[0m'; read -r _; } || true; }

section() { hr; say "$1"; wait_for_enter; }

# --------------------------------------------------------------------------- #
section "0. Environment / Qdrant health"
run $RN healthcheck

section "1. Collection status (proves the corpus is ingested)"
say "Chunk counts by content_type, year, and tags:"
run $RN stats

section "2. Six agent routes — routing decision + full cited answer"

demo_route() {
  local label="$1"; shift
  local query="$1"; shift
  hr
  say "ROUTE: ${label}"
  printf '\033[0;37mQ: %s\033[0m\n' "$query"
  say "route only (deterministic, no LLM answer):"
  run $RN route "$query"
  say "full agent answer (retrieve → cite, or refuse):"
  run $RN ask "$query"
  wait_for_enter
}

demo_route "concept_explanation" \
  "How does retrieval-augmented generation work, and why does it reduce hallucination?"

demo_route "paper_deep_dive" \
  "What does the LoRA paper propose, and how does it reduce fine-tuning cost?"

demo_route "compare_approaches" \
  "Compare DPO, KTO, and SimPO as preference-optimization methods."

demo_route "recent_developments" \
  "What are the most recent developments in mixture-of-experts language models?"

demo_route "find_papers" \
  "Recommend foundational papers on transformers and attention for a beginner."

demo_route "out_of_scope" \
  "What is the best recipe for Hyderabadi biryani?"

# --------------------------------------------------------------------------- #
section "3. Low-confidence refusal (in-domain phrasing, absent from the corpus)"
say "The system must refuse rather than fabricate a citation:"
run $RN ask "What were the exact training FLOPs and cluster cost of GPT-5?"

# --------------------------------------------------------------------------- #
section "4. Evaluation harness (M4)"
if [ "${RN_DEMO_LIVE_EVAL:-0}" = "1" ]; then
  say "LIVE eval over the 40-question golden set (needs LLM + Qdrant; ~minutes):"
  run uv run rn eval
else
  say "Offline dry-run (no external services) — flip RN_DEMO_LIVE_EVAL=1 for the real run:"
  run uv run rn eval --dry-run
fi
say "Report artifacts: eval/report.md (one-page summary) + eval/report.json (machine-readable)."

hr
say "Demo complete. See README.md, ARCHITECTURE.md, and docs/adr/ for the design record."
