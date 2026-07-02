#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh — ONE command to benchmark the running GLM-5.2 server on the
# full LiveBench reasoning suite (200 questions), one question at a time.
#
# Uses the top-2 expert tier by default (the fast recipe; served as GLM5.2-top2).
# The server must already be running (./run_fast.sh). Portable: the dataset is
# pulled from the HuggingFace hub on first run, so this works on any fresh box.
#
#   ./bench/run_benchmark.sh                    # full suite, top-2
#   MODEL=GLM5.2-top8 ./bench/run_benchmark.sh  # baseline tier for comparison
#   LIMIT=20 ./bench/run_benchmark.sh           # quick 20-question smoke test
#   TASK=zebra_puzzle ./bench/run_benchmark.sh  # one task only
#
# Overridable via env: MODEL, BASE, TASK, LIMIT, CATEGORY, MAX_TOKENS, OUT.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/config.sh"

MODEL="${MODEL:-GLM5.2-top2}"
BASE="${BASE:-http://localhost:8000/v1}"
CATEGORY="${CATEGORY:-reasoning}"
OUT="${OUT:-$REPO/livebench_results.json}"

# Prefer the project venv (has sglang + datasets); fall back to system python.
PY="python3"
[ -x "$VENV/bin/python" ] && PY="$VENV/bin/python"

# Make sure we can actually reach the server before pulling a dataset.
if ! curl -s -m 5 "$BASE/models" >/dev/null 2>&1; then
  echo "ERROR: no server at $BASE — start it first with:  ./run_fast.sh" >&2
  exit 1
fi

ARGS=(--model "$MODEL" --base "$BASE" --category "$CATEGORY" --out "$OUT")
[ -n "${TASK:-}" ]       && ARGS+=(--task "$TASK")
[ -n "${LIMIT:-}" ]      && ARGS+=(--limit "$LIMIT")
[ -n "${MAX_TOKENS:-}" ] && ARGS+=(--max-tokens "$MAX_TOKENS")

exec "$PY" "$REPO/bench/livebench/run_livebench.py" "${ARGS[@]}"
