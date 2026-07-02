#!/usr/bin/env bash
# ============================================================================
# run_terminalbench.sh — ONE command to evaluate the running GLM-5.2 server on
# the REAL Terminal-Bench 2.0 harness (Harbor: a Docker container per task with
# the task's own graders), then score it against GLM-5.2's original pass/fail
# labels.
#
# This is the executed-reward companion to run_benchmark.sh (LiveBench). Where
# LiveBench grades a single answer string, Terminal-Bench spins up a real shell
# environment, lets the terminus-2 agent drive the model, and runs the task's
# test suite for a hard pass/fail.
#
#   ./bench/run_terminalbench.sh                 # all 42 tasks, one at a time
#   TASK=fix-git ./bench/run_terminalbench.sh    # one task only
#
# Runs strictly SEQUENTIALLY, exactly like the LiveBench harness — one task at a
# time. This is deliberate: one GPU serves the agent, and running tasks
# concurrently starves per-request decode so agents blow their time budget and
# log a FALSE AgentTimeoutError (a task that passes solo shows as failed).
#
# PORTABLE: on a fresh machine the harness (harbor venv + terminal-bench-2 tasks)
# is bootstrapped automatically the first time you run this — nothing to copy.
# Just start the server (./run_fast.sh) and run this one command.
#
# Env: TASK, MODEL, BASE, TB_DIR, OUT.
# The heavy harness (harbor venv, terminal-bench-2 tasks) lives in TB_DIR; our
# reference labels + verdict aggregator are committed under bench/terminalbench/
# and copied in by the bootstrap. This script is the thin wrapper that
# bootstraps if needed and points the harness at localhost.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/config.sh"

# Pick TB_DIR: honour an explicit override; else reuse an existing repo-local or
# legacy harness; else default to a fresh repo-local one (bootstrapped below).
if [ -z "${TB_DIR:-}" ]; then
  if   [ -d "$REPO/.terminalbench" ];            then TB_DIR="$REPO/.terminalbench"
  elif [ -d /data/projects/isolated_bench ];     then TB_DIR="/data/projects/isolated_bench"
  else                                                TB_DIR="$REPO/.terminalbench"
  fi
fi
export TB_DIR
MODEL="${MODEL:-openai/GLM5.2}"
BASE="${BASE:-http://localhost:8000/v1}"
OUT="${OUT:-runs_full}"

# --- bootstrap the harness if it isn't set up (one-time, idempotent) ----------
if [ ! -x "$TB_DIR/venv/bin/harbor" ] || [ ! -d "$TB_DIR/terminal-bench-2" ] \
   || [ ! -f "$TB_DIR/task_labels.txt" ]; then
  echo ">>> Terminal-Bench harness not found at $TB_DIR — bootstrapping ..."
  bash "$REPO/bench/terminalbench/setup_terminalbench.sh" || {
    echo "ERROR: bootstrap failed" >&2; exit 1; }
fi

# --- rootless docker (harbor drives containers over the user socket) ---------
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-none}"
export OPENAI_API_BASE="$BASE"

# --- sanity: harness, docker, and a live server ------------------------------
HARBOR="$TB_DIR/venv/bin/harbor"
LABELS="$TB_DIR/task_labels.txt"
[ -x "$HARBOR" ]   || { echo "ERROR: no harbor at $HARBOR (set TB_DIR=)" >&2; exit 1; }
[ -f "$LABELS" ]   || { echo "ERROR: no task_labels.txt at $LABELS" >&2; exit 1; }
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker not reachable at $DOCKER_HOST — start it with:" >&2
  echo "       systemctl --user start docker.service" >&2
  exit 1
fi
if ! curl -s -m 5 "$BASE/models" >/dev/null 2>&1; then
  echo "ERROR: no server at $BASE — start it first with:  ./run_fast.sh" >&2
  exit 1
fi

cd "$TB_DIR"
mkdir -p "$OUT" logs_full

# strip the openai/ prefix for the --ak model handle the harness expects
MODEL_BARE="${MODEL#openai/}"

run_one() {
  local name="$1"
  echo ">>> $name"
  "$HARBOR" run -p terminal-bench-2/"$name" -a terminus-2 \
    -m "$MODEL" --ak api_base="$BASE" \
    -o "$OUT/$name" -n 1 --cpus ignore --memory ignore \
    > "logs_full/$name.log" 2>&1
  echo "DONE $name"
}

if [ -n "${TASK:-}" ]; then
  run_one "$TASK"
else
  echo "Running Terminal-Bench 2.0 on $(grep -c . "$LABELS") tasks "\
"sequentially (one at a time) against $BASE ..."
  while read -r name label; do
    [ -z "$name" ] && continue
    run_one "$name"      # strictly sequential — next task starts only when this one ends
  done < "$LABELS"
fi

# --- verdict: executed rewards vs GLM-5.2's original labels -------------------
echo
echo "=== VERDICT (executed reward vs original labels) ==="
"$TB_DIR/venv/bin/python" "$TB_DIR/build_verdict.py" 2>/dev/null \
  || python3 "$TB_DIR/build_verdict.py"
