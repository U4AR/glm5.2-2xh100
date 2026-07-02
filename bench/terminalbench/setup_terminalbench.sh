#!/usr/bin/env bash
# ============================================================================
# setup_terminalbench.sh — bootstrap the Terminal-Bench 2.0 harness so the
# benchmark runs on a FRESH machine with no manual steps.
#
# It reproduces, from scratch and idempotently, everything the run needs:
#   1. a self-contained venv with `harbor` (the terminus-2 agent ships inside it)
#   2. the terminal-bench-2 task set (git clone)
#   3. our reference pass/fail labels + verdict aggregator (committed in-repo)
#
# You normally never call this directly: ./bench/run_terminalbench.sh invokes it
# automatically when TB_DIR isn't set up yet. Run it by hand only to pre-warm.
#
# Everything lands in TB_DIR (default $REPO/.terminalbench, gitignored). Docker
# (rootless) is the one prerequisite it does NOT install — the run script checks
# for it and tells you how to start it.
#
# Overridable via env: TB_DIR, HARBOR_VERSION, TASKS_REPO, PYTHON.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/config.sh"

TB_DIR="${TB_DIR:-$REPO/.terminalbench}"
HARBOR_VERSION="${HARBOR_VERSION:-0.16.1}"   # pinned: the version we validated against
TASKS_REPO="${TASKS_REPO:-https://github.com/laude-institute/terminal-bench-2.git}"
PYTHON="${PYTHON:-python3}"
SRC="$REPO/bench/terminalbench"

mkdir -p "$TB_DIR"

# --- 1) venv + harbor (terminus-2 agent is bundled in the harbor package) -----
if [ ! -x "$TB_DIR/venv/bin/harbor" ]; then
  echo ">>> creating venv + installing harbor==$HARBOR_VERSION (one-time)"
  "$PYTHON" -m venv "$TB_DIR/venv"
  "$TB_DIR/venv/bin/pip" install -q --upgrade pip
  "$TB_DIR/venv/bin/pip" install -q "harbor==$HARBOR_VERSION"
else
  echo ">>> harbor already installed at $TB_DIR/venv"
fi

# --- 2) the terminal-bench-2 task set -----------------------------------------
if [ ! -d "$TB_DIR/terminal-bench-2/.git" ]; then
  echo ">>> cloning terminal-bench-2 tasks (one-time)"
  git clone --depth 1 "$TASKS_REPO" "$TB_DIR/terminal-bench-2"
else
  echo ">>> terminal-bench-2 tasks already present"
fi

# --- 3) our reference labels + verdict aggregator (committed, always refreshed)
cp "$SRC/task_labels.txt"  "$TB_DIR/task_labels.txt"
cp "$SRC/build_verdict.py" "$TB_DIR/build_verdict.py"

echo ">>> Terminal-Bench harness ready at $TB_DIR"
echo "    tasks: $(grep -c . "$TB_DIR/task_labels.txt")   harbor: $("$TB_DIR/venv/bin/harbor" --version 2>/dev/null)"
