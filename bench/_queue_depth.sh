#!/usr/bin/env bash
# Start depth_shadow.sh once fair_baseline.sh has finished.
#
# Waits on a CONTENT MARKER in the log, not on a process pattern. A previous
# queue wrapper used `pgrep -f "bench/determ_fix.sh"` and matched its OWN bash -c
# command line, so it waited forever for itself. Nothing here inspects the
# process table at all.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
MARKER="=== fair_baseline done ==="

for i in $(seq 1 720); do   # up to 2 h
  grep -qaF "$MARKER" "$SP/fair_baseline.out" 2>/dev/null && break
  sleep 10
done

if ! grep -qaF "$MARKER" "$SP/fair_baseline.out" 2>/dev/null; then
  echo "fair_baseline never finished; not starting depth_shadow"
  exit 1
fi

sleep 20   # let its production restore settle before we kill it again
exec bash bench/depth_shadow.sh
