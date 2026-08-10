#!/usr/bin/env bash
# Start cost_isolate.sh once depth_shadow.sh has finished.
# Waits on a CONTENT MARKER, never on the process table -- a previous wrapper
# matched its own `bash -c` argv with pgrep and waited forever for itself.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
for i in $(seq 1 720); do
  grep -qaF "=== depth_shadow done ===" "$SP/depth_shadow.out" 2>/dev/null && break
  sleep 10
done
grep -qaF "=== depth_shadow done ===" "$SP/depth_shadow.out" 2>/dev/null || {
  echo "depth_shadow never finished; not starting cost_isolate"; exit 1; }
sleep 20
exec bash bench/cost_isolate.sh
