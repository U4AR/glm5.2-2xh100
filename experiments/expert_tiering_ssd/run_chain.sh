#!/usr/bin/env bash
# Serialised rung driver. Takes rung scripts as arguments and runs them one at a
# time under an flock, so a second chain cannot start a server while the first
# still owns the GPU.
#
# This exists because rungs H and I both died with CUDA OOM
# ("GPU 0 ... 293.75 MiB is free. Process 63754 has 47.94 GiB memory in use"),
# and the cause was TWO copies of rtx6000_no_thrash.sh running at once: killing
# an earlier driver's `bash -c` wrapper left its child rung script alive, and it
# was still waiting its turn when the new chain reached the same rung. Two
# servers, one card.
#
# Two guards, because they fail differently:
#   flock      one chain at a time, even across sessions and orphaned children.
#   setsid+PGID  killing this driver takes its rung scripts with it, so a
#              half-killed chain cannot linger and ambush the next one.
set -uo pipefail
cd /data/models/RunGLM
LOCK=/tmp/runglm_rung.lock

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another rung chain holds $LOCK (pid $(cat "$LOCK" 2>/dev/null)); refusing to start" >&2
  exit 1
fi
echo $$ >&9

# Any rung script still alive from a previous chain owns no lock but may own a
# server. Clear both before starting.
for p in $(pgrep -f "expert_tiering_ssd/.*\.sh" | grep -v "^$$\$"); do
  [ "$p" = "$$" ] || kill -9 "$p" 2>/dev/null
done
for p in $(pgrep -f "[s]glang"); do kill -9 "$p" 2>/dev/null; done
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-9999}" -lt 2000 ] && break
  sleep 5
done

trap 'kill 0' EXIT INT TERM   # take the whole process group down with us

for s in "$@"; do
  echo "=== chain: $s $(date -u '+%H:%M:%S') ==="
  bash "$s"
done
echo "=== chain complete $(date -u '+%H:%M:%S') ==="
