#!/usr/bin/env bash
# Stop everything: bench harnesses first, then the servers they would otherwise
# relaunch, then free the shared-memory expert store.
#
# The kill PATTERNS live in this file and never on the caller's command line.
# `pkill -f sglang.launch_server` typed inline matches the invoking shell's own
# argv and kills the caller -- that cost four shells and one silently deadlocked
# queue (a waiter whose `pgrep -f "bench/foo.sh"` matched itself) before the rule
# stuck.
set -uo pipefail

# 1. Harnesses first, so nothing reboots a server behind us.
for pat in chain_predict_run.sh fair_baseline.sh step_anatomy.sh \
           fixed_index.sh gather_truth.sh determ_fix.sh determ_ladder.sh \
           why_accept.sh pf_dump_boot.sh pf_verify_boot.sh fused_pred; do
  pkill -f "bench/$pat" >/dev/null 2>&1 || true
done
sleep 2

# 2. Servers.
pkill -f sglang.launch_server >/dev/null 2>&1 || true
pkill -f run_fast.sh >/dev/null 2>&1 || true
pkill -f run_server_int4.sh >/dev/null 2>&1 || true

# 3. Wait for VRAM to actually come back before declaring the box free -- the
#    scheduler processes exit well before the driver releases their allocations,
#    and a next boot that starts too early OOMs for reasons that look unrelated.
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${used:-99999}" -lt 4000 ] && break
  sleep 5
done

# 4. kt's expert store is a /dev/shm segment; a stale one wastes ~220 GB of RAM
#    and makes the next boot map the wrong thing.
rm -f /dev/shm/ktstore_* 2>/dev/null || true

echo "--- processes remaining:"
ps -eo pid,cmd | grep -E "[s]glang.launch_server|[r]un_fast.sh" || echo "  none"
echo "--- GPU:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- /dev/shm:"
ls /dev/shm 2>/dev/null | head -5 || true
df -h /dev/shm | tail -1
