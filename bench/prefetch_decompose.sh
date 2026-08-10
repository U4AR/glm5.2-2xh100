#!/usr/bin/env bash
# Decompose the prefetch loss into interference vs stall.
#
# The full prefetch costs +12.3 ms/step over its own control. "Queuing" is not
# an answer; this splits the number into the two things it can actually be:
#
#   T0  gather OFF                      slots allocated, store pinned, no bytes
#   T1  gather ON, wait OFF, route OFF  bytes move, nothing ever blocks on them
#   T2  gather ON, wait ON,  route OFF  same bytes, compute stream now waits
#   T3  gather ON, wait ON,  route ON   full prefetch (already measured: 84.88)
#   T2b as T2 but depth 2               twice the lookahead window
#
#   T1 - T0  = interference: SM slots and PCIe read bandwidth taken from work
#              that was going to happen anyway.
#   T2 - T1  = stall: the compute stream arriving at the barrier before the
#              bytes did. This is the only part a bigger window can fix, and
#              T2b says whether it does.
#   T2 - T3  = what routing to the landed slot actually buys back on the CPU.
#
# safe2 throughout (never sub2), MTP depth-3, GPU_EXPERTS=100 + 4 landing slots.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT=bench/profile_out/prefetch_rate.json

boot () {  # boot <label> <extra env assignments...>
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" \
    GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/dec_$label.log" 2>&1 &
  for i in $(seq 1 240); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/dec_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  grep -a "kt-prefetch\] step" "$SP/dec_$label.log" | tail -1
}

boot DEC-T0-nogather      KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_WAIT=1
boot DEC-T1-nowait-noroute KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_WAIT=0
boot DEC-T2-wait-noroute   KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_WAIT=1
boot DEC-T2b-depth2        KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_WAIT=1 KT_PREFETCH_DEPTH=2
echo "=== decomposition done ==="
