#!/usr/bin/env bash
# Does the vectorised w2 gather change anything IN THE SERVER?
#
# Root cause, measured (bench/gather_kernel_probe.py): k_w2_weights copied the
# down-projection ONE BYTE per thread-iteration with a 64-bit divide on each
# byte. That kernel is latency-bound, not bandwidth-bound -- 4.1 GB/s at 8
# blocks, 30 GB/s at 64, never reaching the 48 GB/s link limit that the already
# vectorised w13 kernel hits at 8 blocks. A transfer that can only move bytes
# by flooding the GPU with stalled warps cannot overlap with anything, because
# stalled warps still hold their SM slots -- and it cannot be throttled either,
# which is why KT_PREFETCH_BLOCKS=8 made the server WORSE (94 -> 102 ms).
#
# The uint4 rewrite is byte-identical (verified at 8/64/256 blocks for n_tp 1
# and 2) and holds link speed from 8 blocks up. Standalone, the whole gather
# now exposes 0.09 ms/layer = 6.6 ms/step against the 9.5 ms of CPU work it
# buys back. Standalone has been optimistic every single time in this
# investigation, so the only number that counts is this one.
#
# Reference points, same box state, OLD kernel:
#   baseline, no predictor no gather   66.6  ms   42.87 tok/s
#   P0  predictor only                 71.67 ms   39.86
#   P2  gather + barrier, no routing    94.13 ms   30.35
#   P3  full path, numerically exact    84.67 ms   33.74
#
# Break-even for P3 is 76.1 ms (the 9.5 ms CPU saving against P0's 71.67).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT=bench/profile_out/prefetch_rate.json

boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" \
    KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 \
    GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/fix_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/fix_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  # The prefetch path is silent unless KT_PRED_LOOKAHEAD is set; a row without
  # this counter is measuring the scaffolding and nothing else.
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/fix_$label.log" | tail -1 || echo "NONE -- ROW IS VOID"
}

# Cost of the bytes with the fixed kernel, at the old default grid. vs P2 94.13
boot FIX-P2-b64 KT_PREFETCH_BLOCKS=64 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
# Same, throttled. The old kernel could not survive this; the new one should
# hold link speed, and fewer resident warps is what buys overlap. vs P2 94.13
boot FIX-P2-b16 KT_PREFETCH_BLOCKS=16 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
# The shipped path, numerically exact, at both grids. vs P3 84.67, break-even 76.1
boot FIX-P3-b16 KT_PREFETCH_BLOCKS=16 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1
boot FIX-P3-b64 KT_PREFETCH_BLOCKS=64 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/fix_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 2 --tokens 200 \
  --label FIX-prod-restore --out "$OUT" 2>&1 | tail -3
echo "=== fixed-kernel test done ==="
