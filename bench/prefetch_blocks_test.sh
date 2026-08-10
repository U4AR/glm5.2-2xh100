#!/usr/bin/env bash
# Does throttling the gather's grid recover the overlap IN THE SERVER?
#
# bench/graph_overlap_probe.py found a cliff: a host->device pull and a compute
# kernel overlap 100% up to 128 blocks, 87% at 512, and at 1024 they are WORSE
# than serial (5.72 ms vs 2.46). Stalled PCIe warps hold their SM slots, so a
# wide gather locks compute off the GPU entirely. Bandwidth is flat across the
# whole range -- 49.7 GB/s at 8 blocks and at 8192 -- so the width buys nothing.
#
# CAVEAT worth stating before the numbers land: the real grid is
# grid(blocks, 2*n_tp, n_slot) and the boot log reports n_tp=1, so at the
# default blocks=64 the grid is 64*2*4 = 512, not 1024. The probe puts 512 at
# 87% overlap, which accounts for only ~3 ms of the measured 22.46 ms. So this
# predicts a PARTIAL recovery. If B8-P2 lands near 72 ms the block count was
# the whole story and the probe understated it; if it lands near 90 ms the
# block count is a side effect and the root cause is still open.
#
# Reference points, same box state:
#   baseline, no predictor no gather   66.6  ms
#   P0  predictor only                 71.67 ms
#   P2  gather + barrier, no routing    94.13 / 94.74 ms
#   P3  full path, numerically exact    84.67 ms
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
    nohup ./run_fast.sh > "$SP/blk_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/blk_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/blk_$label.log" | tail -1 || echo "NONE -- ROW IS VOID"
}

# The gather's cost, with the grid throttled. Compare to P2 = 94.13.
boot B8-P2-bytes-noroute  KT_PREFETCH_BLOCKS=8 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
# The shipped path, numerically exact. Compare to P3 = 84.67 and baseline 66.6.
boot B8-P3-full           KT_PREFETCH_BLOCKS=8 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/blk_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 2 --tokens 200 \
  --label B8-prod-restore --out "$OUT" 2>&1 | tail -3
echo "=== blocks test done ==="
