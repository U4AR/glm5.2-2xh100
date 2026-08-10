#!/usr/bin/env bash
# The block-count trend has not flattened: 512 -> 82.77, 128 -> 72.97, 64 -> 70.91.
# Standalone says the gather itself starts to stretch below 8 blocks (0.412 ms
# at 8 vs 0.515 at 4), so this is where politeness should stop paying. One row
# to find the knee rather than assume it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
env KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=4 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
  KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 GPU_EXPERTS=100 KT_STORE_SHM=1 \
  RUNGLM_TOPK_MODE=safe2 MTP=1 KT_GPU_PREFILL_THRESHOLD=0 \
  TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/b4.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/b4.log" && { echo "BOOTFAIL b4"; break; }
  sleep 10
done
if curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label TUNE-P3-b4 --out bench/profile_out/prefetch_rate.json 2>&1 | tail -3
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/b4.log" | tail -1
fi
echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/b4_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
echo "=== b4 done ==="
