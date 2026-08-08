#!/usr/bin/env bash
# Now that the gather is bandwidth-bound, how far does politeness go?
#
# The root cause is settled. k_w2_weights copied a byte per thread-iteration
# with a 64-bit divide on each one, which made it LATENCY-bound: it reached
# bandwidth only by running 512 blocks wide, and 512 blocks of PCIe-stalled
# warps hold every SM slot and lock the main stream off the GPU. Throttling it
# was not an option either, because a latency-bound loop just gets slower --
# which is why KT_PREFETCH_BLOCKS=8 on the OLD kernel cost 8 ms/step. Both
# changes are needed, and together they are worth a lot:
#
#   old kernel, 512-wide     94.13 ms   22.46 exposed    0% hidden
#   fixed kernel, 512-wide   91.96 ms   20.29 exposed   10% hidden
#   fixed kernel, 128-wide   82.48 ms   10.81 exposed   52% hidden
#
# Standalone puts the knee lower still (0.412 ms at 8 blocks vs 0.392 at 16,
# i.e. politeness is nearly free below 16), so b8 is worth a row. The other
# open lever is coverage: the counters report 1.99 experts wanted per layer and
# 18.8% of layer-calls SKIPPED because the demand exceeds the 4 landing slots.
# Those skipped layers keep their CPU experts, so widening the slots converts
# directly into CPU work removed -- at the cost of more bytes.
#
# Reference points, same box state:
#   baseline, no predictor no gather   66.64 ms   42.87 tok/s
#   P0  predictor only                 71.67 ms   39.86
#   P3  full path, OLD kernel          84.67 ms   33.74
# Break-even for the full path is the 66.64 baseline.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
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
    GPU_EXPERTS=100 KT_STORE_SHM=1 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/tune_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/tune_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/tune_$label.log" | tail -1 || echo "NONE -- ROW IS VOID"
}

# The headline: full, numerically exact path at the polite grid. vs 84.67 old.
boot TUNE-P3-b8  KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1
# Is politeness still free at 8 blocks, or has the transfer started to stretch?
boot TUNE-P2-b8  KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
# Coverage lever: 6 slots retires most of the 18.8% skipped layers.
boot TUNE-P3-b8-s6 KT_PREFETCH_SLOTS=6 KT_PREFETCH_BLOCKS=8 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/tune_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 2 --tokens 200 \
  --label TUNE-prod-restore --out "$OUT" 2>&1 | tail -3
echo "=== tune done ==="
