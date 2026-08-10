#!/usr/bin/env bash
# Split CONTENTION from BARRIER without leaving CUDA graphs.
#
# The intended probe is unusable. `KT_PREFETCH_WAIT=0` was written to "move the
# bytes and remove the barrier", but removing the join leaves the forked
# pf_stream unjoined when graph capture closes:
#
#     CUDA error: capturing stream has unjoined work (cudaErrorStreamCaptureUnjoined)
#
# It is an EAGER-ONLY probe, and eager numbers cannot be compared with graph-mode
# rows. So the split has to come from the shape of the cost curve instead.
#
# FIT THE LINE. Every row keeps the full machinery -- predictor, gather, join,
# routing -- and varies only how many experts there are to move:
#
#     cost(f) = INTERCEPT + SLOPE x f
#
#     SLOPE     scales with bytes  -> CONTENTION while the transfer runs
#     INTERCEPT independent of bytes -> BARRIER + kernel launch, paid per layer
#                                       whether or not anything moves
#
# `KT_PREFETCH_SLOTS` caps f directly (measured: covered/call saturates at exactly
# the slot count), so 1/2/4/8 slots walk f without touching anything else. All at
# TIER 0, where the prefetch can save nothing and every millisecond over the floor
# is cost.
#
# PRIOR: depth 2 gave the gather a whole extra layer to finish in -- which should
# erase any stall a barrier was causing -- and moved the step 0.54 ms,
# non-monotone. That already points away from the barrier. A large intercept here
# would contradict it and would be the more interesting result.
#
# Slots are held at 4 in the VRAM sense by keeping GPU_EXPERTS=100 throughout;
# rows with fewer slots simply leave the extra capacity unused, so residency is
# constant and no row can be explained by it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
RATE=bench/profile_out/exposure_slope.json
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

wait_for_marker "=== resident_only done ===" ronly.out

boot () {
  local slots="$1"
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting S$slots: KT_PREFETCH_SLOTS=$slots ==="
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
  KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
  KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
  GPU_EXPERTS=100 KT_PREFETCH_SLOTS="$slots" KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
  KT_STORE_SHM=1 MTP=1 MEM_FRACTION=0.94 KT_TOPK_MODE_FILE="$MODEFILE" \
  KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/s_slot$slots.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/s_slot$slots.log" && { echo "BOOTFAIL S$slots"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP S$slots"; return 1; }
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 0 --runs 3 \
    --tokens 200 --label "S$slots-top0" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/s_slot$slots.log" | tail -1 || echo none
}

boot 1
boot 2
boot 4
boot 8

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/slope_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== exposure_slope done ==="
