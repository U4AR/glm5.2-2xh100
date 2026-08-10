#!/usr/bin/env bash
# Depth in GPU-ONLY mode, where losing coverage costs QUALITY instead of TIME.
#
# Depth 2 was rejected in Stage H14/H15 -- 65.82 ms/step against depth 1's 64.29,
# worse than not prefetching. But that was measured at safe2, and the measured
# CAUSE was coverage: residual CPU went 0.59 -> 2.81 ms because a layer the
# predictor missed fell through to the CPU expert path. Cost was IDENTICAL
# (10.85 vs 10.84), so depth never cost bytes -- it cost the CPU work that its
# misses created.
#
# GPU-ONLY MODE REMOVES THAT PENALTY ENTIRELY. There is no CPU fallback: a missed
# expert is substituted from the resident pool, which costs zero time. So in this
# mode depth keeps its benefit -- an extra whole layer of shadow for the transfer
# to hide in -- and loses its cost.
#
# The prize, if the exposed fraction is really a lead-time problem:
#
#   depth 1: 61.58 ms  = 53.31 floor + 1.64 predictor + ~6.6 exposed transfer
#   depth N:   ~55 ms  IF the extra lead time hides that 6.6 ms
#
# ~55 ms coherent would be the target this whole line of work has been chasing.
#
# THE TRADE IS PAID IN COVERAGE, so speed alone proves nothing. Whole-layer
# coverage over layers that need something: depth 1 = 75.0%, depth 2 = 63.1%.
# Twelve points more substitution. That is why every row here is checked for
# coherence at 1200 tokens with the top0 calibration gate armed -- a fast row
# that emits `</think>` four hundred times is not a result.
#
# Depth 3 is included because if the mechanism is lead time, more should keep
# helping until the coverage loss breaks the output. Finding WHERE it breaks is
# the point.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/gpu_depth_rate.json
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

boot () {
  local d="$1"
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting D$d: GPU-only, KT_PRED_FUSED_DEPTH=$d ==="
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
  KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH="$d" \
  KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
  GPU_EXPERTS=100 KT_PREFETCH_SLOTS=4 KT_PRED_P=8 \
  KT_STORE_SHM=1 KT_PREFETCH_BLOCKS=8 MTP=1 MEM_FRACTION=0.94 \
  KT_TOPK_MODE_FILE="$MODEFILE" \
  KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/d_$d.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/d_$d.log" && { echo "BOOTFAIL D$d"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP D$d"; return 1; }

  # SPEED FIRST, as asked: does depth actually buy the step time back?
  for t in 8 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "D$d-top$t" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  done
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/d_$d.log" | tail -1 || echo none

  # THEN coherence, 1200 tokens, with top0 as the calibration case.
  TOKENS=1200 bash bench/coherence_run.sh "depth$d" 8 0
}

boot 1
boot 2
boot 3

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/d_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== gpu_only_depth done ==="
