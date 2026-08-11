#!/usr/bin/env bash
# The two claims the whole design rests on, checked on ONE boot.
#
#   CLAIM 1 (speed):     full routing costs what top0 costs.
#   CLAIM 2 (quality):   full routing is coherent where top0 is not.
#
# Together they are the hypothesis: keep every genuine expert that is already on
# the GPU, substitute only what the link could not carry, and pay top0's price
# for it. Neither claim means anything measured apart -- top0 is the FASTEST
# configuration that exists (it computes nothing genuine) and it produces
# garbage, so speed alone is satisfied by garbage and quality alone is satisfied
# by the 146 ms CPU path.
#
# SAME BOOT IS THE WHOLE METHOD. Both rows come from one server, switched per
# request through the model field, because a baseline agreeing across boots
# licenses nothing about the rows measured beside it -- that error cost this
# project a retracted 3.1 ms result.
#
# Configuration under test: the "cheap both" candidate -- GPU-only, cache AND
# prefetch live, but each made cheap (predictor on 1 layer in 4, cache ticking
# every 128 steps). Predicted ~54.3 ms from measured parts; the prediction is
# recorded here so the run can contradict it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

MODEFILE="$SP/kt_topk_mode.verify_top0"
echo safe8 > "$MODEFILE"
OUT=bench/profile_out/verify_top0.json
mkdir -p bench/profile_out
STRIDE="${KT_PRED_LAYER_STRIDE:-4}"
PERIOD="${KT_ADAPTIVE_PERIOD:-128}"

bash bench/_kill_servers.sh >/dev/null
echo "=== booting: GPU-only, cache + prefetch, stride $STRIDE, period $PERIOD ==="
KT_GPU_ONLY=1 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
KT_PRED_LAYER_STRIDE="$STRIDE" KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
KT_PREFETCH_SLOTS=1 KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD="$PERIOD" KT_ADAPTIVE_LAYERS_PER_TICK=2 \
KT_ADAPTIVE_MAX_SWAP=8 KT_TIER_COUNT_MODE=top8 \
GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_STORE_SHM=1 MTP=1 \
MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/vt0.log" 2>&1 &
for i in $(seq 1 400); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed|Bus error" "$SP/vt0.log" && { echo BOOTFAIL; tail -25 "$SP/vt0.log"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; tail -25 "$SP/vt0.log"; exit 1; }
grep -a "kt-pred-fused] stacked" "$SP/vt0.log" | tail -1

echo "--- warm-up: the cache must converge before either row is timed ---"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 \
  --tokens 1400 --label warm-vt0 --out "$SP/warm_vt0.json" 2>&1 | grep -a "ms/step" || true

echo "--- CLAIM 1: speed. top8 (the configuration) vs top0 (computes nothing genuine) ---"
for t in 8 0 8 0; do
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
    --tokens 200 --label "VT0-top$t" --out "$OUT" 2>&1 | grep -a "ms/step" || true
done
# Interleaved 8/0/8/0 rather than 8,8,0,0: if anything drifts during the run --
# thermals, the cache still settling -- an ordered sweep charges the drift to
# the tier. Two passes also give a same-boot repeatability estimate, which is
# the only honest noise floor for a 0.1 ms claim.

echo "--- counters ---"
grep -a "kt-prefetch] step" "$SP/vt0.log" | tail -2
grep -ac "kt-adaptive] layer=" "$SP/vt0.log" || true

echo "--- CLAIM 2: coherence, 1200 tokens, same boot ---"
TOKENS=1200 bash bench/coherence_run.sh vt0 8 0

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/vt0_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== verify_top0 done ==="
