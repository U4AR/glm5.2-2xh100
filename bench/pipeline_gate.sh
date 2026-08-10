#!/usr/bin/env bash
# The user's pipeline, at its measured-best operating point, through the gate.
#
# Hypothesis under test (user, 2026-08-08): GPU-only base (53 ms) + n+1 router
# pass on the current stream (small) + cache movement running parallel to
# compute. H22's slope fit says the only cheap movement is SMALL f: slots=1 +
# reuse gives coverage 1.0/call at +1.29 ms over the predictor row -> 56.45
# ms/step. This boot re-measures that speed and asks the only question left:
# is it COHERENT, and is it BETTER than resident-only (which is clean at 53.48
# with zero movement)?  If tier8 here is clean and tier2 is clean too (it
# degenerates in resident-only), the movement bought real quality margin.
# top0 must gate degenerate or nothing counts.
#
# KT_PREFETCH_BLOCKS is taken from $1 (the winner of blocks_fine), default 8.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"
BLOCKS="${1:-8}"

wait_for_marker "=== blocks_fine done ===" blocks_fine.out

bash bench/_kill_servers.sh >/dev/null
echo "=== booting GATE: slots=1 reuse=2 blocks=$BLOCKS, full pipeline ==="
KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_PREFETCH_SLOTS=1 KT_PRED_P=8 KT_PREFETCH_BLOCKS="$BLOCKS" \
KT_STORE_SHM=1 MTP=1 MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/gate.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/gate.log" && { echo BOOTFAIL; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; tail -25 "$SP/gate.log"; exit 1; }

echo "--- speed at every tier ---"
for t in 8 4 2 0; do
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
    --tokens 200 --label "Gate-top$t" --out bench/profile_out/gate_rate.json 2>&1 \
    | grep -a "ms/step" || true
done
echo -n "counters: "
grep -a "kt-prefetch\] step" "$SP/gate.log" | tail -1 || echo none

echo "--- coherence, 1200 tokens, top0 as the calibration case ---"
TOKENS=1200 bash bench/coherence_run.sh gate 8 4 2 0

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/gate_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== pipeline_gate done ==="
