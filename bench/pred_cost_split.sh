#!/usr/bin/env bash
# Split the predictor's 1.81 ms/step into its parts, because the budget says it
# is about to become the binding term.
#
#     floor (GPU-only, nothing predicted, nothing moved)  53.35
#     + n+1 fused predictor                               +1.81
#     + gather at slots=1 + reuse                         +1.29
#
# The movement half is closed (H22: superlinear SM contention, no barrier, grid
# optimal, DMA and priority both dead). If the cache drives the gather term
# toward zero, 1.81 ms is what stands between the pipeline and 54 ms -- and
# "1.81 ms of predictor" is a measurement, not a cause. This splits it.
#
# Three rows, each a boot, everything else identical:
#
#   PS-floor   no predictor at all              (KT_PRED_FUSED=0, prefetch off)
#   PS-cols    lookahead gate COLUMNS only      (SELECT=0: the fused gate GEMM
#              still computes the next layer's logits, but no selection kernel
#              runs and nothing is chosen)
#   PS-sel     columns + selection kernel       (GATHER=0: chooses, publishes,
#              moves no bytes)
#
# PS-cols - PS-floor  = the widened gate GEMM
# PS-sel  - PS-cols   = the 75 pred_fused launches and their work
#
# Which one dominates decides the fix and they point opposite ways: a GEMM cost
# means the lookahead columns are not free after all and the answer is to
# predict for fewer layers; a selection cost at ~24 us/layer against a ~3 us
# graph-replay launch means the kernel is doing real work per layer and the
# answer is to make it cheaper, not rarer.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

MODEFILE="$SP/kt_topk_mode.pred_split"
echo safe8 > "$MODEFILE"
OUT=bench/profile_out/pred_cost_split.json
mkdir -p bench/profile_out

boot () {   # boot <logname> <extra env...>
  local log="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $log: $* ==="
  env "$@" \
    KT_GPU_ONLY=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
    KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
    KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 KT_PREFETCH_SLOTS=1 \
    KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
    GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_STORE_SHM=1 MTP=1 \
    MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/$log.log" 2>&1 &
  local i
  for i in $(seq 1 400); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed|Bus error" "$SP/$log.log" && { echo "BOOTFAIL $log"; tail -30 "$SP/$log.log"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $log"; return 1; }
  return 0
}

# Tier 8 only: this is a pure timing split and GPU-only speed is
# tier-independent by construction, so extra tiers would only add boots.
row () {
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 3 \
    --tokens 200 --label "$1" --out "$OUT" 2>&1 | grep -a "ms/step" || true
}

boot PS_floor KT_PRED_FUSED=0 KT_PREFETCH_GATHER=0 KT_PREFETCH_SLOTS=0 && row PS-floor
boot PS_cols  KT_PRED_FUSED=1 KT_PREFETCH_GATHER=0 KT_PREFETCH_SELECT=0 && row PS-cols
boot PS_sel   KT_PRED_FUSED=1 KT_PREFETCH_GATHER=0                      && row PS-sel

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ps_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== pred_cost_split done ==="
