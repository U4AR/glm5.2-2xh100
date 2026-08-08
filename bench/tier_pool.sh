#!/usr/bin/env bash
# Two corrections and one new test.
#
# CORRECTION 1: the tier ladder I measured on the live production server is not
# comparable to the prefetch boot. Production runs GPU_EXPERTS=104 /
# MEM_FRACTION=0.95; the prefetch boots run 100 / 0.94. More resident experts
# means less CPU work, so the CPU pool differs between them. That is the same
# cross-boot subtraction error Stage H6 already caught me making. Boot A redoes
# the no-prefetch ladder at the PREFETCH boot's settings so the pool and the
# prefetch are measured in one configuration.
#
# CORRECTION 2: at top0 the prefetcher still predicts and still gathers -- the
# prediction comes from the ROUTER's true top-2, not from the substituted ids --
# so its bytes are pure waste there. PF-top0 (70.96) is therefore NOT a floor;
# it is the floor plus the predictor plus a fully wasted gather. Boot A's top0
# is the real floor.
#
# THE NEW TEST. The counters explain why the prefetch did not scale with the
# pool: `KT_PRED_P` defaults to 2, so the predictor names at most two experts
# per token NO MATTER WHAT THE TIER IS (measured 2.05 wanted/call averaged over
# all three tiers). At top8 a layer routes through eight experts per token, most
# of them non-resident -- a two-expert prediction cannot cover it, and the
# 4-slot rule skips whatever it does not cover. So the fetch was structurally
# capped, not badly aimed.
#
# Boot B widens both to match the tier: KT_PRED_P=8, KT_PREFETCH_SLOTS=16
# (16 x 9.56 MiB = 153 MiB/card, affordable at MEM_FRACTION=0.94).
#
# Prediction, committed before the run: at top8 boot B beats boot A by more than
# the predictor's ~5.5 ms. If it does not, the cap is not the predicted-set size
# and the next suspect is the gather's exposed time (TODO item 4), which grows
# with the number of experts moved.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/latency_split.json

while pgrep -f "bash bench/prefetch_at_tier8.sh" >/dev/null 2>&1; do sleep 20; done
sleep 20

run_tiers () {
  local prefix="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $prefix: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/tp_$prefix.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/tp_$prefix.log" && { echo "BOOTFAIL $prefix"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $prefix"; return 1; }
  for T in 8 2 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier "$T" --runs 3 \
      --tokens 200 --label "$prefix-top$T" --out "$OUT" 2>&1 | tail -2
    echo -n "  counters after top$T: "
    grep -a "kt-prefetch\] step" "$SP/tp_$prefix.log" | tail -1 || echo none
  done
}

run_tiers OFF100 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0

run_tiers WIDE KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 KT_PRED_POINT=post \
  KT_PRED_P=8 KT_PREFETCH_SLOTS=16 KT_PREFETCH_BLOCKS=8 \
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/tp_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== tier_pool done ==="
