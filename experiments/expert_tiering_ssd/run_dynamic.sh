#!/usr/bin/env bash
# Second sweep: the speed A/B, then dynamic placement at a fixed RAM budget.
#
#   r160_tick_on   full coverage WITH the shipped count-based adaptive cache.
#                  Pairs with the static ladder's r160 (24.01 tok/s, tick off)
#                  to answer "is 24 a regression, or just a frozen cache?"
#   r32_static     control: the static ladder point, re-run for same-session
#                  comparability.
#   r32_count      count-based two-cut movement (KT_TIER_DYNAMIC=1).
#   r32_energy     energy-driven movement (KT_ENERGY=1).
#
# Everything else is held fixed: GPU_EXPERTS=96, MEM_FRACTION=0.85, safe2 +
# MTP depth-3, per-request tier top2.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
OUTDIR="experiments/expert_tiering_ssd/runs"
mkdir -p "$OUTDIR" logs

run_one() {
  local TAG="$1"; shift
  echo "=============================================================="
  echo "[dyn] $TAG  ($(date -u +%H:%M:%S))  env: $*"
  echo "=============================================================="
  pkill -f "sglang.launch_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null || break; sleep 5; done
  sleep 10

  env "$@" MEM_FRACTION=0.85 GPU_EXPERTS=96 \
    nohup bash experiments/expert_tiering_ssd/boot_tiered.sh > "logs/dyn_${TAG}.log" 2>&1 &

  local T0=$(date +%s) READY=0
  for _ in $(seq 1 480); do
    if curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1; then READY=1; break; fi
    pgrep -f "boot_tiered.sh|sglang.launch_server" >/dev/null || { echo "[dyn] $TAG died"; break; }
    sleep 5
  done
  local BOOT=$(( $(date +%s) - T0 ))
  if [ "$READY" != "1" ]; then
    echo "[dyn] $TAG FAILED TO BOOT after ${BOOT}s"; tail -30 "logs/dyn_${TAG}.log"; return
  fi
  echo "[dyn] $TAG ready in ${BOOT}s"

  # Export the same knobs so the harnesses label their rows correctly.
  for kv in "$@"; do export "$kv"; done
  export GPU_EXPERTS=96
  export KT_HOTCORE_RANKING_PT="$REPO/experiments/adaptive_expert_cache/decode_cache/hot_core_ranking.pt"

  # Three passes, not two: dynamic placement needs traffic to converge, and the
  # third pass is what a converged cache actually delivers.
  .venv/bin/python experiments/expert_tiering_ssd/tier_bench.py "$TAG" 3 400 llm 2>&1 | tail -22
  .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py reference "$OUTDIR/${TAG}.json" 2>&1 | tail -22
  grep -cE "\[kt-energy\]|\[kt-tier\]" "logs/dyn_${TAG}.log" | sed "s/^/[dyn] $TAG movement log lines: /"
  grep -E "\[kt-energy\]" "logs/dyn_${TAG}.log" | tail -3
  echo "[dyn] $TAG done boot=${BOOT}s"
}

run_one r160_tick_on  KT_RAM_EXPERTS=160 KT_TIER_DYNAMIC=0 KT_ENERGY=0
run_one r32_static    KT_RAM_EXPERTS=32  KT_TIER_DYNAMIC=0 KT_ENERGY=0
run_one r32_count     KT_RAM_EXPERTS=32  KT_TIER_DYNAMIC=1 KT_ENERGY=0
run_one r32_energy    KT_RAM_EXPERTS=32  KT_TIER_DYNAMIC=0 KT_ENERGY=1

echo "[dyn] sweep complete"
