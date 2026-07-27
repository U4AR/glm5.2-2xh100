#!/usr/bin/env bash
# Sweep the three-tier ladder. For each config: boot, measure decode tok/s and
# host RSS, then run the accuracy eval and save its raw outputs so every config
# can be diffed against whichever run we pick as the reference.
#
# Usage: run_ladder.sh "<ram_experts list>" [gpu_experts] [fill_pool] [count_mode]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

RAM_LIST="${1:-160 96 64 32 16 8}"
GPU_N="${2:-96}"
FILL="${3:-resident}"
COUNT="${4:-top2}"
OUTDIR="experiments/expert_tiering_ssd/runs"
mkdir -p "$OUTDIR" logs

for RAM_N in $RAM_LIST; do
  TAG="g${GPU_N}_r${RAM_N}_${FILL}_${COUNT}"
  echo "=============================================================="
  echo "[ladder] $TAG  ($(date -u +%H:%M:%S))"
  echo "=============================================================="

  pkill -f "sglang.launch_server" 2>/dev/null
  # Wait for the old scheduler to actually release its host RAM and VRAM;
  # booting on top of a dying process OOMs or silently mis-measures RSS.
  for _ in $(seq 1 60); do
    pgrep -f "sglang.launch_server" >/dev/null || break
    sleep 5
  done
  sleep 10

  GPU_EXPERTS=$GPU_N KT_RAM_EXPERTS=$RAM_N KT_TIER_FILL_POOL=$FILL \
    KT_TIER_COUNT_MODE=$COUNT MEM_FRACTION=0.85 \
    nohup bash experiments/expert_tiering_ssd/boot_tiered.sh > "logs/tier_${TAG}.log" 2>&1 &

  BOOT_START=$(date +%s)
  READY=0
  for _ in $(seq 1 480); do   # up to 40 min: the full-RAM point stages all 256
    if curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1; then
      READY=1; break
    fi
    pgrep -f "sglang.launch_server" >/dev/null || { echo "[ladder] $TAG process died"; break; }
    sleep 5
  done
  BOOT_S=$(( $(date +%s) - BOOT_START ))
  if [ "$READY" != "1" ]; then
    echo "[ladder] $TAG FAILED TO BOOT after ${BOOT_S}s; skipping"
    tail -30 "logs/tier_${TAG}.log"
    continue
  fi
  echo "[ladder] $TAG ready in ${BOOT_S}s"

  export GPU_EXPERTS=$GPU_N KT_RAM_EXPERTS=$RAM_N KT_TIER_FILL_POOL=$FILL KT_TIER_COUNT_MODE=$COUNT
  export KT_HOTCORE_RANKING_PT="$REPO/experiments/adaptive_expert_cache/decode_cache/hot_core_ranking.pt"
  export TIER_BOOT_SECONDS=$BOOT_S

  .venv/bin/python experiments/expert_tiering_ssd/tier_bench.py "$TAG" 2 400 llm 2>&1 | tail -25
  .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py reference "$OUTDIR/${TAG}.json" 2>&1 | tail -25
  echo "[ladder] $TAG done boot=${BOOT_S}s"
done

echo "[ladder] sweep complete"
