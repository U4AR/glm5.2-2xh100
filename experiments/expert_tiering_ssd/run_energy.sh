#!/usr/bin/env bash
# Energy-driven placement vs the static split, at a fixed RAM budget, with the
# fill pool corrected to `gpu` (the shipped substitution contract).
#
#   r32_static      control: frozen tiers.
#   r32_energy      energy model moves the RAM tier only (SSD<->RAM).
#   r32_energy_gpu  energy model also allowed to restage the GPU tier, which is
#                   what the "resident on the next token" claim needs.
#
# Each run reports the responsiveness measurement the server accumulates:
# for the top-1 expert per layer, how many decode steps pass between it going
# hot and it actually being resident, and how often that is <= 1 step.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR=experiments/expert_tiering_ssd/runs
mkdir -p "$OUTDIR" logs

run() {
  local TAG="$1"; shift
  echo "=============================================================="
  echo "[energy] $TAG ($(date -u +%H:%M:%S)) : $*"
  echo "=============================================================="
  pkill -f "sglang.launch_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null || break; sleep 5; done
  sleep 10
  rm -f /tmp/kt_energy_report.pt
  env "$@" GPU_EXPERTS=96 MEM_FRACTION=0.85 \
    nohup bash experiments/expert_tiering_ssd/boot_tiered.sh > "logs/en_${TAG}.log" 2>&1 &
  local T0=$(date +%s) READY=0
  for _ in $(seq 1 480); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && { READY=1; break; }
    pgrep -f "boot_tiered.sh|sglang.launch_server" >/dev/null || { echo "[energy] $TAG died"; break; }
    sleep 5
  done
  [ "$READY" = "1" ] || { echo "[energy] $TAG FAILED"; tail -25 "logs/en_${TAG}.log"; return; }
  echo "[energy] $TAG ready in $(( $(date +%s) - T0 ))s"
  for kv in "$@"; do export "$kv"; done
  export GPU_EXPERTS=96
  export KT_HOTCORE_RANKING_PT="$PWD/experiments/adaptive_expert_cache/decode_cache/hot_core_ranking.pt"

  .venv/bin/python experiments/expert_tiering_ssd/tier_bench.py "$TAG" 3 400 llm 2>&1 | grep -E "pass |tok_s\"|rss"
  .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py reference "$OUTDIR/${TAG}.json" 2>&1 | tail -20
  echo "--- movement + responsiveness ---"
  grep -E "\[kt-energy\]" "logs/en_${TAG}.log" | tail -2
  .venv/bin/python - <<'PY'
import os, torch
p = "/tmp/kt_energy_report.pt"
if os.path.exists(p):
    r = torch.load(p, weights_only=False)
    print("responsiveness:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
else:
    print("responsiveness: no report (energy placement not active this run)")
PY
  echo "[energy] $TAG done"
}

run r32_static     KT_RAM_EXPERTS=32 KT_TIER_FILL_POOL=gpu KT_ENERGY=0 KT_TIER_DYNAMIC=0
run r32_energy     KT_RAM_EXPERTS=32 KT_TIER_FILL_POOL=gpu KT_ENERGY=1 KT_ENERGY_GPU=0
run r32_energy_gpu KT_RAM_EXPERTS=32 KT_TIER_FILL_POOL=gpu KT_ENERGY=1 KT_ENERGY_GPU=1
echo "ENERGY sweep complete"
