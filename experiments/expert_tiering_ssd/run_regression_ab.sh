#!/usr/bin/env bash
# Root-cause A/B for the decode regression. Identical residency (RAM=160 =>
# nothing on SSD => the model is exactly the shipped two-tier build); the ONLY
# difference is which experts may fill a substituted slot.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run() {
  local TAG="$1"; shift
  echo "===== $TAG ($(date -u +%H:%M:%S)) : $* ====="
  pkill -f "sglang.launch_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null || break; sleep 5; done
  sleep 10
  env "$@" GPU_EXPERTS=96 MEM_FRACTION=0.85 \
    nohup bash experiments/expert_tiering_ssd/boot_tiered.sh > "logs/ab_${TAG}.log" 2>&1 &
  for _ in $(seq 1 480); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    pgrep -f "boot_tiered.sh|sglang.launch_server" >/dev/null || { echo "$TAG died"; return; }
    sleep 5
  done
  for kv in "$@"; do export "$kv"; done
  export GPU_EXPERTS=96
  .venv/bin/python experiments/expert_tiering_ssd/tier_bench.py "$TAG" 3 400 llm 2>&1 | grep -E "pass |tok_s\"|rss"
}
run ab_r160_fill_gpu       KT_RAM_EXPERTS=160 KT_TIER_FILL_POOL=gpu
run ab_r160_fill_resident  KT_RAM_EXPERTS=160 KT_TIER_FILL_POOL=resident
echo "AB complete"
