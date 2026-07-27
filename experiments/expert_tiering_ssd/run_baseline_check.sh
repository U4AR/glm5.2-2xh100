#!/usr/bin/env bash
# Is there a decode regression vs the pre-change build? Measured with BOTH
# harnesses, strictly sequentially (never two requests in flight: with
# CUDA_GRAPH_MAX_BS=1 a second concurrent request drops the batch out of the
# captured graph and both crawl -- that is how a 43 tok/s pass got recorded
# as 17).
#
#   decbench.py     raw /generate, non-streaming. The harness the ~40.5 tok/s
#                   README headline was measured with.
#   tier_bench.py   /v1/chat/completions, streaming. Reads a few tok/s lower
#                   on the same server -- README: "~33 tok/s" for the same
#                   build decbench called 40.5.
#
# r160 = nothing on SSD = functionally the shipped two-tier build, so it is the
# baseline row. r32 is the tiered configuration.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

run() {
  local TAG="$1"; shift
  echo "=============================================================="
  echo "[base] $TAG ($(date -u +%H:%M:%S)) : $*"
  pkill -f "sglang.launch_server" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "sglang.launch_server" >/dev/null || break; sleep 5; done
  sleep 10
  env "$@" GPU_EXPERTS=96 MEM_FRACTION=0.85 \
    nohup bash experiments/expert_tiering_ssd/boot_tiered.sh > "logs/base_${TAG}.log" 2>&1 &
  for _ in $(seq 1 480); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    pgrep -f "boot_tiered.sh|sglang.launch_server" >/dev/null || { echo "[base] $TAG died"; return; }
    sleep 5
  done
  echo "[base] $TAG ready"
  for kv in "$@"; do export "$kv"; done
  export GPU_EXPERTS=96
  echo "--- decbench (raw /generate, the README headline harness) ---"
  .venv/bin/python bench/perf_probe/decbench.py 300 4 2>&1 | tail -5
  echo "--- tier_bench (chat streaming) ---"
  .venv/bin/python experiments/expert_tiering_ssd/tier_bench.py "${TAG}_chat" 3 400 llm 2>&1 | grep -E "pass "
  echo "--- server-side accept len ---"
  grep -o "accept len: [0-9.]*" "logs/base_${TAG}.log" | awk '{s+=$3;n++} END{if(n)printf "  mean accept len %.3f over %d decode batches\n",s/n,n}'
}

run r160_fill_gpu KT_RAM_EXPERTS=160 KT_TIER_FILL_POOL=gpu
run r32_fill_gpu  KT_RAM_EXPERTS=32  KT_TIER_FILL_POOL=gpu
echo "BASELINE CHECK complete"
