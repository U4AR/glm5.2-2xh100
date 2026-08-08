#!/usr/bin/env bash
# TODO item 3: what is the 3.1 ms between CPF-d1 (70.43) and DEEP-d1 (73.54)?
#
# The counters cannot answer it -- `wanted/call` is the size of the PREDICTED
# non-resident set, not the layer's residual demand -- and the two rows are
# different code paths with different fixed costs. So separate the two terms
# directly: boot each predictor with the fetch machinery OFF (GATHER=0 ROUTE=0
# CPUSKIP=0). The router still runs, the selection ops still run, nothing moves.
#
#   PRED-direct-off - DEEP-baseline  = the direct predictor's fixed cost
#   PRED-chain-off  - DEEP-baseline  = the chain path's fixed cost at depth 1
#   the rest of the 3.1 ms           = the fetch effect (bytes + CPU skips)
#
# Waits for the accuracy run so the two never share the GPU.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/prefetch_rate.json

while pgrep -f "bash bench/drop_acc.sh" >/dev/null 2>&1 ||
      pgrep -f "bash bench/chain_predict_run.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== accuracy run finished, starting item-3 attribution ==="
sleep 20

boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/i3_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/i3_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -3
}
OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"

boot I3-base       $OFF
boot I3-direct-off $OFF KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1
boot I3-chain-off  $OFF KT_CHAIN_PF=1 KT_CHAIN_PF_DEPTH=1

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/i3_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== item3 done ==="
