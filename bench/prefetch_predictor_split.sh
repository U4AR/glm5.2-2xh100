#!/usr/bin/env bash
# Where does the predictor's 5.03 ms/step actually go?
#
# The gather is fixed (84.67 -> 72.97 ms full path). What is left between the
# prefetcher and break-even is 6.4 ms, and 5.03 of it is the predictor -- which
# is NOT arithmetic. Per layer it is a gate GEMM on at most 8 tokens plus the
# full topk scoring path plus a top-P gather plus an inbox copy, and then
# pf_issue's ~20 small tensor ops (demand accumulate, residency mask, topk over
# 256, select, three scatters, four counter updates). Call it ~30 kernels x 75
# layers = ~2200 extra graph nodes at a couple of microseconds each. That is
# the shape of 5 ms.
#
# Fusing it is worthwhile but it is real work, so measure which half to fuse
# first. KT_PREFETCH_SELECT=0 runs the lookahead router and returns from
# pf_issue before selecting anything, which splits the two cleanly:
#
#   baseline (no predictor at all)              66.64 ms
#   router only        (this run)                    ?
#   router + select    (P0, already measured)   71.67 ms
#
# So router = (this run - 66.64), select ops = (71.67 - this run).
#
# NOTE ON THE OTHER IDEA: "just reuse what the previous token routed" is
# already refuted by this project's own instrument -- EXPERT_PREFETCH_PLAN.md
# measures persistence at 27.1% whole-layer coverage against the lookahead
# router's 75.2%. It would trade ~4 ms of predictor for ~6 ms of lost CPU
# saving. Not worth a boot.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/prefetch_rate.json

boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" \
    GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/psplit_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/psplit_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
}

# Re-establish the baseline in THIS box state, so the split is not read against
# a number measured hours and several reboots ago.
boot PSPLIT-baseline    KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
# Lookahead router runs; pf_issue returns immediately. vs baseline and vs 71.67.
boot PSPLIT-router-only KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 KT_PREFETCH_SELECT=0 \
                        KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/psplit_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 2 --tokens 200 \
  --label PSPLIT-prod-restore --out "$OUT" 2>&1 | tail -3
echo "=== predictor split done ==="
