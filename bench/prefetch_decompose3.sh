#!/usr/bin/env bash
# Decomposition, take 2 -- this time with the predictor actually running.
#
# Takes 1 and 2 were void: pf_issue is only reachable from _kt_pred_emit, which
# is gated on _KT_PRED_DEPTHS (env KT_PRED_LOOKAHEAD, default empty). Every one
# of those seven boots allocated the slots, put the store in shm, pinned it, and
# then never predicted or fetched anything. They agree to within 0.7 ms because
# they were the same configuration seven times. Their one real product is a
# noise floor for this box state: 66.05-66.74 ms/step.
#
# KT_PRED_LOOKAHEAD=1 is what turns the thing on. The counter line
# "[kt-prefetch] step N: X fetched/call" MUST appear or the row is void again.
#
#   P0  predictor ON, gather OFF          the predictor's own cost (75 extra
#                                         gate GEMMs + topk per forward)
#   P2  + gather + barrier, route OFF     the cost of the bytes
#   P3  + route + cpuskip                 the shipped path, numerically exact
#   P3b route OFF, cpuskip ON             what the CPU saves by dropping the
#                                         experts, with nothing replacing them.
#                                         Contribution missing -> WRONG TEXT on
#                                         purpose; this is the entire upside.
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
    KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 \
    GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/dec_$label.log" 2>&1 &
  for i in $(seq 1 240); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/dec_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/dec_$label.log" | tail -1 || echo "NONE -- ROW IS VOID"
}

boot PRED2-P2-bytes-noroute KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot PRED2-P3-full          KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1
boot PRED2-P0-nogather      KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot PRED2-P3b-cpuskiponly  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=1
echo "=== take-3 decomposition done ==="
