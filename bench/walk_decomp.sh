#!/usr/bin/env bash
# TODO_ROOT_CAUSES item 1 -- take the walk's 18.2 ms/step APART.
#
# The capture-safe chain at depth 2 costs 18.2 ms/step more than at depth 1: one
# extra approximate layer per real layer. Widening 2 -> 8 expert slots added only
# 3.4 ms, so the expert GEMM is the SMALL term and ~18 ms is something else.
# NOTHING about that is attributed yet, so it gets taken apart rather than
# explained. KT_CHAIN_PF_STAGE turns the components on cumulatively:
#
#   s0  chain off entirely                       (reference)
#   s1  hook + target prediction, NO walk        -> the "free" half
#   s2  + the walked layer's gate and topk
#   s3  + resident substitution, id remap, slot mask
#   s4  + the GPU expert kernel
#   s5  + the shared expert
#   s6  + the TP all-reduce                      (= the real thing)
#
# Every row runs with the prefetch's EFFECT off (GATHER=0 ROUTE=0 CPUSKIP=0), so
# a stage's degraded prediction cannot feed back into coverage and move the step
# time. Then every row-to-row difference is walk COST and nothing else.
#
# s0 is measured TWICE, first and last. Without a boot-to-boot noise figure a
# 2 ms gap between adjacent stages cannot be called a cause, and calling causes
# without evidence is exactly what this list exists to stop.
#
# PRIME SUSPECT: s6 - s5, i.e. 75 extra NCCL all-reduces per step on a 4-token
# tensor. Latency-bound collectives at ~30-50 us would be 2-4 ms on their own,
# and the walk does not actually need the reduce -- it only feeds a router.
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
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/walk_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/walk_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -3
}

boot WALK-s0 KT_CHAIN_PF=0
for st in 1 2 3 4 5 6; do
  boot "WALK-s$st" KT_CHAIN_PF=1 KT_CHAIN_PF_DEPTH=2 KT_CHAIN_PF_K=2 KT_CHAIN_PF_STAGE=$st
done
boot WALK-s0b KT_CHAIN_PF=0          # noise floor: how far does s0 drift alone?

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/walk_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== walk decomp done ==="
