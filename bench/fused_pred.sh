#!/usr/bin/env bash
# Does folding the predictor into the network get it under a millisecond?
#
# The predictor costs 5.47 ms/step = 73 us/layer, and none of it is arithmetic:
# ~36 kernel launches per layer on four-token tensors (Stage H7). Two changes:
#
#   1. The lookahead gate GEMM is folded into the layer's OWN gate. The 75 gate
#      weights are re-laid contiguously at load time and each layer takes a
#      512-row slice, so one F.linear yields this layer's logits and the next
#      layer's. No extra launch, no extra memory -- only ~0.08 ms/step of extra
#      HBM for the wider read.
#   2. Everything after the GEMM -- scoring, top-8, top-P, residency intersect,
#      demand histogram, slot pick, publishing sel/landed/index/stats -- becomes
#      ONE kernel that reads the lookahead half as a column slice in place.
#
# Three launches per layer instead of ~36. Predicted cost: ~0.5 ms/step.
#
# Verified before booted: the kernel is checked against a transcription of the
# shipped path first, and the ladder only runs if that passes.
#
# Rows, ALL in one configuration so every subtraction is within-config:
#   FZ-base       no predictor, no gather      -- the floor
#   FZ-fused-off  fused predictor, gather off  -- the predictor's fixed cost,
#                                                 directly against I3-direct-off's
#                                                 +5.47 ms for the unfused one
#   FZ-fused-on   everything on                -- the shippable number
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/fused_pred.json

bash bench/_kill_servers.sh
echo "=== verifying the fused kernel against the shipped path ==="
.venv/bin/python bench/pred_fused_kernel.py --verify --trials 150 > "$SP/fz_verify.txt" 2>&1
tail -5 "$SP/fz_verify.txt"
grep -q "VERIFY OK" "$SP/fz_verify.txt" || { echo "VERIFY FAILED -- not booting"; exit 1; }
echo "=== benching it standalone ==="
.venv/bin/python bench/pred_fused_kernel.py --bench 2>&1 | tail -8

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/fz_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/fz_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  grep -a "kt-pred-fused" "$SP/fz_$label.log" | head -1
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$OUT" 2>&1 | tail -2
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/fz_$label.log" | tail -1 || echo none
}

OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"

boot FZ-base      $OFF
boot FZ-fused-off $OFF KT_PRED_FUSED=1 KT_PRED_POINT=pre
boot FZ-fused-on  $ON  KT_PRED_FUSED=1 KT_PRED_POINT=pre

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/fz_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== fused_pred done ==="
