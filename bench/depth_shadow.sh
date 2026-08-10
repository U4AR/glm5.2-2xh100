#!/usr/bin/env bash
# Buy shadow instead of accuracy, and stop over-fetching.
#
# The prefetch's 11.00 ms overhead is 1.76 predictor + 9.24 exposed gather. The
# gather is exposed because a layer's transfer needs ~325 us (1.78 experts x
# 9.6 MiB at ~55 GB/s = PCIe link speed) while depth 1 offers only ~200 us of
# cover -- the rest of layer L's MoE plus layer L+1's attention. The difference,
# ~125 us x 74 layers, is paid at wait_event.
#
# Two independent ways to close that gap, neither previously measured:
#
#   MORE SHADOW (S-d2).  Predict two layers ahead. The gather then has layer L's
#   MoE, all of layer L+1, and layer L+2's attention to hide in -- ~1060 us
#   against a 325 us transfer, so it should vanish from the critical path. Costs
#   coverage (~74% -> ~63% whole-layer), which pushes work back to the CPU. That
#   is affordable only because the CPU residual is currently 0.21 ms. Depth was
#   dismissed earlier on ACCURACY alone, which was the wrong frame: the payoff
#   needs accuracy AND hiding, and hiding was never measured. Unlike the chain
#   walk (17.45 ms/step per layer walked) this costs the direct predictor nothing
#   -- same single gate GEMM, further router.
#
#   FEWER BYTES (S-p1).  Fetch one expert per layer instead of 1.78. 183 us fits
#   inside the 200 us shadow, so the transfer hides at depth 1, and the ~1.2
#   experts left over go to the CPU where they overlap with GPU work anyway.
#   Arithmetic says ~98 us/layer against the current ~125.
#
#   S-d2p1 is both at once.
#
# S-base and S-d1 bracket everything: S-d1 must reproduce yesterday's 63.89, or
# the box moved and nothing else in the table can be read.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/depth_shadow_rate.json
DET=bench/profile_out/depth_shadow_det.json

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/s_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/s_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 3 --tokens 200 --tier 2 \
    --label "$label" --out "$DET" 2>&1 | grep -aE "distinct|DETERMIN" || true
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/s_$label.log" | tail -1 || echo none
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"
OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"

boot S-base  $OFF
boot S-d1    $ON $FUSED KT_PRED_FUSED_DEPTH=1
boot S-d2    $ON $FUSED KT_PRED_FUSED_DEPTH=2
boot S-p1    $ON $FUSED KT_PRED_FUSED_DEPTH=1 KT_PRED_P=1
boot S-d2p1  $ON $FUSED KT_PRED_FUSED_DEPTH=2 KT_PRED_P=1

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/s_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== depth_shadow done ==="
