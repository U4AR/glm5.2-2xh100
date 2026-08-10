#!/usr/bin/env bash
# WHERE do the 7.8 ms between the 53.31 floor and 61.09 actually go?
#
# Known, measured: it is not lead time. Depth 1/2/3 read 61.63/61.09/61.46 --
# flat and non-monotone -- and a layer at 61 ms/step over 75 layers is ~820 us
# against a ~500 us transfer, so the bytes already fit inside one layer's shadow.
# Whatever costs the time happens WHILE the transfer runs, not before it starts.
#
# Three candidates and NO evidence yet distinguishing them:
#   (a) CONTENTION  -- the gather's HBM traffic slows the MoE GEMMs, or vice versa
#   (b) BARRIER     -- apply() blocks the main stream on pf_done every layer
#   (c) SM STARVATION -- the 8-block gather cannot get SMs while cutlass holds them
#
# THE ABLATION. Every row is the SAME configuration with ONE thing removed, all
# measured at TIER 0 where the prefetch can save nothing, so every millisecond
# above the floor is pure cost and the subtractions are attributions:
#
#   E0  nothing on                          floor, with slots ALLOCATED
#   E1  + predictor, gather off             E1-E0 = predictor
#   E2  + gather, WAIT=0 ROUTE=0            E2-E1 = CONTENTION ("the bytes still
#                                                   move; the barrier disappears")
#   E3  + wait,   WAIT=1 ROUTE=0            E3-E2 = BARRIER / "arrived late"
#   E4  + route   (the shipped path)        E4-E3 = the mask/index bookkeeping
#   E5  E4 but KT_PREFETCH_BLOCKS=1         grid pressure down 8x
#   E6  E4 but KT_PREFETCH_BLOCKS=32        grid pressure up 4x
#
# E4/E5 test (c) directly: if the cost tracks the gather's grid size, it is
# fighting cutlass for SMs; if it is flat in blocks, it is not.
#
# VRAM is identical in every row (100 resident + 4 slots) so no row can be
# explained by residency. Tier 0 also means keep_mask is all-False, so the
# ROUTE=0 that KT_PREFETCH_WAIT=0 forces cannot change what is computed --
# which is what makes the E2 probe legitimate here and not elsewhere.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/exposure_rate.json
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
    GPU_EXPERTS=100 KT_PREFETCH_SLOTS=4 KT_PRED_P=8 \
    KT_STORE_SHM=1 MTP=1 MEM_FRACTION=0.94 KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/e_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/e_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  for t in 0 8; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "$label-top$t" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  done
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/e_$label.log" | tail -1 || echo none
}

P="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1"

# ONE variable per step. E2->E3 must change ONLY the wait: KT_PREFETCH_WAIT=0
# forces ROUTE=0 internally, so E3 pins ROUTE=0 too. E4 then adds ROUTE back to
# reach the shipped path, and E5/E6 vary only the grid.
boot E0 KT_PRED_FUSED=0 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot E1 $P              KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot E2 $P              KT_PREFETCH_GATHER=1 KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_WAIT=0 KT_PREFETCH_BLOCKS=8
boot E3 $P              KT_PREFETCH_GATHER=1 KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_WAIT=1 KT_PREFETCH_BLOCKS=8
boot E4 $P              KT_PREFETCH_GATHER=1 KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_WAIT=1 KT_PREFETCH_BLOCKS=8
boot E5 $P              KT_PREFETCH_GATHER=1 KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_WAIT=1 KT_PREFETCH_BLOCKS=1
boot E6 $P              KT_PREFETCH_GATHER=1 KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_WAIT=1 KT_PREFETCH_BLOCKS=32

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/e_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== exposure_rootcause done ==="
