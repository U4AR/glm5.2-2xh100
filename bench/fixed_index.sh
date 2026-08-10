#!/usr/bin/env bash
# The landing-slot index was RELATIVE; it had to be ABSOLUTE.
#
# `pf_index` is merged with `logical_to_gpu_index` in apply(), and that table
# holds absolute indices into the layer's 104-expert cutlass tensors, whose last
# four entries are the landing slots. The fused kernel wrote the relative slot
# number (0..3), so every prefetched expert was computed as GPU expert 0..3 --
# four real, resident, entirely unrelated experts.
#
# That is why omitting a landed expert beat computing it (accept 2.532 vs 2.273):
# a wrong expert is worse than no expert. The gather was never at fault -- its
# bytes match the checkpoint exactly, resident and gathered alike, on both ranks
# and both sampled layers.
#
# The unfused pf_issue always had this right (`pf_slot_ids` is
# arange(base, base+slots)). Only the fused kernel was wrong, and its own
# --verify asserted the relative convention, so kernel and test agreed with each
# other and 150 trials passed while the server routed to the wrong experts.
#
#   Y-base  prefetch off              the floor
#   Y-full  prefetch on, index fixed  accept should return toward 2.857; what is
#                                     left over baseline is then the honest cost
#                                     of computing an expert on the GPU (fp8
#                                     activations) instead of the CPU (int8)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DET=bench/profile_out/fixed_index_det.json
RATE=bench/profile_out/fixed_index_rate.json
TXT=bench/profile_out/fixed_index_txt
rm -rf "$TXT"; mkdir -p "$TXT"

echo "=== verifying the fused kernel (absolute slot index) ==="
.venv/bin/python bench/pred_fused_kernel.py --verify --trials 150 2>&1 | tail -3

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/y_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/y_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 3 --tokens 200 --tier 2 \
    --label "$label" --out "$DET" --save-text "$TXT" 2>&1 | tail -8
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$RATE" 2>&1 | tail -2
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/y_$label.log" | tail -1 || echo none
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"

boot Y-base KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot Y-full KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 $FUSED

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/y_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== fixed_index done ==="
