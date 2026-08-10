#!/usr/bin/env bash
# The row that might be the whole answer: FLOOR SPEED, no prefetch, full routing.
#
# `E0-top8` measured **53.40 ms/step** -- identical to its own top0 (53.35) and to
# the 53.31 floor. GPU-only mode with the prefetch switched OFF keeps every
# genuine expert that happens to be GPU-RESIDENT and substitutes the rest. No CPU
# round-trip, no PCIe transfer, so it runs at floor speed by construction.
#
# What makes it interesting is that it is NOT top0. It keeps ~3.2 of the 8 genuine
# experts per token (the resident ~40%) where top0 keeps none, and it does so for
# zero milliseconds. `safe2` keeps 2 -- the top two BY WEIGHT -- and pays 11.88 ms
# of CPU to do it, while throwing away resident genuine experts at ranks 2..7
# because `s.scatter_(1, ids, neg)` bars an originally-selected id from the fill.
#
# THE QUESTION IS RANK vs COUNT. Is keeping the true top-1 (expensive, one expert)
# worth more than keeping 3.2 mid-rank experts (free)? Router weight is skewed, so
# it is genuinely unclear, and only the output settles it.
#
# Tiers swept because in this mode K controls how many genuine experts are
# ELIGIBLE, and eligibility is free -- top8 should dominate top2 at identical
# speed. top0 is the calibration case and must score degenerate or nothing counts.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

wait_for_marker "=== exposure_rootcause done ===" exposure.out

bash bench/_kill_servers.sh >/dev/null
echo "=== booting R-only: GPU-only, prefetch OFF (the E0 configuration) ==="
KT_GPU_ONLY=1 KT_PRED_FUSED=0 \
KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0 \
KT_PREFETCH_SELECTIVE=0 GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_PREFETCH_SLOTS=4 \
KT_STORE_SHM=1 MTP=1 MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ronly.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/ronly.log" && { echo BOOTFAIL; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; tail -25 "$SP/ronly.log"; exit 1; }

echo "--- speed, confirming the floor at every tier ---"
for t in 8 4 2 0; do
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
    --tokens 200 --label "Ronly-top$t" --out bench/profile_out/ronly_rate.json 2>&1 \
    | grep -a "ms/step" || true
done

echo "--- coherence, 1200 tokens, top0 as the calibration case ---"
TOKENS=1200 bash bench/coherence_run.sh ronly 8 4 2 0

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ronly_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== resident_only done ==="
