#!/usr/bin/env bash
# Where does the 64.35 ms decode step actually go, now that the path is correct?
#
# Four boots. Every subtraction happens WITHIN one configuration -- the rule this
# stage learned the hard way when a baseline agreeing across boots was mistaken
# for a licence to subtract rows measured beside it.
#
#   A-base  everything off                floor, and the tier sweep
#   A-pred  predictor on, gather off      the predictor's fixed cost, in THIS
#                                         config rather than remembered from a
#                                         ladder three hours ago
#   A-full  everything on                 the shipped number, plus the torch
#                                         profiler trace (real kernel intervals,
#                                         not an estimate) and the prefetch
#                                         counters
#   A-score prediction quality            KT_PLACE_SCORE=1. This inflates the
#                                         step badly (it scores 4-9 cells per
#                                         layer per step), so its TIMINGS ARE
#                                         MEANINGLESS and only its coverage
#                                         numbers are read.
#
# The tier sweep is per-request (model name -topN), so top8/top2/top0 come from
# ONE boot and their differences are clean: top0 removes the CPU expert path
# entirely, top8 exposes all of it, top2 is what ships.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/anatomy_rate.json
OUT=bench/profile_out/anatomy

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/a_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/a_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  return 0
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"
OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"

# ---- floor + the tier sweep -------------------------------------------------
if boot A-base $OFF; then
  for t in 8 2 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "A-base-top$t" --out "$RATE" 2>&1 | tail -1
  done
fi

# ---- the predictor's fixed cost, same config --------------------------------
if boot A-pred $OFF $FUSED; then
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "A-pred" --out "$RATE" 2>&1 | tail -1
fi

# ---- the shipped path, with a real kernel trace -----------------------------
if boot A-full $ON $FUSED; then
  for t in 8 2 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "A-full-top$t" --out "$RATE" 2>&1 | tail -1
  done
  echo "--- torch profiler trace (real kernel intervals) ---"
  .venv/bin/python bench/profile_occupancy.py --no-stream --no-report \
    --model GLM5.2-top2 --tokens 200 --torch-tokens 120 --torch-steps 200 \
    --out "$OUT" 2>&1 | tail -25
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/a_A-full.log" | tail -1 || echo none
fi

# ---- prediction quality (timings here are junk by construction) -------------
if boot A-score $ON $FUSED KT_PLACE_SCORE=1 KT_PRED_DEPTHS=1 KT_PRED_P=1,2,3,4 \
      KT_PRED_DUMP_PT="$PWD/bench/profile_out/anatomy_pred.pt"; then
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 1 \
    --tokens 200 --label "A-score" --out "$RATE" 2>&1 | tail -1
  echo "--- prediction stats ---"
  grep -aE "kt-pred|coverage|PRED" "$SP/a_A-score.log" | tail -20 || echo none
fi

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/a_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== step_anatomy done ==="
