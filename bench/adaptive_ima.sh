#!/usr/bin/env bash
# Root-cause the illegal memory access that kills adaptive + prefetch together.
#
# Symptom: AG_B booted fine, served ~78 tokens, logged three cache ticks
# (layer=6,7,8, swaps=8, took=11-62ms -- so the stable-slot swap itself works)
# and then died with cudaErrorIllegalAddress. The reported stack is in
# `prepare_for_extend_to_fill_draft_kvcache`, which means nothing: an IMA is
# raised at the next synchronising call, not where it happened.
#
# Three candidates, and they are NOT ordered by suspicion -- they are ordered so
# that whichever answer arrives first excludes the most:
#
#   C  adaptive ALONE (no predictor, no gather, no slots)
#      Does the cache work at all on this build? If C dies, nothing about the
#      prefetch is implicated and the combination is a red herring.
#
#   A  adaptive + predictor, GATHER=0 (nothing is read from the kt store)
#      The gather reads expert weights straight out of the kt CPU store through
#      registered host pointers cached in pf_table at boot. A residency swap
#      calls into the kt wrapper (update_kt_wrapper_masks) and rebuilds a
#      loader; if that re-maps or frees the store's pages, every pf_table entry
#      is dangling and the next gather touches unmapped host memory. That is
#      exactly an async IMA, and it can only happen when both features run --
#      which no run before this one has done.
#
#   B  adaptive + full prefetch, but KT_ADAPTIVE_INCREMENTAL=0
#      Falls back to the full restage. Tests MY change (the stable-slot swap
#      newly wired into the adaptive tick) rather than the interaction. Slow --
#      the full path re-copies ~100 experts per swapping layer -- but it only
#      has to survive, not be fast.
#
# Survival, not speed, is the reading. Each arm generates enough tokens to cover
# ~12 ticks; a row that completes is a row that did not crash.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

MODEFILE="$SP/kt_topk_mode.adaptive_ima"
echo safe8 > "$MODEFILE"
OUT=bench/profile_out/adaptive_ima.json
mkdir -p bench/profile_out

# Short period so the ticks start early and the arm is exercised quickly.
PERIOD="${KT_ADAPTIVE_PERIOD:-32}"

boot () {
  local log="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $log: $* ==="
  env "$@" \
    KT_GPU_ONLY=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
    KT_PREFETCH_SELECTIVE=0 KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
    KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_MAX_SWAP=8 KT_ADAPTIVE_PERIOD="$PERIOD" \
    KT_ADAPTIVE_LAYERS_PER_TICK=2 KT_TIER_COUNT_MODE=top8 \
    GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_STORE_SHM=1 MTP=1 \
    MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/$log.log" 2>&1 &
  local i
  for i in $(seq 1 400); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed|Bus error" "$SP/$log.log" && { echo "BOOTFAIL $log"; tail -20 "$SP/$log.log"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $log"; return 1; }
  return 0
}

probe () {   # probe <label> : survive N tokens across many ticks
  local label="$1"
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 \
    --tokens 500 --label "$label" --out "$OUT" 2>&1 | grep -a "ms/step" \
    || echo "$label: REQUEST FAILED"
  local ticks
  ticks=$(grep -ac "kt-adaptive] layer=" "$SP/$2.log" 2>/dev/null || echo 0)
  if grep -qa "illegal memory access" "$SP/$2.log"; then
    echo "$label: IMA after $ticks logged swaps"
  else
    echo "$label: SURVIVED, $ticks logged swaps"
  fi
}

# C -- adaptive alone. Prefetch entirely absent (SLOTS=0 leaves the path off).
if boot AI_C KT_PRED_FUSED=0 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 \
             KT_PREFETCH_CPUSKIP=0 KT_PREFETCH_SLOTS=0; then
  probe AI-C-adaptive-only AI_C
fi

# A -- predictor runs, publishes, routes; nothing is read from the kt store.
if boot AI_A KT_PRED_FUSED=1 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=1 \
             KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_SLOTS=1 KT_PREFETCH_REUSE=2; then
  probe AI-A-nogather AI_A
fi

# B -- the full pipeline, full restage instead of the stable-slot swap.
if boot AI_B KT_PRED_FUSED=1 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 \
             KT_PREFETCH_CPUSKIP=1 KT_PREFETCH_SLOTS=1 KT_PREFETCH_REUSE=2 \
             KT_ADAPTIVE_INCREMENTAL=0; then
  probe AI-B-fullrestage AI_B
fi

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ai_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== adaptive_ima done ==="
