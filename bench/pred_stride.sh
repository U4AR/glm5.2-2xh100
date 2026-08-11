#!/usr/bin/env bash
# Can the full pipeline -- cache AND prefetch both live -- reach 54 ms/step?
#
# The budget says the predictor is what stands in the way, and it is arithmetic
# on measured parts, not a forecast:
#
#     resident-only floor, nothing moves, nothing predicted   53.48
#     + adaptive cache alone (AI-C, measured)                 54.31   (+0.83)
#     + predictor over all 75 layers (arm A implies)          ~56.1   (+1.81)
#     + gather at slots=1 + reuse                             ~56.5   (+0.4..1.3)
#
# So everything after the cache has to fit in ~0.6 ms, and the predictor alone
# spends three times that. It is not doing three times that much WORK -- 1.81 ms
# over 75 layers is ~24 us per layer for one selection kernel on four tokens --
# but a captured graph replays every node whether or not the layer needed
# anything, and more than half of layer-calls need nothing at all.
#
# KT_PRED_LAYER_STRIDE drops the predictor onto every Nth layer, decided once
# before capture so the skipped layers have no nodes at all. Both halves of the
# cost go together (the widened gate GEMM is gated by the same flag). A skipped
# layer's misses fall back to substitution from the resident pool: a quality
# cost, not a correctness one -- which is why every row here is scored for
# coherence and not just timed.
#
# The trade is explicit: stride N costs ~1/N of the predictor and buys ~1/N of
# the fetch coverage. Whether the cache has already made that coverage
# redundant is exactly the open question.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

MODEFILE="$SP/kt_topk_mode.pred_stride"
echo safe8 > "$MODEFILE"
OUT=bench/profile_out/pred_stride.json
mkdir -p bench/profile_out
WARM_TOKENS="${WARM_TOKENS:-1400}"

boot () {
  local log="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $log: $* ==="
  env "$@" \
    KT_GPU_ONLY=1 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
    KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
    KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 KT_PREFETCH_SLOTS=1 \
    KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
    KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_MAX_SWAP=8 KT_ADAPTIVE_PERIOD=32 \
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
  grep -a "kt-pred-fused] stacked" "$SP/$log.log" | tail -1
  return 0
}

run_arm () {   # run_arm <log> <label>
  echo "--- warm-up $WARM_TOKENS tokens (let the cache converge) ---"
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 \
    --tokens "$WARM_TOKENS" --label "warm-$2" --out "$SP/warm_$2.json" 2>&1 \
    | grep -a "ms/step" || true
  local t
  for t in 8 2; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "$2-top$t" --out "$OUT" 2>&1 | grep -a "ms/step" || true
  done
  echo "--- counters ($2) ---"
  grep -a "kt-prefetch] step" "$SP/$1.log" | tail -2
  grep -ac "kt-adaptive] layer=" "$SP/$1.log" 2>/dev/null || true
}

for S in 2 3; do
  if boot "PSTR_$S" KT_PRED_LAYER_STRIDE=$S; then
    run_arm "PSTR_$S" "stride$S"
    # Coherence only on the arm that beats the target; a fast degenerate row is
    # not a result. top0 is the calibration case and must collapse.
    TOKENS=1200 bash bench/coherence_run.sh "stride$S" 8 2 0
  fi
done

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/pstr_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== pred_stride done ==="
