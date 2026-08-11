#!/usr/bin/env bash
# WHAT DOES THE SPEED COST IN ANSWERS? -- LiveBench reasoning across the ladder.
#
# Everything measured on this branch so far is a SPEED number gated by a
# coherence detector, and that detector is a FLOOR test: it catches collapse
# (`</think>` x400), not the loss of a puzzle. So "clean at top8" has been
# standing in for "as accurate as the real model", which it does not mean.
# This run puts a graded benchmark under each configuration instead.
#
# THE ANCHOR is the hybrid production server at top8: every routed expert is
# genuine, the non-resident ones taken by the CPU path. That is the accuracy
# this model can produce at this quantisation, and every GPU-only row below is
# a SUBSTITUTION of it -- a routed expert that is not resident and not fetched
# is replaced by one that is. The question is how much of the answer that costs.
#
#   REF    hybrid, top8            ~16.8 tok/s   all 8 genuine        anchor
#   FLOOR  GPU-only, nothing moves  53.48 ms     ~3.2/8 genuine, static
#   CACHE  + adaptive cache         54.31 ms     residency follows the routing
#   S4     + prefetch, stride 4     54.51 ms     1 layer in 4 gets a fetch
#   PF1    + prefetch, stride 1     56.63 ms     every layer gets a fetch
#
# S4 is the candidate and it is the one with a QUALITY RISK that has never been
# tested: three layers in four lose their fetch, so the stride is not a free
# 2.12 ms if those layers were the ones that needed it.
#
# METHOD -- the design choices that make the deltas mean something:
#
#   * STRATIFIED, not the first 60. The dataset is ordered by task, so --limit
#     60 would be 60 zebra puzzles. --per-task-limit 20 gives 20 of each.
#   * SAME QUESTIONS, SAME ORDER, temperature 0 everywhere. The comparison is
#     PAIRED: with n=60 the standard error on an absolute score is ~6 points,
#     which would hide anything smaller than a catastrophe, but per-question
#     agreement with the anchor is far sharper -- it only counts the questions
#     where the configurations actually disagree.
#   * WARM-UP BEFORE SCORING on every cache row. The cache needs ~1200 decode
#     steps to converge; grading it before that measures the transient.
#   * Each row is its own boot (every knob here is boot-time, fixed at CUDA
#     graph capture), and the anchor is a boot too -- not the numbers sitting
#     in a memory file from another day.
#
# WHAT THIS RUN CANNOT SETTLE, stated before it starts: n=60 with three tasks
# is a screening instrument. It will separate "answers like the anchor" from
# "visibly worse"; it will NOT resolve a 2-3 point difference between two
# GPU-only rows, and I should not claim one from it.
#
# STATE 2026-08-10: REF is BANKED -- 73.3% (zebra 100 / web_of_lies 70 /
# spatial 50), n=60, 203.6 min, kept at bench/livebench/results/ as well as in
# $SP/acc_ladder/REF.json. Resume the other four rows with
#   ROWS="FLOOR CACHE S4 PF1" bash bench/accuracy_ladder.sh      # ~5.7 h
# 12 of REF's 60 questions scored 0 by TRUNCATION at the 8000-token cap, so read
# the report's trunc column before attributing any delta to substitution.
#
#   bash bench/accuracy_ladder.sh              # all five rows, ~3 h
#   ROWS="S4 FLOOR" bash bench/accuracy_ladder.sh
#   PER_TASK=8 bash bench/accuracy_ladder.sh   # ~1 h smoke
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

PER_TASK="${PER_TASK:-20}"
MAXTOK="${MAXTOK:-8000}"
ROWS="${ROWS:-REF FLOOR CACHE S4 PF1}"
RESDIR="$SP/acc_ladder"
mkdir -p "$RESDIR"
MODEFILE="$SP/kt_topk_mode.acc"
echo safe8 > "$MODEFILE"

wait_up () {   # wait_up <logfile> ; fails loudly rather than grading a dead server
  local log="$1" i
  for i in $(seq 1 400); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed|Bus error" "$log" && {
      echo "BOOTFAIL $log"; grep -aE "CUDA out of memory|Bus error|Error" "$log" | tail -3; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $log"; return 1; }
  return 0
}

boot_gpu_only () {   # boot_gpu_only <logname> <extra env...>
  local log="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $log: $* ==="
  env "$@" \
    KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 \
    GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_STORE_SHM=1 MTP=1 \
    MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/$log.log" 2>&1 &
  wait_up "$SP/$log.log" || return 1
  # Proof the flags took. An inert flag has faked a result on this branch twice.
  grep -a "kt-pred-fused] stacked" "$SP/$log.log" | tail -1 || echo "  (no gate fusion)"
  return 0
}

score () {   # score <row> <model> [warm]
  local row="$1" model="$2" warm="${3:-}"
  if [ -n "$warm" ]; then
    echo "--- $row: warm-up so the cache is converged before anything is graded ---"
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 \
      --tokens 1400 --label "warm-$row" --out "$SP/warm_acc_$row.json" 2>&1 \
      | grep -a "ms/step" || true
  fi
  echo "--- $row: LiveBench reasoning, $PER_TASK per task, temperature 0 ---"
  .venv/bin/python bench/livebench/run_livebench.py \
    --model "$model" --per-task-limit "$PER_TASK" --max-tokens "$MAXTOK" \
    --out "$RESDIR/$row.json" 2>&1 | tail -12
}

for row in $ROWS; do
  case "$row" in
    REF)
      # The anchor: hybrid CPU+GPU, every genuine expert computed. Booted fresh
      # rather than reusing whatever is listening, so it is a row like the others.
      bash bench/_kill_servers.sh >/dev/null
      echo "=== booting REF: hybrid production, all experts genuine at top8 ==="
      KT_TOPK_MODE_FILE="$MODEFILE" KT_GPU_PREFILL_THRESHOLD=0 MTP=1 \
      TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
        nohup ./run_fast.sh > "$SP/acc_ref.log" 2>&1 &
      wait_up "$SP/acc_ref.log" && score REF GLM5.2-top8
      ;;
    FLOOR)
      boot_gpu_only acc_floor \
        KT_PRED_FUSED=0 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 \
        KT_PREFETCH_CPUSKIP=0 KT_PREFETCH_SLOTS=1 KT_ADAPTIVE_DECODE=0 \
        && score FLOOR GLM5.2-top8
      ;;
    CACHE)
      boot_gpu_only acc_cache \
        KT_PRED_FUSED=0 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 \
        KT_PREFETCH_CPUSKIP=0 KT_PREFETCH_SLOTS=1 \
        KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=128 \
        KT_ADAPTIVE_LAYERS_PER_TICK=2 KT_ADAPTIVE_MAX_SWAP=8 KT_TIER_COUNT_MODE=top8 \
        && score CACHE GLM5.2-top8 warm
      ;;
    S4|PF1)
      stride=4; [ "$row" = PF1 ] && stride=1
      boot_gpu_only "acc_$row" \
        KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
        KT_PRED_FUSED=1 KT_PRED_FUSED_STACK=1 KT_PRED_POINT=pre \
        KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 KT_PRED_LAYER_STRIDE=$stride \
        KT_PREFETCH_REUSE=2 KT_PREFETCH_SLOTS=1 KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
        KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=128 \
        KT_ADAPTIVE_LAYERS_PER_TICK=2 KT_ADAPTIVE_MAX_SWAP=8 KT_TIER_COUNT_MODE=top8 \
        && score "$row" GLM5.2-top8 warm
      ;;
  esac
  echo "=== row $row done ($(date +%H:%M)) ==="
done

echo "=== paired report ==="
.venv/bin/python bench/accuracy_report.py "$RESDIR" 2>&1 | tail -60

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/acc_prod.log" 2>&1 &
wait_up "$SP/acc_prod.log" && echo "PROD UP" || echo "PROD FAILED"
echo "=== accuracy_ladder done ==="
