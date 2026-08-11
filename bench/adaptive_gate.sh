#!/usr/bin/env bash
# TARGET: 54 ms/step with the prefetch AND the cache both live.
#
# Where the 56.34 ms gate spends its time over the 53.4 ms resident-only floor
# (H22, measured):
#
#     floor (GPU-only, nothing moves)      53.35
#     + n+1 fused predictor                +1.81
#     + gather at slots=1 + reuse (f=0.37) +1.29
#     = 56.45, and the gate read 56.34
#
# H22 killed every way of making the MOVEMENT cheaper: the join costs ~0.4 ms
# (no barrier to hide), the tax is per-byte SM contention with the cutlass
# GEMMs and it is SUPERLINEAR, the grid is already at its optimum, and both the
# DMA and stream-priority routes are dead on inspection. So the only remaining
# lever is to MOVE LESS: make the experts the predictor keeps asking for
# RESIDENT, because a resident expert costs 0 ms forever and a landing slot
# costs milliseconds every time it is filled.
#
# That is what the adaptive cache does, and this is the arm that has never been
# run together with the prefetch. Prediction: f falls as residency converges on
# the live routing, and the gather term falls with it. It cannot touch the
# predictor's 1.81 ms, so on this arithmetic alone the arm lands ~55.2 and the
# predictor becomes the binding term -- which is a decomposition, not a
# forecast, and the run is what settles it.
#
# TWO ARMS, ONE SCRIPT, IDENTICAL WARM-UP, because the standing rule is that a
# baseline agreeing across boots licenses nothing about the rows measured
# beside it (H6). The 56.34 from two days ago is NOT the control; arm A is.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

MODEFILE="$SP/kt_topk_mode.adaptive_gate"
echo safe8 > "$MODEFILE"
OUT=bench/profile_out/adaptive_gate.json
mkdir -p bench/profile_out

# Long enough for the cache to converge before anything is timed. The tick
# fires every KT_ADAPTIVE_PERIOD steps and moves KT_ADAPTIVE_LAYERS_PER_TICK
# layers, so covering all 75 MoE layers once takes 75/2*32 = 1200 steps.
WARM_TOKENS="${WARM_TOKENS:-1400}"

boot () {   # boot <logname> <extra env assignments...>
  local log="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $log: $* ==="
  env "$@" \
    KT_GPU_ONLY=1 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
    KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
    KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 KT_PREFETCH_SLOTS=1 \
    KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
    GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_STORE_SHM=1 MTP=1 \
    MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/$log.log" 2>&1 &
  local i
  for i in $(seq 1 400); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/$log.log" && { echo "BOOTFAIL $log"; tail -30 "$SP/$log.log"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $log"; tail -25 "$SP/$log.log"; return 1; }
  return 0
}

warm () {   # identical traffic in both arms; only the adaptive arm reacts to it
  echo "--- warm-up: $WARM_TOKENS tokens ---"
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 \
    --tokens "$WARM_TOKENS" --label "warm" --out "$SP/warm_$1.json" 2>&1 \
    | grep -a "ms/step" || true
}

measure () {  # measure <prefix>
  local p="$1" t
  for t in 8 4 2 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "$p-top$t" --out "$OUT" 2>&1 | grep -a "ms/step" || true
  done
  # f, straight from the server's own counters. A prefetch arm whose fetch rate
  # is not reported is not measured -- an arm that silently fetches nothing
  # looks exactly like a cheap one.
  echo "--- fetch counters ($p) ---"
  grep -a "kt-prefetch" "$SP/$p.log" | tail -3
  echo "--- adaptive ticks ($p) ---"
  grep -ac "kt-adaptive" "$SP/$p.log" 2>/dev/null || true
  grep -a "kt-adaptive] layer=" "$SP/$p.log" 2>/dev/null | tail -3
}

# ---- arm A: the control. Prefetch on, cache OFF. -------------------------
# ARM=B skips it to re-run only the cache arm. The two arms are separate boots
# whatever this is set to -- KT_ADAPTIVE_DECODE is read at import and decides
# how the kt store is staged, so there is no per-request A/B for it. This is
# the one comparison on this box that cannot be made within a single run.
#
# /dev/shm SIZING. The cache forces ALL 256 experts per layer into the store
# (an evicted expert must still have weights somewhere to be promoted back)
# where the plain run stages only the ~156 non-resident ones. At 9.56 MiB per
# expert per card that is ~372 GiB, and a default 315 GiB /dev/shm dies with
# SIGBUS at ~layer 66 of 78 -- writing past a tmpfs size limit is a bus error,
# not an allocation failure, so it surfaces as a crash with no OOM message.
#   sudo mount -o remount,size=450G /dev/shm
if [ "${ARM:-AB}" != "B" ] && boot AG_A KT_ADAPTIVE_DECODE=0; then
  warm A
  measure AG_A
fi

# ---- arm B: prefetch on, cache ON. ---------------------------------------
# MAX_SWAP is held under KT_TIER_INCREMENTAL_MAX (16) so every tick takes the
# stable-slot path. Above it the swap declines to the full restage, which
# renumbers every slot and re-copies ~100 experts -- 144 ms on the forward
# thread, ~4.5 ms/step amortised at this period, i.e. larger than the entire
# budget this run is trying to protect.
if boot AG_B KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_MAX_SWAP=8 \
             KT_ADAPTIVE_PERIOD="${KT_ADAPTIVE_PERIOD:-32}" \
             KT_ADAPTIVE_LAYERS_PER_TICK="${KT_ADAPTIVE_LAYERS_PER_TICK:-2}" \
             KT_TIER_COUNT_MODE="${KT_TIER_COUNT_MODE:-top8}"; then
  warm B
  measure AG_B
  echo "--- coherence, 1200 tokens; top0 is the calibration case ---"
  TOKENS=1200 bash bench/coherence_run.sh agB 8 2 0
fi

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ag_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== adaptive_gate done ==="
