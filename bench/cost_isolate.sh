#!/usr/bin/env bash
# Separate what the prefetch COSTS from what it BUYS.
#
# The depth/shadow ladder ended in a tie that cannot be interpreted:
#
#   S-base  (off)              65.77 ms/step
#   S-d1    (depth 1, P=2)     64.29     1.59 fetched/call
#   S-p1    (depth 1, P=1)     64.50     0.84 fetched/call
#   S-d2    (depth 2, P=2)     65.82     1.75 fetched/call
#
# S-p1 moves 47% fewer bytes than S-d1 and lands 0.21 ms behind it. Either the
# bytes were never the cost, or they were and the coverage lost to P=1 gave the
# saving straight back. Those two readings imply opposite next moves, and the
# ladder above cannot tell them apart, because each row mixes a cost that falls
# with bytes and a benefit that falls with coverage.
#
# THE SPLIT. The expert tier is per-request (model name `-topN`), so one boot
# can be asked for tier 0 and tier 2. At tier 0 every expert is substituted with
# a GPU-resident one: there is no CPU expert work left to avoid, so the prefetch
# BUYS EXACTLY NOTHING and whatever it adds over the prefetch-off floor is pure
# cost. This is the same instrument that priced d1's overhead at +11.00 ms in
# Stage H12; here it is swept across configurations.
#
#   cost(cfg)    = tier0(cfg) - tier0(off)
#   benefit(cfg) = cost(cfg) - (tier2(off) - tier2(cfg))
#
# Both come out of ONE boot per configuration, so no subtraction crosses a boot
# -- the rule this project keeps relearning.
#
# Read it as: if cost is flat across P=1/P=2, the bytes are not what the prefetch
# is paying for and fetching fewer of them is a dead lever; the fixed per-layer
# machinery (stream wait, event record, predictor) is. If cost tracks bytes,
# then P=1 lost on coverage and the move is to cut bytes WITHOUT losing coverage
# -- reuse across steps, which is free coverage by construction.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/cost_isolate_rate.json

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/c_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/c_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  # tier 0 = pure cost, tier 2 = what ships. Same boot, so the subtraction is legal.
  for t in 0 2; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "$label-top$t" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  done
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/c_$label.log" | tail -1 || echo none
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"
OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"

boot C-off   $OFF
boot C-d1P2  $ON $FUSED KT_PRED_FUSED_DEPTH=1
boot C-d1P1  $ON $FUSED KT_PRED_FUSED_DEPTH=1 KT_PRED_P=1
boot C-d2P2  $ON $FUSED KT_PRED_FUSED_DEPTH=2
# Predictor on, gather off: the fixed machinery with ZERO bytes on the wire.
# If cost is flat across P, this row is where the flat part lives.
boot C-nogather KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=0 \
                $FUSED KT_PRED_FUSED_DEPTH=1

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/c_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== cost_isolate done ==="
