#!/usr/bin/env bash
# WIDE died of CUDA OOM during weight loading -- 28 MiB short, with 16 MiB free.
#
# That is TODO item 7's VRAM cliff, now bounded: at MEM_FRACTION=0.94 a 4-slot
# landing buffer (4 x 9.56 = 38 MiB/card) boots and a 16-slot one (153 MiB) does
# not. The extra 115 MiB is the whole difference.
#
# Dropping to 0.92 buys ~2 GB, which is ample. But MEM_FRACTION is exactly the
# variable that Stage H6 caught moving a prefetch row by 3.2 ms, so I will NOT
# compare a 0.92 prefetch row against the 0.94 baseline. Both rows are re-run at
# 0.92 -- the no-prefetch ladder and the widened prefetch -- so the comparison
# lives inside one setting.
#
# Reference, measured at 0.94 (valid among themselves, not against these):
#   OFF100-top8 150.13   OFF100-top2 66.56   OFF100-top0 53.06
#
# Prediction unchanged: at tier 8, where the CPU pool is 97 ms, WIDE should beat
# OFF by more than the predictor's ~5.5 ms. Widening P from 2 to 8 also moves
# ~4x the bytes, so the alternative outcome is that the link saturates and
# head-of-line blocking shows up as a LOSS rather than a smaller win. Those two
# failures look different and the counters distinguish them.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT=bench/profile_out/latency_split.json

while pgrep -f "bash bench/tier_pool.sh" >/dev/null 2>&1; do sleep 20; done
sleep 20

run_tiers () {
  local prefix="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $prefix: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.92 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/tp_$prefix.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/tp_$prefix.log" && { echo "BOOTFAIL $prefix"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $prefix"; return 1; }
  for T in 8 2 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier "$T" --runs 3 \
      --tokens 200 --label "$prefix-top$T" --out "$OUT" 2>&1 | tail -2
    echo -n "  counters after top$T: "
    grep -a "kt-prefetch\] step" "$SP/tp_$prefix.log" | tail -1 || echo none
  done
}

run_tiers OFF92 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0

run_tiers WIDE92 KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 KT_PRED_POINT=post \
  KT_PRED_P=8 KT_PREFETCH_SLOTS=16 KT_PREFETCH_BLOCKS=8 \
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/tpr_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== tier_pool_retry done ==="
