#!/usr/bin/env bash
# Does gating mask consumption on `pf_issued` restore both reproducibility and
# accept length?
#
# Two questions per row, one boot, because they are the same question asked at
# different resolutions:
#   determinism   -- 5 greedy repeats of one prompt, distinct completion hashes
#   prefetch_rate -- ms/step and accept length, the numbers that decide shipping
#
# Rows:
#   F-base   prefetch off              the floor, in this config
#   F-nofix  prefetch on, fix OFF      the bug, reproduced deliberately in the
#                                      same boot config, so the comparisons below
#                                      are within-config rather than against a
#                                      remembered number from another day
#   F-tmax   prefetch on, fix OFF,     the mechanism test that does NOT depend on
#            KT_PRED_TMAX=4096         my patch being correct. Raising the token
#                                      gate makes the predictor fire on the
#                                      PREFILL batch too, so the masks are fresh
#                                      when prefill reads them -- by a completely
#                                      different route than the fix. If this goes
#                                      deterministic, stale masks are the cause
#                                      and the remaining suspect (a read-before-
#                                      write race on the landing slots) is dead,
#                                      because that race would survive here
#   F-full   prefetch on, fix ON       must be 1 distinct hash. If accept comes
#                                      back to ~2.857 the regression WAS the
#                                      stale mask and the -3.3% step time is a
#                                      real win
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DET=bench/profile_out/determinism.json
RATE=bench/profile_out/determ_fix_rate.json

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/f_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/f_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 5 --tokens 200 --tier 2 \
    --label "$label" --out "$DET" 2>&1 | tail -12
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$RATE" 2>&1 | tail -2
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/f_$label.log" | tail -1 || echo none
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"

boot F-base  KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot F-nofix $ON $FUSED KT_PREFETCH_FRESH=0
boot F-tmax  $ON $FUSED KT_PREFETCH_FRESH=0 KT_PRED_TMAX=4096
boot F-full  $ON $FUSED KT_PREFETCH_FRESH=1

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/f_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== determ_fix done ==="
