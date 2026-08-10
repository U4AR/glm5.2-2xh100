#!/usr/bin/env bash
# The prefetch has only ever been benchmarked where it has the LEAST to gain.
#
# Measured on the live server just now, ms/step (the forward shape is identical
# across tiers -- same 4-token verify batch -- so ms/step is comparable):
#
#   top8 (all 8 routed experts honoured)   147.83     CPU expert path = 94.39 ms
#   top2 (the shipped default)              65.80     CPU expert path = 12.36 ms
#   top0 (zero CPU experts)                 53.44     the floor
#
# Every prefetch row this project has ever produced was measured at top2, where
# the total pool the prefetch can win is 12.36 ms -- and the predictor costs
# 5.47 ms before a single byte moves. It recovers 2.1-2.5 ms of that pool and
# nets +3 ms. That is not a broken prefetch; it is a prefetch competing for
# scraps left by a lever that already fired.
#
# At top8 the pool is 94.39 ms. The predictor's cost does not change. If the
# fetch removes even the same FRACTION of the pool it does at top2 (~18%), that
# is ~17 ms against 5.5 ms of cost -- a large win instead of a small loss.
#
# One boot, prefetch on, then all three tiers measured inside it (the tier is a
# per-request field, so no reboot between them). Compare against the prod rows
# above, which were taken minutes earlier on the same box.
#
# Prediction, committed before the run: PF-top8 lands BELOW 147.83 by more than
# the 5.5 ms the predictor costs. If it does not, the fetch effect does not
# scale with the size of the CPU pool and something else caps it -- coverage
# (74-79% of layer-calls fully covered) or the 4-slot skip rule (18.6% of
# layer-calls want more than 4 distinct experts and are skipped outright).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT=bench/profile_out/latency_split.json

bash bench/_kill_servers.sh
echo "=== booting prefetch-on server (post-state predictor) ==="
KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 KT_PRED_POINT=post \
KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/pf8.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/pf8.log" && { echo "BOOTFAIL"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; exit 1; }

for T in 8 2 0; do
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier "$T" --runs 3 \
    --tokens 200 --label "PF-top$T" --out "$OUT" 2>&1 | tail -2
done
echo -n "counters: "; grep -a "kt-prefetch\] step" "$SP/pf8.log" | tail -1

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/pf8_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== prefetch_at_tier8 done ==="
