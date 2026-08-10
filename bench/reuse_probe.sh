#!/usr/bin/env bash
# How many of the experts we fetch are ALREADY sitting in a landing slot?
#
# Stage H15 left exactly one lever. Benefit is capped at 13.31 ms (the whole CPU
# expert path) and is already 95.6% collected, so coverage is worth nothing more;
# cost is 1.64 ms + ~5.1 ms per expert fetched per layer-call. The only way left
# to gain is to move fewer bytes AT THE SAME COVERAGE -- and re-fetching an
# expert whose bytes are already in the slot is pure waste.
#
# This boot does NOT change behaviour. KT_PREFETCH_REUSE=1 is measure-only: the
# kernel probes, per call, how many landing slots hold an expert this call wants
# again, adds it to stats[4], and picks slots exactly as before. Two reasons for
# measuring before building:
#
#   - the hit rate has never been observed, and the closest proxy (`persist` in
#     the chain sweep, ~26%) is an UPPER bound -- slots hold ~1.8 fetched
#     experts, a subset of everything routed last step, so the real figure must
#     be lower;
#   - at ~5.1 ms per expert/call the answer converts straight into milliseconds,
#     so the build/don't-build line is a number, not a judgement.
#
# The step rate here must also reproduce ~64.3 ms/step. Mode 1 adds one shared
# memory probe per slot per call and must be free; if this row is slower than
# C-d1P2 was, the probe itself is not free and the reuse path starts in debt.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
RATE=bench/profile_out/reuse_probe_rate.json

wait_for_marker "=== chain-predict done ===" chain_rerun.out

bash bench/_kill_servers.sh >/dev/null
echo "=== booting R-measure: KT_PREFETCH_REUSE=1 (measure only) ==="
KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
KT_PREFETCH_REUSE=1 \
GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/r_measure.log" 2>&1 &

for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/r_measure.log" && { echo "BOOTFAIL"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; tail -30 "$SP/r_measure.log"; exit 1; }

.venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
  --tokens 200 --label R-measure --out "$RATE" 2>&1 | grep -a "ms/step" || true
echo "--- counters ---"
grep -a "kt-prefetch\]" "$SP/r_measure.log" | tail -4

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/r_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== reuse_probe done ==="
