#!/usr/bin/env bash
# Re-run of the one row that decides the question. W1 died of a host-RAM OOM at
# layer 68 on the first attempt -- not a code failure of the WAIT=0 path, it
# never reached a forward. The page cache was thrashing because a storage
# benchmark had just evicted the weights; it is warm again now.
#
# W1: gather runs, every byte moves, every SM it wants is taken -- but the
# compute stream does NOT wait on pf_done. So this isolates what the transfer
# costs by EXISTING, with the blocking removed.
#
# Reference points, same box state:
#   P0 predictor only, no gather   71.67 ms
#   W2 gather + barrier            94.74 ms   (reproduced take-3's 94.13)
#
#   W1 ~= 72  -> bytes are free; all 23 ms is the compute stream BLOCKED,
#                i.e. the gather is not finishing inside a full layer of slack.
#   W1 ~= 94  -> the gather consumes a shared resource whether or not anyone
#                waits for it. Next question is which: SMs or host memory.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT=bench/profile_out/prefetch_rate.json

pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*

echo "=== booting WHY-W1-nowait (retry) ==="
env KT_PREFETCH_GATHER=1 KT_PREFETCH_WAIT=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0 \
  KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 \
  GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
  RUNGLM_TOPK_MODE=safe2 MTP=1 \
  KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/why_W1_retry.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/why_W1_retry.log" && { echo "BOOTFAIL W1 retry"; break; }
  sleep 10
done

if curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "=== measuring WHY-W1-nowait ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label WHY-W1-nowait-retry --out "$OUT" 2>&1 | tail -4
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/why_W1_retry.log" | tail -1 || echo "NONE -- ROW IS VOID"
else
  echo "W1 RETRY DID NOT COME UP"; tail -5 "$SP/why_W1_retry.log"
fi

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/why_prod2.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 2 --tokens 200 \
  --label WHY-prod-restore2 --out "$OUT" 2>&1 | tail -3
echo "=== w1 retry done ==="
