#!/usr/bin/env bash
# Split the 22.46 ms into BYTES vs WAITING. One boot each, plus a prod restore.
#
# The claim under test is my own: that the gather costs 22 ms because it
# contends with the CPU experts for host DRAM. That story does not survive its
# own evidence -- the occupancy profile has DRAM at 38% of 379 GB/s, and the
# gather adds only ~18.7 GB/s. 5% more traffic cannot cost 34% more step time.
# So stop reasoning and separate the two possibilities:
#
#   W1  GATHER=1 WAIT=0   bytes move, SMs are taken, but the compute stream
#                         never waits on pf_done. Cost here = what the transfer
#                         COSTS by existing (bandwidth, SMs, memory system).
#   W2  GATHER=1 WAIT=1   adds the barrier back. Cost here = the same, plus
#                         however long the compute stream sits blocked because
#                         the gather has not finished.
#
# W1 ~= 0  and W2 ~= 22  -> the bytes are free; the 22 ms is pure WAITING, i.e.
#                          the gather is not finishing inside a whole layer of
#                          slack. A latency/issue problem, not a bandwidth one.
# W1 ~= 22 and W2 ~= 22  -> the transfer genuinely consumes a shared resource.
#                          Then find which: SMs (cutlass slows) or host memory.
#
# WAIT=0 forces ROUTE=0 in kt_ep_wrapper (reading an unsynchronised slot is a
# data race that shows up as garbage text, not an error). So both rows here are
# timing probes; neither is a correctness run. Reference points from the valid
# take-3 sweep, same box state:
#     P0 (predictor, no gather) 71.67 ms      P2 (gather + barrier) 94.13 ms
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/prefetch_rate.json

boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" \
    KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 \
    GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/why_$label.log" 2>&1 &
  for i in $(seq 1 240); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/why_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  # Without this counter the row is void -- the whole take-1/take-2 sweep died
  # here, silently, because pf_issue was never reachable.
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/why_$label.log" | tail -1 || echo "NONE -- ROW IS VOID"
}

boot WHY-W1-nowait KT_PREFETCH_GATHER=1 KT_PREFETCH_WAIT=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot WHY-W2-wait   KT_PREFETCH_GATHER=1 KT_PREFETCH_WAIT=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/why_prod.log" 2>&1 &
for i in $(seq 1 240); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
.venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 2 --tokens 200 \
  --label WHY-prod-restore --out "$OUT" 2>&1 | tail -3
echo "=== whyserial done ==="
