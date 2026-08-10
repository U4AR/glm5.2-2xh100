#!/usr/bin/env bash
# Second half of the decomposition. Part 1 measured the bytes and found them
# free: gather-on/wait-off/route-off and gather-on/wait-on/route-off both land
# within noise of the no-gather control (66.64 / 66.74 / 66.28 ms). So neither
# interference nor stall explains the loss. That leaves the routing flip, which
# does two separable things:
#
#   T3   route ON,  cpuskip ON   the shipped behaviour, numerically exact
#   T3a  route ON,  cpuskip OFF  GPU also computes the landed expert, CPU still
#                                computes it too -> double-counted, WRONG TEXT.
#                                Prices the GPU side of the flip alone.
#   T3b  route OFF, cpuskip ON   CPU drops the expert, GPU never picks it up ->
#                                contribution missing, WRONG TEXT. Prices what
#                                the CPU actually saves by not computing it.
#
# T3a and T3b are timing probes; their output is expected to be incoherent and
# is not a regression. T3 is the only row whose text should be read.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT=bench/profile_out/prefetch_rate.json

boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" \
    GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/dec_$label.log" 2>&1 &
  for i in $(seq 1 240); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/dec_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  echo "=== measuring $label ==="
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -4
  grep -a "kt-prefetch\] step" "$SP/dec_$label.log" | tail -1
}

boot DEC-T3-full            KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1
boot DEC-T3a-gpuonly        KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=0
boot DEC-T3b-cpuskiponly    KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=1
echo "=== routing-flip decomposition done ==="
