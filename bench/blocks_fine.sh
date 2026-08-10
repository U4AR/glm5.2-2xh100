#!/usr/bin/env bash
# Where exactly is the bottom of the U?
#
# Stage H21 found the gather cost is U-shaped in grid size -- 28.73 ms at 1 block,
# 6.91 at 8, 15.11 at 32 -- because the gather kernel and the cutlass MoE GEMMs
# compete for SMs. Too few blocks and the transfer is slow enough to be exposed;
# too many and it starves the compute it is meant to hide behind.
#
# But the sweep was 1/8/32, and **8 was tuned for the OLD byte-granular gather**,
# which reached only 30 GB/s and needed to flood the machine with stalled warps to
# move anything at all. The current uint4 kernel holds 46 GB/s -- link speed --
# "from 8 blocks upward", which says 8 is the FLOOR of the useful range, not the
# optimum of it. The minimum could sit anywhere from 2 to 16.
#
# This is the cheapest possible win: one env var, no code, and the curve is steep
# on both sides of 8 (a factor of 4 at 1, a factor of 2 at 32). If the true bottom
# is at 4 or 6, it is worth ~1-2 ms/step for nothing.
#
# All at TIER 0 with identical bytes (f = 1.91 in every H21 row regardless of
# grid), so any difference is grid alone.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
RATE=bench/profile_out/blocks_fine.json
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

wait_for_marker "=== exposure_slope done ===" slope.out

boot () {
  local b="$1"
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting B$b: KT_PREFETCH_BLOCKS=$b ==="
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
  KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
  KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
  GPU_EXPERTS=100 KT_PREFETCH_SLOTS=4 KT_PRED_P=8 KT_PREFETCH_BLOCKS="$b" \
  KT_STORE_SHM=1 MTP=1 MEM_FRACTION=0.94 KT_TOPK_MODE_FILE="$MODEFILE" \
  KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/b_$b.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/b_$b.log" && { echo "BOOTFAIL B$b"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP B$b"; return 1; }
  for t in 0 8; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "B$b-top$t" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  done
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/b_$b.log" | tail -1 || echo none
}

for b in 2 4 6 12 16; do boot "$b"; done

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/bfine_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== blocks_fine done ==="
