#!/usr/bin/env bash
# One boot, far enough to dump a resident expert and a gathered one, then stop.
# The dump happens inside pf_build_table during weight load, so no request is
# ever served and the server is killed as soon as the layer-40 dump lands.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DUMP=bench/profile_out/pf_dump
rm -rf "$DUMP"; mkdir -p "$DUMP"
bash bench/_kill_servers.sh >/dev/null

KT_PF_DUMP="$DUMP" KT_PREFETCH_VERIFY_LAYERS=3,40 \
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
  KT_PRED_FUSED=1 KT_PRED_POINT=pre \
  GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
  RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
  KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/pfdump.log" 2>&1 &

for i in $(seq 1 200); do
  ls "$DUMP"/L40_tp1.pt >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/pfdump.log" && { echo BOOTFAIL; break; }
  sleep 5
done
echo "=== dump lines ==="
grep -a "kt-prefetch\]\[dump\]" "$SP/pfdump.log" || echo "NO DUMP LINES"
bash bench/_kill_servers.sh >/dev/null
ls -la "$DUMP" 2>/dev/null
echo "=== comparing against the checkpoint ==="
.venv/bin/python bench/gather_vs_checkpoint.py --dump "$DUMP" 2>&1 | tail -40
echo "=== pf_dump done ==="
