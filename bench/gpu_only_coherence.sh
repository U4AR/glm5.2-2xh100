#!/usr/bin/env bash
# Does GPU-ONLY mode stay coherent? Long samples, with the detector calibrated.
#
# The slot ladder priced GPU-only mode at ~61.6 ms/step for FULL routing against
# 146.11 with the CPU. That number is worthless until the output is checked,
# because the mode's whole trade is "substitute whatever the link could not
# carry" -- and substitution damage is invisible at 200 tokens. Every 200-token
# sample this box has produced scored clean, including tiers known to degrade.
#
# So: 1200 tokens, and the detector must PROVE it can fail on this boot before
# any row counts. `coherence_run.sh` samples top0 (substitute everything, known
# degenerate) and top8 (full routing) alongside the tiers under test; if top0
# does not score degenerate, the run is declared uncalibrated and nothing in it
# is a pass. Same discipline as the depth-0 control that validated the predictor.
#
# Booted at the G-s4 configuration -- the fastest row, 61.58 ms/step at top8, and
# therefore the one with the MOST substitution and the most to prove.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

wait_for_marker "=== gpu_only done ===" gpu_only.out

bash bench/_kill_servers.sh >/dev/null
echo "=== booting GPU-only (G-s4 config) for the coherence pass ==="
KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 \
GPU_EXPERTS=100 KT_PREFETCH_SLOTS=4 KT_PRED_P=8 \
KT_STORE_SHM=1 KT_PREFETCH_BLOCKS=8 MTP=1 MEM_FRACTION=0.94 \
KT_TOPK_MODE_FILE="$MODEFILE" \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/coh_boot.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/coh_boot.log" && { echo BOOTFAIL; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; tail -25 "$SP/coh_boot.log"; exit 1; }

TOKENS=1200 bash bench/coherence_run.sh gpuonly 8 4 2 0

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/coh_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== gpu_only_coherence done ==="
