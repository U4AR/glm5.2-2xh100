#!/usr/bin/env bash
# Is the prefetch a win against what PRODUCTION actually runs?
#
# Every prefetch number so far compares GPU_EXPERTS=100 + 4 landing slots against
# GPU_EXPERTS=100 with those 4 slots ALLOCATED BUT IDLE. That keeps VRAM equal
# between the two rows, which is the right control for "does the machinery pay
# for itself" -- but it is NOT the shipping question, because production holds
# 104 resident experts and no slots.
#
# A landing slot is a real expert-sized allocation (~9.73 MiB per card per layer,
# ~730 MiB per slot across 75 MoE layers), so the prefetch buys its staging area
# by giving up four residents. The fair question is whether four experts' worth of
# prefetching beats four experts' worth of residency.
#
#   F104-plain   GPU_EXPERTS=104, SLOTS=0   what production runs today
#   F100-idle    GPU_EXPERTS=100, SLOTS=4   the handicapped baseline used all day
#   F100-fetch   GPU_EXPERTS=100, SLOTS=4   prefetch on, index fixed
#
# F100-fetch vs F104-plain is the number that decides whether this ships.
# F100-idle is kept so the earlier +2.18 ms can be reconciled rather than
# quietly replaced.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/fair_baseline_rate.json
DET=bench/profile_out/fair_baseline_det.json

boot () {
  local label="$1"; local experts="$2"; local slots="$3"; shift 3
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: experts=$experts slots=$slots $* ==="
  env "$@" GPU_EXPERTS="$experts" KT_PREFETCH_SLOTS="$slots" \
    KT_STORE_SHM=1 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/fb_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/fb_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 3 --tokens 200 --tier 2 \
    --label "$label" --out "$DET" 2>&1 | tail -4
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$RATE" 2>&1 | grep -a "ms/step" || true
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"
OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"

boot F104-plain 104 0 $OFF
boot F100-idle  100 4 $OFF
boot F100-fetch 100 4 $ON $FUSED

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/fb_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== fair_baseline done ==="
