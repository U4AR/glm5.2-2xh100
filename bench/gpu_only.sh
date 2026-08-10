#!/usr/bin/env bash
# GPU-ONLY EXPERTS: full routing without ever touching the CPU.
#
# THE IDEA. `top0` already never touches the CPU -- it substitutes every
# non-resident expert with a resident stand-in -- and costs the 52.90 ms floor,
# but the routing it runs is not the routing the model asked for. Full routing
# (`top8`) keeps every genuine expert and costs 149.48 ms/step, almost all of it
# the CPU computing the ~61% of wanted experts that are not resident.
#
# This mode keeps the genuine expert whenever the LINK had time to carry it, and
# substitutes only the remainder:
#
#     keep_mask &= (resident | prefetch-landed)
#
# so the CPU path is zero BY CONSTRUCTION rather than by 95.6% coverage, and the
# number of landing slots becomes a QUALITY DIAL. Stage H15 prices each landed
# expert at ~5.1 ms/step, so the whole mode is
#
#     step = 52.90 floor + 1.64 predictor + 5.1 x f      f = experts landed/call
#
# and f is bounded by the slot count. Full routing has D ~ 12 distinct experts
# per layer of which ~61% are non-resident, so f_max ~ 7.3: a fully-covered top8
# would cost ~92 ms/step against 149.48 with the CPU. That is the prize --
# **full-quality routing at ~1.6x the speed of the CPU path** -- and every slot
# short of that is a graceful quality degradation rather than a stall.
#
# TWO RULES INVERTED FOR THIS MODE, both deliberate:
#
#   KT_PREFETCH_SELECTIVE=0. The all-or-nothing rule exists because a partially
#   covered layer still pays its CPU call, so partial bytes buy nothing. There is
#   no CPU call here: every landed expert is one fewer substitution, so partial
#   coverage is pure quality. Leaving it on would discard most of the point.
#
#   Slots are swept at CONSTANT VRAM (resident + slots = 104), because a slot and
#   a resident expert cost the same memory. The question is not "are slots good"
#   but "is a dynamic slot worth more than a permanently resident expert" --
#   Stage H12 measured 4 extra residents at only 0.36 ms, so the bar is low.
#
# Tier is per-request, so each boot reports top8 (full routing, the quality
# target), top2 (today's shipped tier) and top0 (the substitute-everything floor)
# without rebooting.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RATE=bench/profile_out/gpu_only_rate.json
DET=bench/profile_out/gpu_only_det.json

# THE TIER MODE COMES FROM A FILE, NOT FROM RUNGLM_TOPK_MODE.
# `RUNGLM_TOPK_MODE` is read by NOTHING -- it appears only in bench scripts. The
# real switch is the sentinel `/tmp/kt_topk_mode` (KT_TOPK_MODE_FILE), and every
# earlier ladder ran at safe2 because that file happened to say so, not because
# the script asked for it. A private file per experiment makes the mode declared
# rather than ambient, and leaves production's own sentinel untouched.
MODEFILE="$SP/kt_topk_mode.gpu_only"
echo safe8 > "$MODEFILE"

boot () {
  local label="$1"; local experts="$2"; local slots="$3"; local predp="$4"; shift 4
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: GPU_EXPERTS=$experts SLOTS=$slots PRED_P=$predp $* ==="
  env "$@" GPU_EXPERTS="$experts" KT_PREFETCH_SLOTS="$slots" KT_PRED_P="$predp" \
    KT_STORE_SHM=1 KT_PREFETCH_BLOCKS=8 MTP=1 MEM_FRACTION=0.94 \
    KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/g_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/g_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; tail -25 "$SP/g_$label.log"; return 1; }
  .venv/bin/python bench/determinism.py --runs 2 --tokens 200 --tier 8 \
    --label "$label" --out "$DET" --save-text "$SP/g_text" 2>&1 | grep -aE "distinct|DETERMIN" || true
  for t in 8 2 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "$label-top$t" --out "$RATE" 2>&1 | grep -a "ms/step" || true
  done
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/g_$label.log" | tail -1 || echo none
  echo "  --- 300 chars @top8 ---"
  head -c 300 "$SP/g_text/$label.txt" 2>/dev/null; echo
}

ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"
FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1"
GO="KT_GPU_ONLY=1 KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2"

# Reference: full routing WITH the CPU, no prefetch. The quality target and the
# 149 ms it currently costs.
boot G-ref  104 0 2 KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0

# The dial. Same total VRAM in every row.
boot G-s4   100 4  8 $ON $FUSED $GO
boot G-s8    96 8  8 $ON $FUSED $GO
boot G-s16   88 16 8 $ON $FUSED $GO

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/g_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== gpu_only done ==="
