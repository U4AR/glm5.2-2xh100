#!/usr/bin/env bash
# Re-establish the predictor baseline, then test the stride -- both of which
# today's runs got wrong, in ways that only showed up on inspection.
#
# TWO DEFECTS BEING CORRECTED HERE:
#
#   1. KT_PRED_FUSED_STACK was never set, so `_kt_fuse_gates` never ran and the
#      75 lookahead gate GEMMs were NEVER folded into the layers' own gates.
#      Every predictor cost quoted today (1.81 ms/step) is the UNSTACKED cost.
#      The stacked path is the one the fused design was built around and it has
#      not been measured on this box.
#
#   2. KT_PRED_LAYER_STRIDE was implemented only inside _kt_fuse_gates, so with
#      (1) in force it was inert. The run meant to test it reproduced the
#      unstrided number to 0.26 ms -- a no-op wearing the costume of a result.
#      It now applies in the unstacked path too, keyed on layer id in both so
#      the two select the same layers.
#
# Rows, each its own boot, everything else identical (GPU-only, cache and
# prefetch both live, 1 slot + reuse, blocks 8):
#
#   SS-unstacked   stride 1, no gate fusion   -- reproduces today's ~56.8 and
#                                                anchors the comparison
#   SS-stacked     stride 1, gate fusion ON   -- what the fused design costs
#   SS-s4          stride 4, gate fusion ON   -- the 54 ms candidate, finally
#                                                actually strided
#
# The prediction, recorded before the run so it can be contradicted: stacking
# should remove ~2 ms of gate GEMM launches, and stride 4 should remove ~3/4 of
# whatever selection cost remains. If SS-stacked does not beat SS-unstacked,
# the "fold the GEMM into the network" idea is worth less than its docstring
# claims and the 54 ms budget needs rebuilding from scratch rather than patching.
#
# A VRAM hazard, known and bounded: _kt_fuse_gates allocates a 225 MiB gate
# stack. It stages through host memory so the peak is max(old, new) rather than
# old + new, but weight loading still finishes with ~100 MiB spare at 100
# resident experts, and the fused rows have died 28 MiB short before now
# (fragmentation, not exhaustion). If a stacked row fails to boot, that is the
# cause, and GPU_EXPERTS is the only lever on it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"

MODEFILE="$SP/kt_topk_mode.stacked"
echo safe8 > "$MODEFILE"
OUT=bench/profile_out/stacked_stride.json
mkdir -p bench/profile_out

boot () {
  local log="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $log: $* ==="
  env "$@" \
    KT_GPU_ONLY=1 KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
    KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
    KT_PREFETCH_SELECTIVE=0 KT_PREFETCH_REUSE=2 KT_PREFETCH_SLOTS=1 \
    KT_PRED_P=8 KT_PREFETCH_BLOCKS=8 \
    KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=128 KT_ADAPTIVE_LAYERS_PER_TICK=2 \
    KT_ADAPTIVE_MAX_SWAP=8 KT_TIER_COUNT_MODE=top8 \
    GPU_EXPERTS="${GPU_EXPERTS:-100}" KT_STORE_SHM=1 MTP=1 \
    MEM_FRACTION="${MEM_FRACTION:-0.94}" KT_TOPK_MODE_FILE="$MODEFILE" \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/$log.log" 2>&1 &
  local i
  for i in $(seq 1 400); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed|Bus error" "$SP/$log.log" && { echo "BOOTFAIL $log"; grep -aE "CUDA out of memory|Bus error" "$SP/$log.log" | tail -3; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $log"; return 1; }
  # PROOF THE FLAGS TOOK. A stacked row with no "stacked" line, or a strided row
  # whose fused count equals the layer count, is inert -- print it rather than
  # discover it afterwards, which is the mistake this whole script exists to fix.
  grep -a "kt-pred-fused] stacked" "$SP/$log.log" | tail -1 || echo "  (UNSTACKED: no gate fusion)"
  return 0
}

row () {   # row <log> <label>
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 8 --runs 1 \
    --tokens 1400 --label "warm-$2" --out "$SP/warm_$2.json" 2>&1 | grep -a "ms/step" || true
  local t
  for t in 8 0; do
    .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
      --tokens 200 --label "$2-top$t" --out "$OUT" 2>&1 | grep -a "ms/step" || true
  done
  grep -a "kt-prefetch] step" "$SP/$1.log" | tail -1
}

boot SS_un  KT_PRED_FUSED_STACK=0 KT_PRED_LAYER_STRIDE=1 && row SS_un  SS-unstacked
boot SS_st  KT_PRED_FUSED_STACK=1 KT_PRED_LAYER_STRIDE=1 && row SS_st  SS-stacked
if boot SS_s4 KT_PRED_FUSED_STACK=1 KT_PRED_LAYER_STRIDE=4; then
  row SS_s4 SS-s4
  # Only the candidate gets a coherence run: three layers in four lose their
  # fetch here, and that is a QUALITY risk, so a speed number alone is not a
  # result. top0 is the calibration case.
  TOKENS=1200 bash bench/coherence_run.sh ss4 8 0
fi

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ss_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== stacked_stride done ==="
