#!/usr/bin/env bash
# Are the gathered expert weights CORRECT, or merely self-consistent?
#
# The gather is an independent CUDA kernel reading kt's store through raw
# pointers. The shipped GPU-prefill path reads the same store through kt's own
# C++ write_weights_to_buffer and is known bit-coherent. Two readers, one proven.
# If they disagree about layout -- nibble order, sign convention, group axis --
# every landed expert computes plausible garbage, deterministically, and the only
# symptom is quality quietly dropping. Which is what we see: accept 2.273 vs
# 2.857, repeatable to the step.
#
# The direct byte comparison is awkward (kt does not stage GPU-resident experts,
# so there is no free ground truth in VRAM). But the model can be asked instead,
# with nothing but env vars, because ROUTE and CPUSKIP decompose into three
# arithmetic outcomes for a landed expert:
#
#   X-full  ROUTE=1 CPUSKIP=1   computed once, by the GPU, from gathered weights
#   X-drop  ROUTE=0 CPUSKIP=1   omitted entirely (CPU told to skip, GPU never told
#                               to compute) -- the expert contributes NOTHING
#   X-dbl   ROUTE=1 CPUSKIP=0   counted twice: CPU computes it AND the GPU adds
#                               its gathered copy
#
# The logic, and it is a genuine fork:
#
#   If the gathered weights are RIGHT, X-full is numerically near-exact and should
#   land close to X-base, while X-drop (a real contribution missing) and X-dbl (a
#   real contribution doubled) should both be worse.
#
#   If the gathered weights are WRONG, X-full injects garbage into the residual
#   stream, which is worse than injecting nothing -- so X-drop should BEAT X-full.
#
# X-full is already known to sit far below X-base (2.273 vs 2.857), which on its
# own says the landed experts are not numerically equivalent to what they replace.
# X-drop vs X-full is what separates "wrong weights" from "right weights, wrong
# kernel precision".
#
# All rows carry KT_PREFETCH_FRESH=1, so every one of them is deterministic and
# the accept numbers mean something. Text hashes are saved so the rows can be
# compared against each other afterwards.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
DET=bench/profile_out/gather_truth_det.json
RATE=bench/profile_out/gather_truth_rate.json
TXT=bench/profile_out/gather_truth_txt
rm -rf "$TXT"; mkdir -p "$TXT"

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/x_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/x_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 3 --tokens 200 --tier 2 \
    --label "$label" --out "$DET" --save-text "$TXT" 2>&1 | tail -10
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$RATE" 2>&1 | tail -2
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/x_$label.log" | tail -1 || echo none
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"

boot X-base KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot X-full KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 $FUSED
boot X-drop KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=1 $FUSED
boot X-dbl  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=0 $FUSED

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/x_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== gather_truth done ==="
