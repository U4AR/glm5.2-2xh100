#!/usr/bin/env bash
# ADAPTIVE decode-time expert cache + safe2 + MTP depth-3, UNIFORM cold start.
# The cache must climb from uniform (~28 tok/s) toward oracle (~41) ON ITS OWN
# while serving decode traffic. All experts staged in CPU store (evictions
# always CPU-backed) -> host RSS ~400GB (box has 629GB).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
source "$REPO/config.sh"
# Scheduler workers intentionally receive a scrubbed environment, so the
# routing control files must retain their historical /tmp defaults.
RUNTIME_DIR="${RUNTIME_DIR:-/tmp}"
KT_TOPK_MODE_FILE="${KT_TOPK_MODE_FILE:-/tmp/kt_topk_mode}"
KT_SKIP_CPU_FILE="${KT_SKIP_CPU_FILE:-/tmp/kt_skip_cpu}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO/.triton-cache}"
mkdir -p "$RUNTIME_DIR" "$TRITON_CACHE_DIR"
rm -f "$KT_SKIP_CPU_FILE"
echo "safe2" > "$KT_TOPK_MODE_FILE"
export MTP=1
export SPEC_DECODE=1
export SPEC_STEPS=3
export SPEC_DRAFT_TOKENS=4
export SPEC_DRAFT_ATTN=triton
export SGLANG_ENABLE_SPEC_V2=True
export DISABLE_CUDA_GRAPH=0
export CUDA_GRAPH_MAX_BS=1
export KT_HIT_STATS=0
export KT_GPU_PREFILL_THRESHOLD="${KT_GPU_PREFILL_THRESHOLD:-0}"
# Overridable so the hardware profiler / a single-GPU run can lower it; the
# validated 2xH100 adaptive default stays 0.85.
export MEM_FRACTION="${MEM_FRACTION:-0.85}"
# Per-layer resident expert count. The profiler auto-sizes this from per-card
# VRAM for non-2xH100 topologies (e.g. ~24 on a single H100 under TP=1); 96 is
# the validated 2xH100 default when nothing set it.
export GPU_EXPERTS=${GPU_EXPERTS:-96}
# --- warm start: if a persisted hot-core ranking exists, boot the GPU already
# holding the hottest-N experts (any N) instead of a uniform cold start. The
# adaptive cache then only has to track per-workload drift, not climb from 0.5
# coverage. Set WARM_START=0 to force the old uniform cold start. ---
RANK_PT="experiments/adaptive_expert_cache/decode_cache/hot_core_ranking.pt"
PRIOR_PT="experiments/adaptive_expert_cache/decode_cache/hot_core_prior.pt"
if [ "${WARM_START:-1}" = "1" ] && [ -f "$RANK_PT" ]; then
  WARM_MASK="$RUNTIME_DIR/kt_warm_mask_${GPU_EXPERTS}.pt"
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/build_hot_core.py \
      mask "$GPU_EXPERTS" "$WARM_MASK" || { echo "warm-mask build failed, falling back to uniform"; unset WARM_MASK; }
  if [ -n "${WARM_MASK:-}" ]; then
    export PLACEMENT=oracle
    export KT_ORACLE_MASK_PT="$WARM_MASK"
    if [ -f "$PRIOR_PT" ]; then
      export KT_ADAPTIVE_PRIOR_PT="$REPO/$PRIOR_PT"
      export KT_ADAPTIVE_PRIOR_MASS="${KT_ADAPTIVE_PRIOR_MASS:-64}"
    fi
    echo "[warm-start] booting from hot-core ranking -> $WARM_MASK (N=$GPU_EXPERTS/layer)"
  else
    export PLACEMENT=uniform
  fi
else
  export PLACEMENT=uniform
fi
export KT_METHOD=RAWINT4
export KT_RAWINT4_BACKEND=avx512_packed
export TRITON_CACHE_DIR
# --- adaptive cache ---
export KT_ADAPTIVE_DECODE=1
export KT_ADAPTIVE_PERIOD="${KT_ADAPTIVE_PERIOD:-32}"
export KT_ADAPTIVE_LAYERS_PER_TICK="${KT_ADAPTIVE_LAYERS_PER_TICK:-2}"
export KT_ADAPTIVE_MAX_SWAP="${KT_ADAPTIVE_MAX_SWAP:-24}"
export KT_ADAPTIVE_MARGIN="${KT_ADAPTIVE_MARGIN:-1.3}"
export KT_ADAPTIVE_DECAY="${KT_ADAPTIVE_DECAY:-0.98}"
export KT_ADAPTIVE_MIN_EVENTS="${KT_ADAPTIVE_MIN_EVENTS:-64}"
exec bash run_server_int4.sh
