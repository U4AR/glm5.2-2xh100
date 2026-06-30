#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_fast.sh — high-speed GLM-5.2 decode on 2x H100 NVL via top-K expert
# substitution (the "top-2" trick). Wraps run_server_int4.sh.
#
# Idea: GLM-5.2 routes each token to 8 of 256 experts; most live in CPU RAM and
# the per-layer CPU<->GPU sync is the decode bottleneck. We KEEP the genuine top-K
# experts (which carry the routing "identity") and SUBSTITUTE the low-weight tail
# with the best GPU-resident experts (ranked by the router's own scores). Fewer
# CPU experts per layer => shorter critical path => faster decode.
#
# Measured on 2x H100 NVL (TP2, GPU_EXPERTS=104), decode tok/s (5-run median):
#   baseline (all 8 as routed)   14.7   1.00x   reference
#   KEEP=4 (substitute 4)        18.5   1.25x   clean, no quality loss observed
#   KEEP=2 (substitute 6)        22.0   1.49x   DEFAULT; rare repetition loops
#   KEEP=0 + CPU-skip            29.2   1.98x   degenerates — NOT recommended
#
# The mechanism is a tiny logical reroute (sglang deepseek_v2.py DeepseekV2MoE);
# it is gated by the sentinel file /tmp/kt_topk_mode so no rebuild is needed.
#
# Usage:
#   ./run_fast.sh                 # KEEP=2 (top-2), the high-speed default
#   KEEP=4 ./run_fast.sh          # safer quality, 1.25x
#   KEEP=0 ./run_fast.sh          # max speed, also enables the CPU-skip (lower quality)
#   MODE=off ./run_fast.sh        # disable substitution -> plain baseline
#   MTP=1 ./run_fast.sh           # + NEXTN/MTP speculative decode (depth-3) on top
#
# MTP (2026-06-30): MTP now works UNDER CUDA GRAPHS at any KEEP (the kt verify-
# batch buffer bug is fixed in cuda_graph_runner.py). Stacks with top-2:
#   KEEP=2          no-MTP   22.0
#   KEEP=2 MTP=1    depth-1  29.8   accept 1.7
#   KEEP=2 MTP=1    depth-3  34.1   accept 2.5   <- peak; MTP=1 default
#   KEEP=2 MTP=1    depth-5  28.9   accept 2.6   (verify cost > accept gain)
#   normal MTP=1    depth-3  ~      also coherent (14.7 base). depth-3 is the sweet spot.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

KEEP=${KEEP:-2}                 # number of genuine top experts to keep
MODE=${MODE:-sub}               # sub = substitute the tail; off = baseline
MTP=${MTP:-0}                   # 1 = enable NEXTN/MTP speculative decode (depth-3)

# --- write the top-K sentinel the model worker reads at import -------------
if [ "$MODE" = "off" ]; then
  rm -f /tmp/kt_topk_mode /tmp/kt_skip_cpu
  echo "[run_fast] substitution DISABLED (plain baseline)."
else
  printf 'sub%s' "$KEEP" > /tmp/kt_topk_mode
  echo "[run_fast] top-K substitution ENABLED: keep genuine top-$KEEP, substitute the rest (/tmp/kt_topk_mode=sub$KEEP)."
  # The CPU-path skip (extra ~1.2x) is ONLY correct when every routed expert is
  # GPU-resident, i.e. KEEP=0. Enabling it with KEEP>0 drops the kept CPU experts
  # = the dominant signal = garbage, so we only arm it for KEEP=0.
  if [ "$KEEP" = "0" ]; then
    touch /tmp/kt_skip_cpu
    echo "[run_fast] KEEP=0 -> CPU submit/sync skip ARMED (/tmp/kt_skip_cpu). Max speed, expect quality drift."
  else
    rm -f /tmp/kt_skip_cpu
  fi
fi

# --- the winning INT4 recipe (see README) ----------------------------------
export MODEL=${MODEL:-/cache/nvme0/models/GLM-5.2-W4AFP8}
export KT_METHOD=${KT_METHOD:-RAWINT4}
export KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-/cache/nvme0/models/GLM-5.2-W4AFP8}
export KT_RAWINT4_BACKEND=${KT_RAWINT4_BACKEND:-avx512_packed}
export GPU_EXPERTS=${GPU_EXPERTS:-104}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-4096}
export MEM_FRACTION=${MEM_FRACTION:-0.94}

# --- optional NEXTN/MTP speculative decode (works under CUDA graphs now) ----
if [ "$MTP" = "1" ]; then
  export SPEC_DECODE=1
  export SPEC_STEPS=${SPEC_STEPS:-3}            # depth-3 = peak net tok/s here
  export SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-4}
  export SPEC_DRAFT_ATTN=${SPEC_DRAFT_ATTN:-triton}  # flashmla/fa3/compressed draft crash; triton works
  echo "[run_fast] MTP ENABLED: NEXTN spec-decode steps=$SPEC_STEPS draft_tokens=$SPEC_DRAFT_TOKENS draft_attn=$SPEC_DRAFT_ATTN."
fi

exec bash run_server_int4.sh
