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
# DEFAULT (2026-06-30): top-2 substitution + NEXTN/MTP depth-3 = ~34 tok/s.
#   ./run_fast.sh                 # KEEP=2 + MTP depth-3  -> 34.1 tok/s  (DEFAULT)
#   MTP=0 ./run_fast.sh           # KEEP=2, no MTP        -> 22.0 tok/s
#   KEEP=4 ./run_fast.sh          # safer quality + MTP   -> ~ (1.25x base + MTP)
#   KEEP=0 ./run_fast.sh          # max speed + MTP       -> ~40 tok/s, quality drift
#   MODE=off ./run_fast.sh        # plain routing + MTP   -> 17.5 tok/s
#   MODE=off MTP=0 ./run_fast.sh  # plain baseline        -> 14.7 tok/s (reference)
#
# Decode tok/s, 2x H100 NVL, TP2, GPU_EXPERTS=104 (all coherent under CUDA graphs
# except KEEP=0 which drifts). MTP works UNDER CUDA GRAPHS at any KEEP now — the kt
# verify-batch buffer bug is fixed in cuda_graph_runner.py (see BLOG_MTP_CUDAGRAPH.md).
#
#   config                     tok/s   gain vs its no-MTP base   accept
#   plain baseline (no MTP)     14.7    reference                 1.0
#   plain + MTP depth-3         17.5    +19%                      ~2.0
#   KEEP=2 (top-2, no MTP)      22.0    1.49x over baseline       1.0
#   KEEP=2 + MTP depth-1        29.8    +36% over KEEP=2          1.7
#   KEEP=2 + MTP depth-3        34.1    +55% over KEEP=2  <-DEF   2.5
#   KEEP=2 + MTP depth-5        28.9    (verify cost > gain)      2.6
#   KEEP=0 + MTP depth-3       ~40      fastest, quality drift    ~2.0
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

KEEP=${KEEP:-2}                 # number of genuine top experts to keep
MODE=${MODE:-sub}               # sub = substitute the tail; off = baseline
MTP=${MTP:-1}                   # 1 = NEXTN/MTP speculative decode (depth-3); DEFAULT ON

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
# Paths come from config.sh (the one place to edit them); all env-overridable.
source "$(dirname "$0")/config.sh"
export MODEL=${MODEL:-$W4AFP8_MODEL}
export KT_METHOD=${KT_METHOD:-RAWINT4}
export KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-$MODEL}
export KT_RAWINT4_BACKEND=${KT_RAWINT4_BACKEND:-avx512_packed}
export GPU_EXPERTS=${GPU_EXPERTS:-104}
# KV pool: 4096 was a short-benchmark cap that silently truncated real chats at
# ~3.6k total tokens (finish_reason "length"). MLA fp8 KV is ~43.9 KB/token.
# NOTE: the KV pool is carved from the mem_fraction_static budget, but the
# (1-mem_fraction) slack is what CUDA-graph capture + the NEXTN/MTP draft + the
# flashmla prefill workspace need at warmup. Pushing mem_fraction to 0.97 and the
# pool to the full 131072 (5.5GB) OOM'd there (only 3.3GB slack left). With 104
# GPU experts, weights+fixed ~= 87GB, so ~82k tokens (3.6GB pool) is the practical
# max that still leaves the ~5GB runtime headroom the old 0.94/4096 config had.
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-81920}
export MEM_FRACTION=${MEM_FRACTION:-0.95}

# --- optional NEXTN/MTP speculative decode (works under CUDA graphs now) ----
if [ "$MTP" = "1" ]; then
  export SPEC_DECODE=1
  export SPEC_STEPS=${SPEC_STEPS:-3}            # depth-3 = peak net tok/s here
  export SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-4}
  export SPEC_DRAFT_ATTN=${SPEC_DRAFT_ATTN:-triton}  # flashmla/fa3/compressed draft crash; triton works
  echo "[run_fast] MTP ENABLED: NEXTN spec-decode steps=$SPEC_STEPS draft_tokens=$SPEC_DRAFT_TOKENS draft_attn=$SPEC_DRAFT_ATTN."
fi

exec bash run_server_int4.sh
