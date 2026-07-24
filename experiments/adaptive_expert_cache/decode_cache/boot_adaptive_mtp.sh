#!/usr/bin/env bash
# ADAPTIVE decode-time expert cache + safe2 + MTP depth-3, UNIFORM cold start.
# The cache must climb from uniform (~28 tok/s) toward oracle (~41) ON ITS OWN
# while serving decode traffic. All experts staged in CPU store (evictions
# always CPU-backed) -> host RSS ~400GB (box has 629GB).
cd /data/models/RunGLM
rm -f /tmp/kt_skip_cpu
echo "safe2" > /tmp/kt_topk_mode
export MTP=1
export SPEC_DECODE=1
export SPEC_STEPS=3
export SPEC_DRAFT_TOKENS=4
export SPEC_DRAFT_ATTN=triton
export SGLANG_ENABLE_SPEC_V2=True
export DISABLE_CUDA_GRAPH=0
export CUDA_GRAPH_MAX_BS=1
export KT_HIT_STATS=0
export KT_GPU_PREFILL_THRESHOLD=0
export MEM_FRACTION=0.85
export GPU_EXPERTS=96
export PLACEMENT=uniform
export KT_METHOD=RAWINT4
export KT_RAWINT4_BACKEND=avx512_packed
export TRITON_CACHE_DIR=/data/models/RunGLM/.triton-cache
# --- adaptive cache ---
export KT_ADAPTIVE_DECODE=1
export KT_ADAPTIVE_PERIOD=32
export KT_ADAPTIVE_LAYERS_PER_TICK=2
export KT_ADAPTIVE_MAX_SWAP=24
export KT_ADAPTIVE_MARGIN=1.3
export KT_ADAPTIVE_DECAY=0.98
export KT_ADAPTIVE_MIN_EVENTS=64
exec bash run_server_int4.sh
