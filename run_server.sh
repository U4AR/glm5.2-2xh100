#!/usr/bin/env bash
# Launch GLM-5.2-FP8 on 2x H100 NVL (TP2) + KT-Kernel CPU-GPU heterogeneous MoE.
# Adapted from the GLM-5.2 SGLang+KT tutorial (which targets 8-GPU TP8 / Intel-AMX).
# This box: 2x H100 NVL (96GB), AMD EPYC 9V84 (80 cores, 2 NUMA nodes), 629GB RAM.
set -euo pipefail

VENV=/data/models/RunGLM/.venv
MODEL=/data/models/GLM-5.2-FP8

source "$VENV/bin/activate"   # also exports LD_LIBRARY_PATH=$VENV/lib (hwloc/numa)

# --- runtime env -----------------------------------------------------------
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # older name, harmless
# NOTE: SGLANG_ENABLE_JIT_DEEPGEMM is set further down, coupled to CUDA graphs.
# CUDA-graph capture turns on the NSA indexer's dual-stream path, which calls
# deep_gemm.get_num_sms() -> needs DeepGEMM imported. So graphs ON => DeepGEMM ON.
export TOKENIZERS_PARALLELISM=false
# keep HF from touching the near-full root disk
export HF_HOME=/data/models/RunGLM/.hf
mkdir -p "$HF_HOME"

# --- tunables (the two knobs to adjust on OOM) -----------------------------
# GPU experts/layer: HIGHER = faster AND less CPU RAM, limited by VRAM.
#   48 -> ~556GB CPU experts (RAM headroom ~74GB), ~68GB/card on GPU.
#   Raise toward 54-56 for more speed once the first boot's VRAM use is known.
GPU_EXPERTS=${GPU_EXPERTS:-54}
# mem-fraction = GPU budget for weights+KV. The CUDA-graph capture reserve sits
# INSIDE this budget, so going too LOW starves the KV pool ("Not enough memory,
# increase --mem-fraction-static"). GLM weights ~84GB/card => need >=0.93. 0.94
# leaves ~5.8GB KV headroom and still fits 95.8GB physical with graphs captured.
MEM_FRACTION=${MEM_FRACTION:-0.94}
CPUINFER=${CPUINFER:-72}            # 80 cores, 2 NUMA -> 36/node, leaves headroom
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-8192}
MAX_RUNNING=${MAX_RUNNING:-2}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-2048}
# CUDA graphs ON by default: they collapse the eager per-layer kernel launches
# into one graph replay and TRIPLE decode throughput here (3.3 -> ~8.7 tok/s).
# Decode was per-step launch/serialization-overhead bound, NOT CPU/swap bound.
# Set DISABLE_CUDA_GRAPH=1 only to fall back to the old slow eager path.
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}   # 0=enable graphs (fast), 1=disable
# Only ever decode at batch <= MAX_RUNNING, so cap graph capture there: far less
# capture time + VRAM than the default (captures bs 1..256).
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-1}

CG_FLAG=""
if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
  CG_FLAG="--disable-cuda-graph"
  export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}  # cutlass FP8 GEMM
else
  # Three things are REQUIRED for graph capture to succeed on this setup:
  #  1) DeepGEMM imported  -> NSA dual-stream calls deep_gemm.get_num_sms()
  #  2) --disable-custom-all-reduce -> its CUDA-IPC graph-buffer registration
  #     fails capture ("get_graph_buffer_ipc_meta: invalid argument"); NCCL
  #     all-reduce is fast on 2x H100 NVLink anyway
  #  3) mem-fraction high enough (see MEM_FRACTION above)
  export SGLANG_ENABLE_JIT_DEEPGEMM=1
  CG_FLAG="--cuda-graph-max-bs $CUDA_GRAPH_MAX_BS --disable-custom-all-reduce"
fi

# Dynamic GPU-expert update keeps a FULL-expert copy resident on the host so it
# can re-stage any expert onto the GPU at runtime. On this box that redundant
# host copy (~154GB = the 4050 GPU experts) is what overflows 629GB RAM into NVMe
# swap and caps decode at ~3.4 tok/s. Disabling it leaves only the 202 CPU
# experts/layer (~571GB) on the host -> fits in RAM -> no swap. Default: OFF.
DYN_UPDATE=${DYN_UPDATE:-0}
DYN_FLAG=""; [ "$DYN_UPDATE" = "1" ] && DYN_FLAG="--kt-enable-dynamic-expert-update"

# --- MTP / NEXTN speculative decoding (lever #2) ---------------------------
# Model ships 1 nextn predict layer (layer 78). NEXTN maps GlmMoeDsa ->
# DeepseekV3ForCausalLMNextN draft (server_args.py:285,2701); draft model path
# auto-defaults to --model-path, so no separate draft path needed. Constraint:
# topk MUST be 1 (spec v2), and num_draft_tokens is forced to num_steps+1.
# The nextn draft is a FULL MoE layer loaded ON GPU (kt does NOT offload it),
# so it costs a few GB/card -> pair SPEC_DECODE=1 with a lower GPU_EXPERTS
# (e.g. 48) on first boot to leave VRAM for the draft + its cuda graphs.
# Needs a working tilelang (apache-tvm-ffi pinned to 0.1.11; 0.1.12 aborts).
SPEC_DECODE=${SPEC_DECODE:-0}
SPEC_STEPS=${SPEC_STEPS:-1}              # 1 nextn layer -> 1 draft step
SPEC_TOPK=${SPEC_TOPK:-1}               # must be 1
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-2}  # forced to SPEC_STEPS+1 anyway
SPEC_FLAG=""
if [ "$SPEC_DECODE" = "1" ]; then
  SPEC_FLAG="--speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk $SPEC_TOPK --speculative-num-draft-tokens $SPEC_DRAFT_TOKENS"
fi

echo "GLM-5.2-FP8  TP2  gpu_experts=$GPU_EXPERTS  mem_fraction=$MEM_FRACTION  cpuinfer=$CPUINFER  cuda_graph=$([ "$DISABLE_CUDA_GRAPH" = 1 ] && echo off || echo on)  spec_decode=$([ "$SPEC_DECODE" = 1 ] && echo "NEXTN(steps=$SPEC_STEPS,topk=$SPEC_TOPK,draft=$SPEC_DRAFT_TOKENS)" || echo off)"

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --kt-weight-path "$MODEL" \
  --kt-cpuinfer "$CPUINFER" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-num-gpu-experts "$GPU_EXPERTS" \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \
  $DYN_FLAG \
  --kt-expert-placement-strategy uniform \
  --tp-size 2 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static "$MEM_FRACTION" \
  --kv-cache-dtype fp8_e4m3 \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  $CG_FLAG \
  $SPEC_FLAG \
  --attention-backend nsa \
  --fp8-gemm-backend cutlass \
  --disable-shared-experts-fusion \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name GLM5.2
