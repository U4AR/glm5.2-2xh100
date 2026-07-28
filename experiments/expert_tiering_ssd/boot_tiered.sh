#!/usr/bin/env bash
# Three-tier expert store: GPU VRAM / host RAM / NVMe SSD.
#
#   GPU_EXPERTS=N      experts per layer resident in VRAM
#   KT_RAM_EXPERTS=M   experts per layer staged in the kt CPU store (host RAM)
#   -> the remaining 256-N-M live ONLY on disk and are never fetched on the
#      critical path; when routed they are substituted with their nearest
#      equivalent among the resident experts.
#
# Host RAM cost is ~1.42 GiB per RAM-tier slot across the model, so
# KT_RAM_EXPERTS=32 runs in ~45 GB instead of the ~390 GB the two-tier build
# needs. Boot is also much faster: only the staged experts are read off disk.
#
# Knobs specific to this experiment:
#   KT_TIER_FILL_POOL  gpu (default) | resident -- who may stand in for a
#                      substituted slot. `gpu` is the shipped contract: a
#                      substituted slot must never reach the CPU expert path.
#   KT_TIER_COUNT_MODE top2 (default) | top8 | top8w -- which routed experts
#                      vote in the cache-update counters
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source "$REPO/config.sh"

RUNTIME_DIR="${RUNTIME_DIR:-/tmp}"
KT_TOPK_MODE_FILE="${KT_TOPK_MODE_FILE:-/tmp/kt_topk_mode}"
KT_SKIP_CPU_FILE="${KT_SKIP_CPU_FILE:-/tmp/kt_skip_cpu}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO/.triton-cache}"
mkdir -p "$RUNTIME_DIR" "$TRITON_CACHE_DIR"
rm -f "$KT_SKIP_CPU_FILE"
echo "${KT_TOPK_MODE:-safe2}" > "$KT_TOPK_MODE_FILE"

export MTP=1
export SPEC_DECODE=1
export SPEC_STEPS=3
export SPEC_DRAFT_TOKENS=4
export SPEC_DRAFT_ATTN=triton
export SGLANG_ENABLE_SPEC_V2=True
export DISABLE_CUDA_GRAPH=0
export CUDA_GRAPH_MAX_BS=1
export KT_HIT_STATS=0
# Bulk GPU prefill streams EVERY expert out of the kt CPU store, which under a
# RAM cap no longer holds the SSD tier. kt_ep_wrapper forces this to 0 anyway
# when KT_RAM_EXPERTS is set; make it explicit here.
export KT_GPU_PREFILL_THRESHOLD=0
export MEM_FRACTION="${MEM_FRACTION:-0.85}"
export GPU_EXPERTS=${GPU_EXPERTS:-96}
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"

# --- tier sizes ---
export KT_RAM_EXPERTS="${KT_RAM_EXPERTS:-32}"
export KT_TIER_FILL_POOL="${KT_TIER_FILL_POOL:-gpu}"
export KT_TIER_COUNT_MODE="${KT_TIER_COUNT_MODE:-top2}"

# The RAM tier is seeded from the same committed per-layer ranking the GPU
# warm start slices, so the two tiers are consecutive slices of ONE order
# (hottest -> VRAM, next-hottest -> RAM, tail -> SSD) instead of two unrelated
# choices. Without this the RAM tier falls back to lowest-id-first.
RANK_PT="$REPO/experiments/adaptive_expert_cache/decode_cache/hot_core_ranking.pt"
PRIOR_PT="$REPO/experiments/adaptive_expert_cache/decode_cache/hot_core_prior.pt"
if [ -f "$RANK_PT" ]; then
  export KT_HOTCORE_RANKING_PT="$RANK_PT"
fi

if [ "${WARM_START:-1}" = "1" ] && [ -f "$RANK_PT" ]; then
  WARM_MASK="$RUNTIME_DIR/kt_warm_mask_${GPU_EXPERTS}.pt"
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/build_hot_core.py \
      mask "$GPU_EXPERTS" "$WARM_MASK" || { echo "warm-mask build failed, falling back to hotcore"; unset WARM_MASK; }
  if [ -n "${WARM_MASK:-}" ]; then
    export PLACEMENT=oracle
    export KT_ORACLE_MASK_PT="$WARM_MASK"
    [ -f "$PRIOR_PT" ] && export KT_ADAPTIVE_PRIOR_PT="$PRIOR_PT" \
                       && export KT_ADAPTIVE_PRIOR_MASS="${KT_ADAPTIVE_PRIOR_MASS:-64}"
  else
    export PLACEMENT=hotcore
  fi
else
  export PLACEMENT=hotcore
fi

export KT_METHOD=RAWINT4
export KT_RAWINT4_BACKEND=${KT_RAWINT4_BACKEND:-avx512_packed}
export TRITON_CACHE_DIR

# Counters stay on: they are the demand signal, dumped per tick for the Phase 1
# promotion policy. The residency tick itself early-returns while KT_RAM_EXPERTS
# is set (static tiers in Phase 0).
# Both are overridable so an A/B can turn the tick off entirely. Note the
# no-colon form on the dump path: KT_ADAPTIVE_COUNTS_DUMP_PT= (explicitly
# empty) must survive as empty, which ${VAR:-default} would silently replace.
export KT_ADAPTIVE_DECODE="${KT_ADAPTIVE_DECODE:-1}"
export KT_ADAPTIVE_PERIOD="${KT_ADAPTIVE_PERIOD:-32}"
export KT_ADAPTIVE_COUNTS_DUMP_PT="${KT_ADAPTIVE_COUNTS_DUMP_PT-$RUNTIME_DIR/kt_tier_counts.pt}"

# --- dynamic placement -----------------------------------------------------
# KT_TIER_DYNAMIC=1  count-based two-cut movement
# KT_ENERGY=1        energy-driven movement (two timescales, rarity bonus,
#                    sticky decay). Both need the rebuilt kt-kernel, which
#                    supplies per-expert promote/evict.
export KT_TIER_DYNAMIC="${KT_TIER_DYNAMIC:-0}"
export KT_ENERGY="${KT_ENERGY:-0}"
export KT_ENERGY_PERIOD="${KT_ENERGY_PERIOD:-4}"
export KT_ENERGY_HOLD_STEPS="${KT_ENERGY_HOLD_STEPS:-4}"
export KT_ENERGY_MAX_MOVES="${KT_ENERGY_MAX_MOVES:-4}"
export KT_ENERGY_GPU="${KT_ENERGY_GPU:-0}"
export KT_ENERGY_REPORT_PT="${KT_ENERGY_REPORT_PT:-$RUNTIME_DIR/kt_energy_report.pt}"
# Runtime promotion reads single experts off disk, so hold the safetensors
# mmaps open rather than paying ~200ms to recreate the loader each time.
if [ "${KT_TIER_DYNAMIC}" = "1" ] || [ "${KT_ENERGY}" = "1" ]; then
  export KT_TIER_KEEP_LOADER=1
fi

echo "[tier] GPU=$GPU_EXPERTS RAM=$KT_RAM_EXPERTS SSD=$((256 - GPU_EXPERTS - KT_RAM_EXPERTS)) per layer"
echo "[tier] fill_pool=$KT_TIER_FILL_POOL count_mode=$KT_TIER_COUNT_MODE mem_fraction=$MEM_FRACTION"
exec bash run_server_int4.sh
