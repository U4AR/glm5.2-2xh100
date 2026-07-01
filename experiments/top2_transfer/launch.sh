#!/usr/bin/env bash
# Wrapper to boot the GLM-5.2 INT4 server with experiment env overrides.
# Usage: launch.sh <logfile> [extra KEY=VAL ...]
# Repo root = two levels up from experiments/top2_transfer/. Paths via config.sh.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
source "$REPO/config.sh"
LOG="$1"; shift
export MODEL="${MODEL:-$W4AFP8_MODEL}"
export KT_METHOD=RAWINT4
export KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-$MODEL}"
export KT_RAWINT4_BACKEND=avx512_packed
export GPU_EXPERTS=104
export MAX_TOTAL_TOKENS=4096
export MEM_FRACTION=0.94
export CPUINFER=72
for kv in "$@"; do export "$kv"; done
bash run_server_int4.sh > "$LOG" 2>&1 < /dev/null
