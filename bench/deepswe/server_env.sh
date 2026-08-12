#!/usr/bin/env bash
# THE single source of truth for this benchmark's server configuration.
#
# Both the launcher and watchdog.sh source this file, because duplicating the
# config in two places has now cost two runs:
#   * a stale MEM_FRACTION=0.88 restarted a healthy server into a boot loop
#   * a stale MAX_TOTAL_TOKENS=81920 (and no CONTEXT_LENGTH) silently shrank the
#     context from 131072 back to 81920 on every watchdog restart, killing a
#     200-step agent whose conversation was 125,447 tokens
# Change values HERE and nowhere else.

export GPU_EXPERTS=80              # 88 -> 80 buys the VRAM for the larger KV pool
export MEM_FRACTION=0.94           # floor is ~0.90 (weights); BELOW that it will not boot
export CONTEXT_LENGTH=131072
# The pool must EXCEED the context length, not equal it: it holds the whole
# sequence plus working space plus MTP's per-draft-token slots. With pool ==
# context a single long request had zero slack, nothing to retract, and died with
# "Out of memory even after retracting all other requests" 24 times in a row.
export MAX_TOTAL_TOKENS=163840     # 1.25x CONTEXT_LENGTH
export CHUNKED_PREFILL=2048
export MAX_RUNNING=8               # >2, so client retries cannot starve the agent
export SLEEP_ON_IDLE=0             # CRITICAL: 1 flushes the prefix cache between
                                   # agent steps -> 11x slowdown (100 vs 9 s/step)
export KT_GPU_PREFILL_THRESHOLD=0  # GPU prefill + a large pool together OOM
export TRITON_CACHE_DIR=/data/triton-cache
