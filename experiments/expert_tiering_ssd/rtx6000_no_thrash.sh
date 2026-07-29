#!/usr/bin/env bash
# Follow-up to emulate_rtx6000.sh: was 17.5 tok/s the machine, or the thrash?
#
# The first emulation ran KT_TIER_DYNAMIC=1 + KT_ADAPTIVE_DECODE=1 at GPU=25 /
# RAM=64, i.e. 89 of 256 experts reachable per layer. The server log shows the
# tier mover never quiescing -- every layer, every period, for 414+ events:
#
#   [kt-tier] layer=20 gpu_swaps=2 ram_evict=6 ram_promote=6
#             cov gpu=0.696 ram=0.200 ssd=0.104  took=220ms
#   [kt-tier-roi] layer=20 promoted_prev=6 called=6 (100%) evicted_again=1
#
# evicted_again > 0 means it is throwing out experts it wants back. The working
# set does not fit in 64 RAM slots, so movement is permanent rather than
# transient. That is consistent with the three measured passes DECLINING
# (20.8 -> 17.5 -> 16.2) with accept falling 2.87 -> 2.28, and with the recorded
# finding that ~85% of movement cost is paid in accept length, not step rate.
#
# So this measures the same box with movement OFF. Two rungs:
#
#   H  GPU=25 RAM=64, static     same residency as the thrashing run, so the
#                                delta is movement and nothing else.
#   I  GPU=25 RAM=84, static     the largest RAM tier that still fits 140 GB.
#                                Rung H measured 116 GB at RAM=64, i.e. ~1.05
#                                GB per slot on top of a ~49 GB fixed floor, so
#                                84 slots lands near 137 GB. If H is fast and I
#                                is faster, the recommendation is "size the RAM
#                                tier to the box and never move".
#
# Movement requires KT_ADAPTIVE_DECODE=0 as well: boot_tiered.sh force-enables
# KT_TIER_DYNAMIC whenever the adaptive cache is on, because the adaptive cache
# evicts into weights that the static tier never staged (silent garbage). Both
# rungs therefore run the warm-start oracle placement, fixed for the whole run.
#
# Same caveat as the first emulation: an H100 stands in for the Blackwell card,
# so these remain UPPER BOUNDS on bandwidth and on kernel selection.
set -uo pipefail
cd /data/models/RunGLM
M=logs/rtx6000_no_thrash.log
say() { echo "$@" | tee -a "$M"; }

kill_server() {
  for p in $(pgrep -f "[s]glang.launch_server"); do kill $p 2>/dev/null; done
  for i in $(seq 1 40); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "[s]glang"); do kill -9 $p 2>/dev/null; done
  sleep 8
}

wait_ready() {
  local L=$1
  sleep 30
  for i in $(seq 1 400); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && return 0
    grep -qE "Traceback|CUDA out of memory|Killed" "$L" && break
    pgrep -f "[s]glang.launch_server" >/dev/null || break
    sleep 10
  done
  return 1
}

rung() {  # NAME GPU_N RAM_N
  local NAME=$1 GPU_N=$2 RAM_N=$3
  local L="logs/nothrash_${NAME}.log"
  say ""
  say "=== $NAME : GPU=$GPU_N RAM=$RAM_N ($((GPU_N+RAM_N))/256 reachable), NO movement ==="
  kill_server
  rm -f "$L"
  setsid taskset -c 0-15 env CUDA_VISIBLE_DEVICES=0 \
    GPU_EXPERTS="$GPU_N" KT_RAM_EXPERTS="$RAM_N" \
    KT_TIER_DYNAMIC=0 KT_ADAPTIVE_DECODE=0 \
    AUTO_PROFILE=0 TP_SIZE=1 CPUINFER=12 NUMA_NODES="0" KT_THREADPOOL_COUNT=1 \
    MEM_FRACTION=0.85 MAX_TOTAL_TOKENS=8192 \
    MAX_RUNNING=8 CUDA_GRAPH_MAX_BS=8 \
    KT_TOPK_MODE=safe2 WARM_START=1 \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    bash experiments/expert_tiering_ssd/boot_tiered.sh > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED"; grep -E "Error|Traceback|out of memory|Killed" "$L" | tail -6 | tee -a "$M"; return; }

  say "   host RSS $(free -g | awk 'NR==2{print $3}')GB   VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
  # Static placement, so there is nothing to converge -- but two warm passes
  # still cover CUDA-graph replay and page-cache warmth on the SSD tier.
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 2 500 llm >/dev/null 2>&1
  say "   single stream (5 passes -- watching for the decline, not just the median):"
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 5 500 llm \
    2>&1 | grep -E "^pass" | sed 's/^/      /' | tee -a "$M"
  say "   batch scaling:"
  .venv/bin/python bench/intelligence_tier/batch_bench.py GLM5.2 256 1,2,4,8 \
    2>&1 | tail -12 | sed 's/^/      /' | tee -a "$M"
  say "   movement events during run: $(grep -ac '\[kt-tier\]' "$L" 2>/dev/null || echo 0)  (expect 0)"
}

say "###### RTX PRO 6000 emulation, movement OFF $(date -u) ######"
say "baseline to beat: 20.8/17.5/16.2 tok/s declining, with movement ON"
rung "H_ram64" 25 64
rung "I_ram84" 25 84
say ""
say "###### done $(date -u) ######"
