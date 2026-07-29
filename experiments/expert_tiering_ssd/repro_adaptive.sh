#!/usr/bin/env bash
# Reproduce the historical numbers on their OWN launcher and protocol.
#
# Every figure quoted from memory -- adaptive 38-40 tok/s, and the ladder rows
# N=32 -> 32.3, N=16 -> 30.3, N=8 -> 27.1, N=24 -> 31-34 converged -- came from
# experiments/adaptive_expert_cache/decode_cache, i.e. run_adaptive.sh. I
# benchmarked run_fast.sh instead, which differs in three ways that all push the
# same direction, so my numbers were guaranteed to land low:
#
#   1. LAUNCHER. run_fast.sh uses STATIC hotcore placement with no adaptive
#      decode cache and no warm start. boot_adaptive_mtp.sh sets
#      KT_ADAPTIVE_DECODE=1, a warm-start oracle mask, and a persisted counter
#      prior at mass 64. Memory records this as uniform 28.2 -> adaptive 38-40,
#      so the cache alone is worth roughly +35%.
#   2. CONVERGENCE. The cache has to converge before it is worth anything: the
#      live ladder logged N=24 as "27 cold -> 31-34 converged". The published
#      protocol is 12-14 convergence passes on one topic, THEN measure. I ran
#      three short warm-up prompts.
#   3. HARNESS. Those numbers came from conv_driver.py -- 500-token essays,
#      tok/s from usage.completion_tokens (correct under MTP, where one stream
#      chunk carries accept_len tokens). I used decbench.py at 300 tokens.
#
# So this replays the historical configuration exactly and asks the only
# question that can distinguish a regression from a protocol mismatch:
#
#   F  TP2 N=24  adaptive, converged  -> historical says 31-34
#   G  TP2 N=96  adaptive, converged  -> historical says 38-40 (the headline)
#   E  TP1 N=24  adaptive, converged  -> the HONEST single-GPU number, measured
#                                         on the same footing as the history it
#                                         is being compared against
#
# If F and G land in their bands, nothing regressed and my earlier single-GPU
# figure was simply measured on a weaker configuration. If they land below,
# the regression is real and E is not the interesting result.
set -uo pipefail
cd /data/models/RunGLM
M=logs/repro_adaptive.log
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

rung() {  # NAME VISIBLE N
  local NAME=$1 VIS=$2 N=$3
  local L="logs/ra_${NAME}.log"
  say ""
  say "=== $NAME : GPUs=[$VIS] GPU_EXPERTS=$N  adaptive + warm start + prior ==="
  kill_server
  rm -f "$L"
  setsid env CUDA_VISIBLE_DEVICES="$VIS" \
    GPU_EXPERTS="$N" WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
    AUTO_PROFILE=0 TP_SIZE="$(echo "$VIS" | awk -F, '{print NF}')" \
    CPUINFER=72 NUMA_NODES="0 1" KT_THREADPOOL_COUNT=2 \
    MEM_FRACTION=0.85 MAX_TOTAL_TOKENS=4096 \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    bash experiments/adaptive_expert_cache/decode_cache/boot_adaptive_mtp.sh \
    > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED"; grep -E "Error|Traceback|out of memory|Killed" "$L" | tail -5 | tee -a "$M"; return; }

  # The published protocol: converge the cache first, then measure. Reporting a
  # cold number here would repeat the exact mistake this script exists to fix.
  say "   -- 14 convergence passes (cache must converge before it is worth anything)"
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 14 500 llm \
    2>&1 | tail -14 | sed 's/^/      /' | tee -a "$M"
  say "   -- 3 measure passes"
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 3 500 llm \
    2>&1 | tail -5 | sed 's/^/      /' | tee -a "$M"
  say "   VRAM $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
  say "   host RSS $(free -g | awk 'NR==2{print $3}')GB"
  say "   $(grep -oE 'accept len: [0-9.]+' "$L" | tail -40 | awk '{s+=$3;n++} END{printf "accept %.3f (n=%d)", s/n, n}')"
}

say "###### adaptive repro, historical launcher + protocol $(date -u) ######"
say "targets: F(TP2 N=24) 31-34 | G(TP2 N=96) 38-40 | E(TP1 N=24) unknown, first measurement"
rung "F_tp2_n24"  "0,1" 24
rung "G_tp2_n96"  "0,1" 96
rung "E_tp1_n24"  "0"   24
say ""
say "###### done $(date -u) ######"
