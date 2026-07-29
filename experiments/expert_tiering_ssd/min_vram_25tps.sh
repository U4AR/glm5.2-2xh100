#!/usr/bin/env bash
# What is the SMALLEST single-GPU VRAM footprint that still decodes at 25 tok/s?
#
# Measured on 1 card (TP1) so the number is a card spec, not a per-card share of
# a pair. Protocol copies the historical low-VRAM ladder exactly, because that is
# the only way the footprints mean anything:
#
#   MEM_FRACTION=0.60 + MAX_TOTAL_TOKENS=4096
#     sglang sizes its KV pool to fill mem_fraction, so on a 95 GiB card an
#     uncapped run reports ~90 GiB used at EVERY N and measures nothing. Capping
#     total tokens at 4096 makes nvidia-smi's number track the expert budget,
#     which is what a prospective buyer actually needs to know.
#   adaptive + warm start + prior, 14 convergence passes, then 3 measure passes
#     The cache is worth nothing until converged: rung G started at 23.5 tok/s
#     cold and settled at 39. Reporting a cold number here would understate every
#     rung by a third and put the 25 tok/s crossing in the wrong place.
#
# N=24 comes from rung E. This adds 16, 8 and 4 -- 4 is the floor, since N=0 is
# an empty-resident crash, not a configuration.
#
# Expect the curve to be FLAT, and treat that as the finding rather than as a
# failed sweep: on 2xL40 hotcore N=8/16/30 measured 18.19/20.33/20.11 tok/s
# across a 2x range in CPU trips per token, because ~97% of the per-step cost is
# a fixed per-layer submit+sync toll that coverage cannot touch. If VRAM is a
# weak lever here too, the honest answer to "how much VRAM for 25 tok/s" is
# "past a low threshold, VRAM is not what decides it".
set -uo pipefail
cd /data/models/RunGLM
M=logs/min_vram_25tps.log
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

rung() {  # N
  local N=$1
  local L="logs/mv_n${N}.log"
  say ""
  say "=== TP1 GPU_EXPERTS=$N  (mem_fraction 0.60, max_total_tokens 4096) ==="
  kill_server
  rm -f "$L"
  setsid env CUDA_VISIBLE_DEVICES=0 \
    GPU_EXPERTS="$N" WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
    AUTO_PROFILE=0 TP_SIZE=1 CPUINFER=72 NUMA_NODES="0 1" KT_THREADPOOL_COUNT=2 \
    MEM_FRACTION=0.60 MAX_TOTAL_TOKENS=4096 \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    bash experiments/adaptive_expert_cache/decode_cache/boot_adaptive_mtp.sh \
    > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED"; grep -E "Error|Traceback|out of memory|Killed" "$L" | tail -5 | tee -a "$M"; return; }

  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 14 500 llm >/dev/null 2>&1
  say "   measure:"
  .venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 3 500 llm \
    2>&1 | grep -E "^pass" | sed 's/^/      /' | tee -a "$M"
  say "   VRAM $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1)"
  say "   host RSS $(free -g | awk 'NR==2{print $3}')GB"
}

say "###### minimum single-GPU VRAM for 25 tok/s $(date -u) ######"
say "reference: rung E is the same protocol at N=24"
rung 16
rung 8
rung 4
say ""
say "Read the VRAM column as a CARD SPEC (TP1, so no second card shares the"
say "trunk). If tok/s is flat across N, VRAM is not the lever and the answer is"
say "the floor that boots, not the floor that hits 25."
say "###### done $(date -u) ######"
