#!/usr/bin/env bash
# Rung J: the missing cell. GPU=25, RAM=64, SSD=167 -- but at FULL CPU.
#
# The RTX PRO 6000 emulation changed two things at once against rung E, so its
# 29.9 -> 17.5 drop cannot be attributed:
#
#   E    TP1  GPU=24  RAM=all   SSD=0    72 workers   29.9 tok/s
#   emu  TP1  GPU=25  RAM=64    SSD=167  12 workers   17.5 tok/s
#   J    TP1  GPU=25  RAM=64    SSD=167  72 workers   <- THIS
#
# J vs E isolates the SSD tier (only the store shape differs).
# emu vs J isolates the 16 vCPU (only the worker count differs).
#
# Movement stays ON and configured exactly as the emulation had it, so J is the
# emulation with its CPU restored and nothing else touched.
#
# Worth stating the prediction before measuring, because it is not obviously
# "slower": an SSD tier REMOVES CPU work. A substituted expert costs zero CPU
# compute, so at 89/256 reachable a large share of routed experts never reach
# the CPU at all -- that is the same mechanism that makes sub2/top0 fast. So J
# could plausibly beat E on step rate while losing on accept length. If J lands
# near E, the SSD tier is roughly free at full CPU and the emulation's deficit
# is essentially all vCPU count. If J lands well below E, the SSD tier carries
# a real cost that the 12-worker run was masking.
set -uo pipefail
cd /data/models/RunGLM
M=logs/ssd_at_full_cpu.log
say() { echo "$@" | tee -a "$M"; }

kill_server() {
  for p in $(pgrep -f "[s]glang.launch_server"); do kill $p 2>/dev/null; done
  for i in $(seq 1 40); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "[s]glang"); do kill -9 $p 2>/dev/null; done
  # Process exit != VRAM released. Wait for the card to actually drain, or the
  # next boot lands on top of the dying server and OOMs (that is exactly what
  # killed rungs H and I: two servers, 47.9 GiB still held by the old one).
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-9999}" -lt 2000 ] && break
    sleep 5
  done
  sleep 5
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

L=logs/j_ssd_fullcpu.log
say "###### J: GPU=25 RAM=64 SSD=167 at FULL CPU (72 workers) $(date -u) ######"
say "compare: E(SSD=0, 72w)=29.9   emu(SSD=167, 12w)=17.5"
kill_server
rm -f "$L"
# No taskset: all cores, both NUMA nodes -- the ONLY difference from the emulation.
setsid env CUDA_VISIBLE_DEVICES=0 \
  GPU_EXPERTS=25 KT_RAM_EXPERTS=64 \
  KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_TIER_INCREMENTAL=1 \
  KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 \
  AUTO_PROFILE=0 TP_SIZE=1 CPUINFER=72 NUMA_NODES="0 1" KT_THREADPOOL_COUNT=2 \
  MEM_FRACTION=0.85 MAX_TOTAL_TOKENS=8192 \
  MAX_RUNNING=8 CUDA_GRAPH_MAX_BS=8 \
  KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
  TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  bash experiments/expert_tiering_ssd/boot_tiered.sh > "$L" 2>&1 < /dev/null &
disown
wait_ready "$L" || { say "BOOT FAILED"; grep -E "Error|Traceback|out of memory|Killed" "$L" | tail -6 | tee -a "$M"; exit 1; }

say "   host RSS $(free -g | awk 'NR==2{print $3}')GB   VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
.venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 14 500 llm >/dev/null 2>&1
say "   measure (5 passes -- report the TREND, the emulation's declined):"
.venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 5 500 llm \
  2>&1 | grep -E "^pass" | sed 's/^/      /' | tee -a "$M"
say "   tier movement events: $(grep -ac '\[kt-tier\]' "$L" 2>/dev/null || echo 0)"
say "###### done $(date -u) ######"
