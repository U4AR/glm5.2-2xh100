#!/usr/bin/env bash
# Emulate a 1x RTX PRO 6000 Blackwell / 140 GB RAM / 16 vCPU box on this host.
#
# Three constraints, emulated separately because they bind separately:
#
#   VRAM 96 GB   This card is an H100 NVL at 95830 MiB, i.e. within 1% of the
#                RTX PRO 6000's 96 GB. One card, TP1. Essentially exact.
#   RAM 140 GB   Neither shipped launcher fits: static placement at N=25 needs
#                (256-25) x 1.38 = 320 GiB and the adaptive cache stages all 256
#                (~390 GiB measured today). Only the tiered build fits, so this
#                runs boot_tiered.sh with the RAM tier sized to land under 140.
#   16 vCPU      taskset to cores 0-15 -- ONE NUMA node, since a 16-vCPU
#                instance would not straddle two -- with CPUINFER and the thread
#                pool count sized accordingly. This is the constraint that
#                matters: the decode bottleneck is the CPU expert path, and this
#                host normally runs it on 72 workers across 2 NUMA pools.
#
# WHAT THIS CANNOT EMULATE, stated up front: the GPU. An H100 NVL has HBM3 at
# ~3.35 TB/s; the RTX PRO 6000 has GDDR7 at roughly half that, and Blackwell
# would run the Marlin/Triton kernels rather than the CUTLASS/FlashMLA pair this
# card uses. So every number here is an UPPER BOUND for the real machine. The
# 2xL40 work suggests the GPU is not the binding term at this speed (its top0
# probe hit 42.77 tok/s once the CPU path was removed), which is why the
# emulation is worth running at all -- but "upper bound" is the honest label.
set -uo pipefail
cd /data/models/RunGLM
M=logs/emulate_rtx6000.log
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

RAM_EXPERTS="${RAM_EXPERTS:-64}"   # ~49 GB fixed + 1.19 GB/slot => ~125 GB
GPU_N="${GPU_N:-25}"               # what hardware_profile picks for a 96 GB card at TP1
L=logs/emu_rtx6000_server.log

say "###### emulate 1x RTX PRO 6000 / 140 GB / 16 vCPU $(date -u) ######"
say "GPU_EXPERTS=$GPU_N  KT_RAM_EXPERTS=$RAM_EXPERTS  cores 0-15  CPUINFER=12"
say "NOTE: H100 GPU stands in for Blackwell => these are UPPER BOUNDS."

kill_server
rm -f "$L"
# taskset wraps the launcher so every child -- scheduler, CPUInfer workers --
# inherits the 16-core mask.
setsid taskset -c 0-15 env CUDA_VISIBLE_DEVICES=0 \
  GPU_EXPERTS="$GPU_N" KT_RAM_EXPERTS="$RAM_EXPERTS" \
  KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_TIER_INCREMENTAL=1 \
  KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 \
  AUTO_PROFILE=0 TP_SIZE=1 CPUINFER=12 NUMA_NODES="0" KT_THREADPOOL_COUNT=1 \
  MEM_FRACTION=0.85 MAX_TOTAL_TOKENS=8192 \
  MAX_RUNNING=8 CUDA_GRAPH_MAX_BS=8 \
  KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
  TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  bash experiments/expert_tiering_ssd/boot_tiered.sh > "$L" 2>&1 < /dev/null &
disown
wait_ready "$L" || { say "BOOT FAILED"; grep -E "Error|Traceback|out of memory|Killed" "$L" | tail -6 | tee -a "$M"; exit 1; }

say ""
say "-- host RSS after boot: $(free -g | awk 'NR==2{print $3}')GB  (target: under 140)"
say "-- VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"

say ""
say "-- single stream, converged (14 warm passes then 3 measured)"
.venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 14 500 llm >/dev/null 2>&1
.venv/bin/python experiments/adaptive_expert_cache/decode_cache/conv_driver.py 3 500 llm \
  2>&1 | grep -E "^pass" | sed 's/^/   /' | tee -a "$M"

say ""
say "-- batch scaling (per-stream and aggregate)"
.venv/bin/python bench/intelligence_tier/batch_bench.py GLM5.2 256 1,2,4,8 \
  2>&1 | tail -20 | sed 's/^/   /' | tee -a "$M"

say ""
say "-- accuracy at this residency (GPU $GPU_N + RAM $RAM_EXPERTS = $((GPU_N+RAM_EXPERTS)) of 256 reachable)"
GPU_EXPERTS=$GPU_N KT_RAM_EXPERTS=$RAM_EXPERTS KT_TIER_FILL_POOL=gpu TIER_BENCH_MODEL=GLM5.2 \
  .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py compare \
  experiments/expert_tiering_ssd/runs/ref66.json rtx6000_emu 2>&1 \
  | grep -E "qa_n|qa_accuracy|qa_ci95|qa_loop_rate" | sed 's/^/   /' | tee -a "$M"

say ""
say "###### done $(date -u) ######"
