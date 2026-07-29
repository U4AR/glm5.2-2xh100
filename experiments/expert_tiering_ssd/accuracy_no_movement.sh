#!/usr/bin/env bash
# Is the movement-off speed real output, or is it the model looping?
#
# Rung H (GPU=25 RAM=64, movement OFF, 16 vCPU) measured 26.7 tok/s -- +53% over
# the same box with movement on. But its accept length is 3.52 against a HARD
# CEILING OF 4 (--speculative-num-draft-tokens 4), i.e. 88% of drafted tokens
# accepted. Coherent rung G accepts 2.446, or 61%. A draft head only reaches 88%
# when the target's continuation is trivially predictable, and the 66-item eval
# at this exact residency already measured a 56% loop rate.
#
# So 26.7 tok/s is not yet a speed result. tok/s = accept x steps/s, and if the
# accept term is inflated by degeneration then the headline number is partly a
# measure of how badly the model is broken. This settles it:
#
#   H_acc  GPU=25 RAM=64, movement OFF   89/256 reachable
#   I_acc  GPU=25 RAM=84, movement OFF  109/256 reachable
#
# Prediction, recorded before running: both land near the 0.35 accuracy / 0.56
# loop rate already measured with movement ON at 89/256, because substitution --
# not churn -- is what removes the genuine experts. If instead H_acc comes back
# near 1.0, then substitution at this residency is benign and the reachability
# curve needs revisiting.
#
# Movement OFF is the interesting case precisely because the mover is the only
# mechanism that ever repairs a substitution: without it, an expert on SSD stays
# substituted forever. If accuracy is the same either way, the mover is buying
# nothing at all and should simply be off on RAM-poor boxes.
set -uo pipefail
cd /data/models/RunGLM
M=logs/accuracy_no_movement.log
say() { echo "$@" | tee -a "$M"; }

kill_server() {
  for p in $(pgrep -f "[s]glang.launch_server"); do kill $p 2>/dev/null; done
  for i in $(seq 1 40); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "[s]glang"); do kill -9 $p 2>/dev/null; done
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

rung() {  # NAME GPU_N RAM_N
  local NAME=$1 GPU_N=$2 RAM_N=$3
  local L="logs/acc_${NAME}.log"
  say ""
  say "=== $NAME : GPU=$GPU_N RAM=$RAM_N ($((GPU_N+RAM_N))/256 reachable), movement OFF ==="
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

  GPU_EXPERTS=$GPU_N KT_RAM_EXPERTS=$RAM_N TIER_BENCH_MODEL=GLM5.2 \
    .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py compare \
    experiments/expert_tiering_ssd/runs/ref66.json "$NAME" 2>&1 \
    | grep -E "qa_n|qa_accuracy|qa_ci95|qa_loop_rate" | sed 's/^/   /' | tee -a "$M"
}

say "###### accuracy with movement OFF $(date -u) ######"
say "reference: movement ON at 89/256 measured acc 0.3485, loops 0.5606"
say "prediction: both rungs land near that; substitution, not churn, is the cause"
rung "H_ram64_noMove" 25 64
rung "I_ram84_noMove" 25 84
say ""
say "###### done $(date -u) ######"
