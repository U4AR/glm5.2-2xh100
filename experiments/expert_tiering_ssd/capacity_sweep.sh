#!/usr/bin/env bash
# Where is the crossover between "buy RAM" and "move experts"?
#
# Three measurements now agree that tier movement is a capacity story, not a
# latency or a signal-quality story:
#   - movement is saturated at 100% of budget on every visit, forever;
#   - 87-89% of promoted experts are genuinely called before the next decision,
#     and ~1% are demoted again, so the promotions are not noise;
#   - quadrupling the tick rate (84s -> 21s promotion latency) changed accuracy
#     by nothing at all.
# There is always another warm expert outside the tier because the working set
# is bigger than the tier. That predicts a crossover: below some RAM size
# movement is repairing real damage, above it movement is pure overhead.
#
# Known endpoints: at RAM=72 frozen already holds reference accuracy and
# movement only costs speed; at RAM=32 frozen is broken (0.5625, 44% loops) and
# movement repairs it to 0.75. So the crossover is between 32 and 72, and
# nobody has looked. Each rung is measured BOTH ways against the same 66-item
# set, which is the whole point -- a config is only interesting if it holds
# quality, and quality claims at 16 items were single-item noise.
set -uo pipefail
cd /data/models/RunGLM
M=logs/capacity_sweep.log
REF=experiments/expert_tiering_ssd/runs/ref66.json
say() { echo "$@" | tee -a "$M"; }

kill_server() {
  for p in $(pgrep -f "[s]glang.launch_server"); do kill $p 2>/dev/null; done
  for i in $(seq 1 40); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 3; done
  for p in $(pgrep -f "[s]glang"); do kill -9 $p 2>/dev/null; done
  sleep 8
}

warm() {
  for t in "Explain how mixture-of-experts routing works." \
           "Explain the immune response to a viral infection." \
           "Explain how a B-tree index speeds up database lookups." \
           "Explain the causes of inflation in an economy." \
           "Explain how plate tectonics shapes mountain ranges." \
           "Explain counterpoint in Baroque music."; do
    curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"GLM5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"$t\"}],\"temperature\":0.0,\"max_tokens\":400}" >/dev/null 2>&1
  done
}

boot() {  # TAG env...
  local TAG=$1; shift
  local L="logs/cap_${TAG}.log"
  kill_server
  rm -f "$L"
  setsid env \
    GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=4096 \
    KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
    KT_ADAPTIVE_COUNTS_DUMP_PT= \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    "$@" \
    bash experiments/expert_tiering_ssd/boot_tiered.sh \
    > "$L" 2>&1 < /dev/null &
  disown
  sleep 45
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && return 0
    grep -qE "Traceback|CUDA out of memory|Killed" "$L" && break
    pgrep -f "[s]glang.launch_server" >/dev/null || break
    sleep 10
  done
  say "   BOOT FAILED ($TAG)"; grep -E "Error|Traceback" "$L" | tail -5 | tee -a "$M"
  return 1
}

row() {  # TAG RAM MODE
  local TAG=$1 RAM=$2 MODE=$3
  say ""
  say "=== $TAG :: RAM=$RAM $MODE ==="
  local ENVS
  if [ "$MODE" = "frozen" ]; then
    ENVS="KT_TIER_DYNAMIC=0 KT_ADAPTIVE_DECODE=0"
  else
    ENVS="KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_TIER_MAX_PROMOTE=0 KT_TIER_MAX_RAM_MOVE=4"
  fi
  boot "$TAG" KT_RAM_EXPERTS=$RAM $ENVS || return
  warm
  say "   c1 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  say "   c2 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  say "   stall $(.venv/bin/python experiments/expert_tiering_ssd/token_latency.py 400 2 2>&1 | grep -E 'gap p50|hitches|STALL_FRACTION' | tr '\n' ' ')"
  say "   RSS $(free -g | awk 'NR==2{print $3}')GB"
  GPU_EXPERTS=104 KT_RAM_EXPERTS=$RAM KT_TIER_FILL_POOL=gpu TIER_BENCH_MODEL=GLM5.2-top2 \
    .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py compare "$REF" "$TAG" 2>&1 \
    | grep -E "qa_n|qa_accuracy|qa_ci95|qa_loop_rate|gen_identical|gen_prefix_agreement\"" | tee -a "$M"
}

# Chain behind the decomposition ladder.
for i in $(seq 1 360); do
  grep -q "decompose complete" logs/decompose.log 2>/dev/null && break
  sleep 30
done

say "###### capacity crossover sweep $(date -u) ######"

# Reference for the 66-item set: SSD tier EMPTY (RAM = 256-104 = 152), so every
# expert is reachable and no substitution happens. Every row below is scored
# against this, which is what makes "degradation" mean something.
if [ ! -f "$REF" ]; then
  say ""
  say "=== building 66-item reference: RAM=152, SSD=0 ==="
  if boot ref66 KT_RAM_EXPERTS=152 KT_TIER_DYNAMIC=0 KT_ADAPTIVE_DECODE=0; then
    warm
    say "   c1 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
    TIER_BENCH_MODEL=GLM5.2-top2 \
      .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py reference "$REF" 2>&1 \
      | tail -3 | tee -a "$M"
  fi
fi

for RAM in 16 32 48 72; do
  row "r${RAM}_frozen" $RAM frozen
  row "r${RAM}_move"   $RAM move
done

say ""
say "###### capacity sweep complete $(date -u) ######"
