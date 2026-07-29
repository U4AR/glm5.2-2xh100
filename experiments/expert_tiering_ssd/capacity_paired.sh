#!/usr/bin/env bash
# Where is the crossover between "buy RAM" and "move experts"?
#
# Both endpoints are already measured, so only the middle is worth spending time
# on: at RAM=72 frozen holds reference accuracy and movement only costs speed;
# at RAM=32 frozen is broken (0.5625 acc, 44% loops) and movement repairs it to
# 0.75. The crossover is between them. This measures RAM=32 and 48.
#
# PAIRED, and that is the point. Boot-to-boot variance is the dominant noise
# term here (repeat boots of one config span 31.16-38.67 tok/s) while
# within-boot sd is only 0.57. Measuring frozen and move as separate BOOTS puts
# the larger noise term inside the very comparison being made. So both arms run
# inside ONE boot, using the KT_TIER_FREEZE_FILE switch.
#
# Order is forced and deliberate: the boot starts FROZEN, because the frozen
# arm must be measured at WARM-START residency (the globally-ranked hot core),
# which only exists before any drift. Releasing the freeze then lets residency
# drift and the move arm is measured after it has. The reverse order is
# impossible -- residency cannot be un-drifted -- and freezing mid-run would
# measure "drifted but not moving", which is the freeze PROBE's question, not
# this one.
#
# Time in boot is not a confound: 9 median-of-12 blocks on one unchanged server
# were flat (33.87 -> 33.79) across 20 minutes of continuous movement.
set -uo pipefail
cd /data/models/RunGLM
M=logs/capacity_paired.log
REF=experiments/expert_tiering_ssd/runs/ref66.json
FRZ=/data/tmp/claude-1002/-data-models-RunGLM/29a056dd-68d9-420b-94b1-6622afbfac3c/scratchpad/CAPFREEZE
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

wait_ready() {  # LOGFILE
  local L=$1
  sleep 30
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && return 0
    grep -qE "Traceback|CUDA out of memory|Killed" "$L" && break
    pgrep -f "[s]glang.launch_server" >/dev/null || break
    sleep 10
  done
  return 1
}

# Visits and accept length for the arm that just ran, selected by the log line's
# own TIMESTAMP (epoch seconds) rather than by line offset.
#
# Line offsets race the server's buffered writes. The freeze probe hit this: it
# sampled `wc -l` right after freezing, ~16 pre-freeze visits had not flushed
# yet, and they landed past that offset and were charged to the frozen phase --
# which then looked 87% frozen and tripped the "must be 0" guard even though a
# 147 s window with zero visits proved the freeze was total. Timestamps are
# stamped at event time and do not move when the buffer does.
phase() {  # LOG START_EPOCH NAME
  .venv/bin/python - "$1" "$2" "$3" <<'PY' | tee -a "$M"
import datetime, re, statistics, sys
path, start, name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
t_start = datetime.datetime.fromtimestamp(start)
acc, visits = [], 0
for line in open(path, errors="ignore"):
    m = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
    if not m:
        continue
    if datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") < t_start:
        continue
    a = re.search(r'accept len: ([\d.]+)', line)
    if a:
        acc.append(float(a.group(1)))
    if '[kt-tier] layer=' in line:
        visits += 1
a = f"{statistics.mean(acc):.3f}" if acc else "n/a"
print(f"   {name:12s} accept {a} (n={len(acc)})  tier_visits={visits}")
PY
}

say "###### capacity crossover, PAIRED within boot $(date -u) ######"

# ---- reference: RAM=152 => SSD tier empty => every expert reachable ---------
# Reuses the server already booted by the previous attempt if it is alive.
if [ ! -f "$REF" ]; then
  say ""
  say "=== 66-item reference (RAM=152, SSD=0) ==="
  if ! curl -s -m 5 http://127.0.0.1:8000/health_generate >/dev/null 2>&1; then
    kill_server
    setsid env KT_RAM_EXPERTS=152 KT_TIER_DYNAMIC=0 KT_ADAPTIVE_DECODE=0 \
      GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=4096 \
      KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
      KT_ADAPTIVE_COUNTS_DUMP_PT= TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
      bash experiments/expert_tiering_ssd/boot_tiered.sh \
      > logs/cap_ref66.log 2>&1 < /dev/null &
    disown
    wait_ready logs/cap_ref66.log || { say "   REF BOOT FAILED"; exit 1; }
  else
    say "   (reusing the already-booted RAM=152 server)"
  fi
  warm
  say "   ref c1 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  TIER_BENCH_MODEL=GLM5.2-top2 \
    .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py reference "$REF" 2>&1 \
    | tail -2 | tee -a "$M"
fi

# ---- paired rows -----------------------------------------------------------
row() {  # RAM
  local RAM=$1
  local L="logs/capp_r${RAM}.log"
  say ""
  say "=== RAM=$RAM (frozen and move, same boot) ==="
  kill_server
  rm -f "$L"
  touch "$FRZ"          # start FROZEN: warm-start residency, no movement
  setsid env \
    KT_RAM_EXPERTS=$RAM KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 \
    KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 \
    KT_TIER_INCREMENTAL=1 KT_TIER_FREEZE_FILE="$FRZ" \
    GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=4096 \
    KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
    KT_ADAPTIVE_COUNTS_DUMP_PT= TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    bash experiments/expert_tiering_ssd/boot_tiered.sh > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED (RAM=$RAM)"; grep -E "Error|Traceback" "$L" | tail -4 | tee -a "$M"; return; }

  # -- arm 1: FROZEN at warm-start residency
  warm
  local P0=$(date +%s)
  say "   frozen  $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  phase "$L" "$P0" "frozen"
  GPU_EXPERTS=104 KT_RAM_EXPERTS=$RAM KT_TIER_FILL_POOL=gpu TIER_BENCH_MODEL=GLM5.2-top2 \
    .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py compare "$REF" "r${RAM}_frozen" 2>&1 \
    | grep -E "qa_n|qa_accuracy|qa_ci95|qa_loop_rate|gen_prefix_agreement\"" | tee -a "$M"

  # -- arm 2: release the freeze, let residency drift, then measure
  rm -f "$FRZ"
  warm; warm                       # drift under real traffic before measuring
  local P1=$(date +%s)
  say "   move    $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  phase "$L" "$P1" "move"
  GPU_EXPERTS=104 KT_RAM_EXPERTS=$RAM KT_TIER_FILL_POOL=gpu TIER_BENCH_MODEL=GLM5.2-top2 \
    .venv/bin/python experiments/expert_tiering_ssd/accuracy_eval.py compare "$REF" "r${RAM}_move" 2>&1 \
    | grep -E "qa_n|qa_accuracy|qa_ci95|qa_loop_rate|gen_prefix_agreement\"" | tee -a "$M"

  say "   RSS $(free -g | awk 'NR==2{print $3}')GB"
}

row 32
row 48
rm -f "$FRZ"
say ""
say "frozen arm must show tier_visits=0. If it does not, the freeze did not take"
say "and that arm is not frozen -- discard the row rather than reading it."
say "###### capacity paired complete $(date -u) ######"
