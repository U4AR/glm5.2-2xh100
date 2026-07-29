#!/usr/bin/env bash
# Is the accept-length loss caused by the RESIDENCY movement produces, or by
# the ACT of moving?
#
# The ladder established that ~85% of what tier movement costs is degraded MTP
# accept length (2.693 frozen -> 2.318 moving), not time spent blocked. Two
# mechanisms fit that equally well and they have opposite fixes:
#
#   RESIDENCY : the drifted set is a worse model than the warm-start hot-core
#               ranking it replaced, so the fixed draft head agrees less often.
#               Fix = don't drift, or drift toward a better objective.
#   TRANSIENT : each swap disrupts generation and it recovers in between.
#               Fix = swap more gently or less often.
#
# Comparing two BOOTS cannot separate them: a frozen boot also has different
# (warm-start) residency, so it changes both variables at once. This runs ONE
# boot in three phases, changing exactly one thing:
#
#   A  moving      -- residency drifting, swaps happening
#   B  frozen      -- residency HELD WHERE IT DRIFTED TO, swaps stopped
#   A' moving      -- swaps resumed, to show B was not just drift or warm-up
#
# Predictions, stated before running:
#   RESIDENCY => B keeps A's low accept (~2.32). tok/s rises a little because
#                the move work stops, but stays well under frozen-boot 40.5.
#   TRANSIENT => B's accept climbs back toward 2.69 and tok/s approaches 40.5.
# A' must return to A for either reading to be trusted.
set -uo pipefail
cd /data/models/RunGLM
M=logs/freeze_probe.log
FRZ=/data/tmp/claude-1002/-data-models-RunGLM/29a056dd-68d9-420b-94b1-6622afbfac3c/scratchpad/FREEZE
L=logs/freeze_probe_server.log
say() { echo "$@" | tee -a "$M"; }

rm -f "$FRZ"
for p in $(pgrep -f "[s]glang.launch_server"); do kill $p 2>/dev/null; done
for i in $(seq 1 40); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 3; done
for p in $(pgrep -f "[s]glang"); do kill -9 $p 2>/dev/null; done
sleep 8

say "###### freeze probe $(date -u) ######"
say "residency-vs-swapping, one boot, three phases. RAM=72 SSD=80 GPU=104."

rm -f "$L"
setsid env \
  KT_RAM_EXPERTS=72 KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 \
  KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 \
  KT_TIER_INCREMENTAL=1 KT_TIER_FREEZE_FILE="$FRZ" \
  GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=4096 \
  KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
  KT_ADAPTIVE_COUNTS_DUMP_PT= \
  TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  bash experiments/expert_tiering_ssd/boot_tiered.sh > "$L" 2>&1 < /dev/null &
disown
sleep 45
ok=0
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && { ok=1; break; }
  grep -qE "Traceback|CUDA out of memory|Killed" "$L" && break
  pgrep -f "[s]glang.launch_server" >/dev/null || break
  sleep 10
done
[ "$ok" = 1 ] || { say "BOOT FAILED"; grep -E "Error|Traceback" "$L" | tail -5 | tee -a "$M"; exit 1; }

warm() {
  for t in "Explain how mixture-of-experts routing works." \
           "Explain the immune response to a viral infection." \
           "Explain how a B-tree index speeds up database lookups." \
           "Explain the causes of inflation in an economy."; do
    curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"GLM5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"$t\"}],\"temperature\":0.0,\"max_tokens\":400}" >/dev/null 2>&1
  done
}

# Accept length and visit count for the phase that just ran, taken from the
# tail of the server log rather than the whole file, so each phase is scored on
# its own steps.
phase() {  # NAME  start_line
  local NAME=$1 START=$2
  .venv/bin/python - "$L" "$START" "$NAME" <<'PY' | tee -a "$M"
import re, statistics, sys
path, start, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
acc, visits = [], 0
for i, line in enumerate(open(path, errors="ignore")):
    if i < start:
        continue
    m = re.search(r'accept len: ([\d.]+)', line)
    if m:
        acc.append(float(m.group(1)))
    if '[kt-tier] layer=' in line:
        visits += 1
if acc:
    print(f"   {name:14s} accept {statistics.mean(acc):.3f} "
          f"(n={len(acc)})  tier_visits={visits}")
else:
    print(f"   {name:14s} NO DECODE SAMPLES  tier_visits={visits}")
PY
}

# --- phase A: moving -------------------------------------------------------
warm
A0=$(wc -l < "$L")
say "   A  moving   $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
phase "A moving" "$A0"

# --- phase B: freeze movement, residency stays where it drifted ------------
touch "$FRZ"
sleep 20   # let any in-flight tick finish and the freeze take effect
B0=$(wc -l < "$L")
say "   B  frozen   $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
phase "B frozen" "$B0"

# --- phase A': resume, to prove B was the freeze and not drift/warm-up -----
rm -f "$FRZ"
sleep 20
C0=$(wc -l < "$L")
say "   A' moving  $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
phase "A' moving" "$C0"

say ""
say "B tier_visits must be ~0 -- if it is not, the freeze did not take and the"
say "phase means say nothing. A' must return to A."
say "###### freeze probe complete $(date -u) ######"
