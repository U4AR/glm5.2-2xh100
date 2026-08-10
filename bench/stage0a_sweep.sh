#!/usr/bin/env bash
# Stage 0A: the CPU expert path's cost curve, measured rather than inferred.
#
# Two boots of the SAME configuration, sweeping the per-request tier K over
# {0,1,2,4,6,8} in each:
#
#   normal   CPU expert path active
#   noCPU    /tmp/kt_skip_cpu armed -> submit/sync never fires
#
# step_ms(K) - step_ms_noCPU(K) is the CPU path's contribution at that K, with
# every other cost (attention, dense trunk, MTP draft+verify, graph replay)
# differenced away. The K=0 point isolates the FIXED component directly: the
# submit/sync still fires in the normal boot but every expert is masked off, so
# whatever remains is what a layer pays merely for having a CPU path at all.
#
# This is the number the whole streaming design turns on. If the curve is
# a + b*K with a large a, an interior split buys little and only reaching
# ZERO CPU experts in a layer pays (the corner). If a ~ 0 it is proportional and
# the balance formula is the whole game. Section 1.2 of the plan already
# overturned the original "fixed submit cost" reading once; this measures it
# end to end instead of inferring it from the oracle ceiling.
#
# Config matches the 0A baseline row exactly (GPU_EXPERTS=60, safe8, MTP
# depth-3, packed RAWINT4, TP=2) so the result is comparable to
# bench/profile_out/rates/cpu-stage0a-legacy-e60-safe8.json (15.49 tok/s).
#
# THE SENTINEL TRICK, because it looks like a mistake otherwise: kt_ep_wrapper
# reads a HARDCODED "/tmp/kt_skip_cpu" once at import, while run_fast.sh deletes
# $KT_SKIP_CPU_FILE whenever MODE=safe. Pointing KT_SKIP_CPU_FILE at a decoy
# makes the launcher delete the decoy and leaves the real sentinel standing, so
# the no-CPU boot is deterministic instead of racing the launcher's rm.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
M=logs/stage0a_sweep.log
say() { echo "$@" | tee -a "$M"; }

GPU_N="${GPU_N:-60}"
TIERS="${TIERS:-0,1,2,4,6,8}"
RUNS="${RUNS:-3}"
TOKENS="${TOKENS:-200}"

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

sweep() {  # NAME  skip_cpu(0|1)
  local NAME=$1 SKIP=$2
  local L="logs/stage0a_${NAME}.log"
  say ""
  say "=== $NAME : GPU_EXPERTS=$GPU_N safe8 TP2 MTP3  skip_cpu=$SKIP ==="
  kill_server
  rm -f "$L" /tmp/kt_skip_cpu /tmp/kt_skip_cpu_decoy
  if [ "$SKIP" = "1" ]; then
    touch /tmp/kt_skip_cpu
    say "   sentinel /tmp/kt_skip_cpu ARMED (launcher will delete the decoy instead)"
  fi
  setsid env MODE=safe KEEP=8 GPU_EXPERTS="$GPU_N" MTP=1 \
    KT_CPU_EXPERT_OPTS=none \
    KT_SKIP_CPU_FILE=/tmp/kt_skip_cpu_decoy \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    bash run_fast.sh > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED"; grep -E "Error|Traceback|out of memory|Killed" "$L" | tail -6 | tee -a "$M"; return 1; }

  # Confirm the boot actually took the intended path rather than assuming it.
  if grep -qa "CPU expert submit/sync SKIPPED" "$L"; then
    say "   confirmed: CPU path SKIPPED this boot"
    [ "$SKIP" = "1" ] || { say "   *** WRONG: skip active on the normal boot"; return 1; }
  else
    say "   confirmed: CPU path ACTIVE this boot"
    [ "$SKIP" = "0" ] || { say "   *** WRONG: skip did not take on the noCPU boot"; return 1; }
  fi

  .venv/bin/python bench/cpu_fixed_cost.py --tiers "$TIERS" --runs "$RUNS" \
    --tokens "$TOKENS" --out "bench/profile_out/stage0a_${NAME}.json" \
    2>&1 | tee -a "$M"
}

say "###### Stage 0A cost curve $(date -u) ######"
say "reference: cpu-stage0a-legacy-e60-safe8 = 15.49 tok/s median"
sweep "normal" 0
sweep "noCPU"  1

say ""
say "-- differencing the two sweeps --"
.venv/bin/python - <<'PY' 2>&1 | tee -a "$M"
import json, pathlib
def load(p):
    f = pathlib.Path(p)
    if not f.is_file():
        return None
    return json.loads(f.read_text())["summary"]
a = load("bench/profile_out/stage0a_normal.json")
b = load("bench/profile_out/stage0a_noCPU.json")
if not a or not b:
    print("missing a sweep; nothing to difference")
    raise SystemExit(0)
print(f"{'K':>3} {'normal':>9} {'noCPU':>9} {'CPU part':>9} {'CPU %':>7}")
rows = []
for k in sorted({int(x) for x in a if x.isdigit()} & {int(x) for x in b if x.isdigit()}):
    n, m = a[str(k)]["ms_per_step"], b[str(k)]["ms_per_step"]
    d = n - m
    print(f"{k:>3} {n:9.2f} {m:9.2f} {d:9.2f} {d/n*100:6.1f}%")
    rows.append((k, d))
if len(rows) >= 2:
    fixed = dict(rows).get(0)
    top = dict(rows).get(8)
    if fixed is not None and top is not None and top:
        print(f"\nfixed component (K=0 difference): {fixed:.2f} ms/step")
        print(f"CPU path at K=8:                  {top:.2f} ms/step")
        print(f"fixed share of the CPU path:      {fixed/top*100:.1f}%")
        print(f"marginal per kept expert:         {(top-fixed)/8:.3f} ms/step")
        print("\nA large fixed share means only reaching ZERO CPU experts in a")
        print("layer pays; a small one means an interior split is worth its share.")
PY
say ""
say "###### done $(date -u) ######"
