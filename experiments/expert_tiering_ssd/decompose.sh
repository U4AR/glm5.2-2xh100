#!/usr/bin/env bash
# Where does the 40.62 -> 29.67 actually GO?
#
# So far the cost of tier movement has only ever been measured in aggregate:
# turn movement on, watch throughput fall. That bundles three separate things
# together -- counting demand every step, deciding what to move every tick, and
# moving it -- and the fix for each is completely different. The ~14% that went
# missing (4.5% blocked should give ~38.8, measured 33.45) says at least one of
# them is not where the accounting thinks it is.
#
# Each rung adds exactly ONE mechanism to the one below it, so a drop between
# adjacent rungs is that mechanism's price and nothing else:
#
#   A frozen   nothing at all                                   (control, 40.62)
#   B count    + per-step in-graph demand counters (no ticks)
#   C decide   + the full selection every 32 steps, ZERO moves allowed
#   D ram      + RAM<->SSD movement                             (= "nogpu", 33.45)
#   D0 ram-npf   D with the prefetch lookahead OFF              (its A/B)
#   E gpu-i    + RAM<->GPU movement via the stable-slot swap    (cf49da9)
#   F gpu-f    + RAM<->GPU movement via the full restage        (= old, 29.67)
#
# B and C are the rungs that have never existed. If the loss is concentrated
# there, then movement was never the problem and every "make moves cheaper"
# result so far has been optimising the wrong half.
#
# All rows at RAM=72/SSD=80/GPU=104 -- the rung that holds reference accuracy,
# so nothing here is bought with substitution.
set -uo pipefail
cd /data/models/RunGLM
M=logs/decompose.log
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

row() {  # PASS TAG extra-env...
  local PASS=$1; local TAG=$2; shift 2
  local L="logs/dc_${PASS}_${TAG}.log"
  say ""
  say "=== [$PASS] $TAG :: $* ==="
  kill_server
  rm -f "$L"
  setsid env \
    KT_RAM_EXPERTS=72 GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=4096 \
    KT_TOPK_MODE=safe2 WARM_START=1 KT_ADAPTIVE_PRIOR_MASS=64 \
    KT_ADAPTIVE_COUNTS_DUMP_PT= KT_TIER_PROFILE=1 \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    "$@" \
    bash experiments/expert_tiering_ssd/boot_tiered.sh \
    > "$L" 2>&1 < /dev/null &
  disown
  sleep 45
  local ok=0
  for i in $(seq 1 200); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && { ok=1; break; }
    grep -qE "Traceback|CUDA out of memory|Killed" "$L" && break
    pgrep -f "[s]glang.launch_server" >/dev/null || break
    sleep 10
  done
  if [ "$ok" != 1 ]; then
    say "   BOOT FAILED"; grep -E "Error|Traceback" "$L" | tail -5 | tee -a "$M"; return
  fi

  # Coherence probe BEFORE the timings. The stable-slot swap (rung E) writes
  # W4AFP8 expert weights into live GPU slots and has never run against real
  # weights -- and a broken placement stays fast while emitting garbage, so a
  # tok/s number alone would not notice. One trivial question is enough to
  # catch the catastrophic case; the full 66-item eval comes later.
  say "   probe $(curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"GLM5.2","messages":[{"role":"user","content":"What is the chemical symbol for gold? Reply with only the symbol."}],"temperature":0.0,"max_tokens":600}' \
      | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:60]) or "EMPTY")' 2>&1)"

  warm   # warm FIRST: a fresh number measures the warm-up, not the config
  say "   c1 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  say "   c2 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"
  say "   c3 $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)"

  .venv/bin/python - "$L" <<'PY' | tee -a "$M"
import re, sys, datetime, collections
path = sys.argv[1]
took, gsw, nochange = [], [], 0
ph = collections.defaultdict(list)
ts = []
for line in open(path, errors="ignore"):
    m = re.search(r'\[kt-tier\] layer=(\d+) gpu_swaps=(\d+).*?took=(\d+)ms', line)
    if m:
        gsw.append(int(m.group(2))); took.append(int(m.group(3)))
        t = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
        if t: ts.append(datetime.datetime.strptime(t.group(1), '%Y-%m-%d %H:%M:%S'))
        continue
    p = re.search(r'\[kt-tier-prof\] layer=\d+ (NOCHANGE )?total=([\d.]+)ms (.*)', line)
    if p:
        if p.group(1): nochange += 1
        for seg in p.group(3).split():
            k, _, v = seg.partition('=')
            ph[k].append(float(v))
if not took and not nochange:
    print("   NO TIER ACTIVITY"); raise SystemExit
if took:
    span = (ts[-1] - ts[0]).total_seconds() if len(ts) > 1 else 0.0
    print(f"   moves {len(took)}  mean took {sum(took)/len(took):.1f}ms  "
          f"gpu_swaps/visit {sum(gsw)/len(gsw):.2f}  "
          f"blocked {100*sum(took)/1000.0/span if span else 0:.1f}% of {span:.0f}s")
print(f"   no-change visits {nochange}")
if ph:
    # Per-phase mean over every visit that reached that phase. The phases are
    # cumulative segments of one visit, so they sum to the total.
    order = ["d2h","select","bcast","diff","gpu_incr","gpu_full","evict","roi",
             "promote","sync","log"]
    print("   phase means (ms): " + "  ".join(
        f"{k}={sum(ph[k])/len(ph[k]):.1f}x{len(ph[k])}" for k in order if k in ph))
PY

  say "   RSS $(free -g | awk 'NR==2{print $3}')GB  tier-fail $(grep -c 'kt-tier.*selection failed' "$L" 2>/dev/null)"
}

# Wait for the within-boot noise run to release the server.
for i in $(seq 1 120); do
  grep -q "within-boot noise complete" logs/noise_within_boot.log 2>/dev/null && break
  sleep 20
done

say "###### cost decomposition ladder $(date -u) ######"
say "RAM=72 SSD=80 GPU=104 safe2+MTP-d3. controls: frozen 40.62, coupled 29.67, nogpu 33.45"

FROZEN="KT_TIER_DYNAMIC=0 KT_ADAPTIVE_DECODE=0"
COUNT="KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=0"
DECIDE="KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=0 KT_TIER_MAX_RAM_MOVE=0"
RAM="KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=0 KT_TIER_MAX_RAM_MOVE=4"
GPUI="KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 KT_TIER_INCREMENTAL=1"
GPUF="KT_TIER_DYNAMIC=1 KT_ADAPTIVE_DECODE=1 KT_ADAPTIVE_PERIOD=32 KT_TIER_MAX_PROMOTE=2 KT_TIER_MAX_RAM_MOVE=4 KT_TIER_INCREMENTAL=0"

row p1 A_frozen  $FROZEN
row p1 B_count   $COUNT
row p1 C_decide  $DECIDE
row p1 D_ram     $RAM
row p1 D0_nopf   $RAM KT_TIER_PREFETCH_LAYERS=0
row p1 E_gpu_inc $GPUI
row p1 F_gpu_full $GPUF

say ""
say "###### pass 2, reverse order (boot-to-boot control) $(date -u) ######"
row p2 F_gpu_full $GPUF
row p2 E_gpu_inc  $GPUI
row p2 D_ram      $RAM
row p2 A_frozen   $FROZEN

say ""
say "###### decompose complete $(date -u) ######"
