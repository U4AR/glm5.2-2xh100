#!/usr/bin/env bash
# Is the single-GPU gap the missing CARD, or the smaller resident expert set?
#
# The 1x H100 run measured 23.74 tok/s at GPU_EXPERTS=24, against ~30.5 for the
# shipped 2-GPU config at GPU_EXPERTS=104. Two things changed at once there, so
# that pair cannot say which one cost the tokens.
#
# The low-VRAM ladder in experiments/adaptive_expert_cache/decode_cache is NOT
# the missing control: every one of its rows ran on 2xH100 at TP2 and shrank
# only the VRAM FOOTPRINT (N=16 -> 30.3, N=32 -> 32.3 tok/s). It never removed a
# card. Read as single-GPU numbers it would say the second GPU is nearly free,
# which is exactly the claim being tested here.
#
# So hold the harness fixed and change ONE thing per rung:
#   A  TP1  N=24   (measured already: 23.74)
#   B  TP2  N=24   A + the second card, everything else identical -> TP cost
#   C  TP2  N=104  B + the full resident set at its shipped settings -> residency
#
# B is the rung that matters. It shares mem_fraction, KV pool, routing contract,
# placement and benchmark with A, so B-A is the second GPU and nothing else.
# C changes mem_fraction and the KV pool too, because 104 experts/layer do not
# fit at 0.85 -- it is the shipped operating point, not a clean one-variable step.
set -uo pipefail
cd /data/models/RunGLM
M=logs/tp_decompose.log
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
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && return 0
    grep -qE "Traceback|CUDA out of memory|Killed" "$L" && break
    pgrep -f "[s]glang.launch_server" >/dev/null || break
    sleep 10
  done
  return 1
}

warm() {
  for t in "Explain how mixture-of-experts routing works." \
           "Explain the immune response to a viral infection." \
           "Explain how a B-tree index speeds up database lookups."; do
    curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"GLM5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"$t\"}],\"temperature\":0.0,\"max_tokens\":300}" >/dev/null 2>&1
  done
}

# accept length for the window that just ran, by timestamp (see capacity_paired.sh
# for why line offsets race the server's buffered writes)
accept() {  # LOG START_EPOCH
  .venv/bin/python - "$1" "$2" <<'PY' | tee -a "$M"
import datetime, re, statistics, sys
path, start = sys.argv[1], float(sys.argv[2])
t0 = datetime.datetime.fromtimestamp(start)
acc = []
for line in open(path, errors="ignore"):
    m = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', line)
    if not m or datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") < t0:
        continue
    a = re.search(r'accept len: ([\d.]+)', line)
    if a:
        acc.append(float(a.group(1)))
print(f"        accept {statistics.mean(acc):.3f} (n={len(acc)})" if acc else "        accept n/a")
PY
}

rung() {  # NAME  VISIBLE  N  MEMFRAC  MAXTOK
  local NAME=$1 VIS=$2 N=$3 MF=$4 MT=$5
  local L="logs/tpd_${NAME}.log"
  say ""
  say "=== $NAME : GPUs=[$VIS] GPU_EXPERTS=$N mem_fraction=$MF max_total_tokens=$MT ==="
  kill_server
  rm -f "$L"
  # MODE=safe explicitly: the 2xh100 profile computes its OWN gpu_experts (104)
  # for the routing decision and would hand back sub2 regardless of the N we
  # override here. sub2 must never be the default and must never be the thing
  # a TP comparison silently switches on.
  setsid env CUDA_VISIBLE_DEVICES="$VIS" \
    GPU_EXPERTS="$N" MEM_FRACTION="$MF" MAX_TOTAL_TOKENS="$MT" \
    MODE=safe KEEP=2 MTP=1 \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    ./run_fast.sh > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED"; grep -E "Error|Traceback|out of memory" "$L" | tail -4 | tee -a "$M"; return; }
  warm
  local T0=$(date +%s)
  say "   $(.venv/bin/python bench/perf_probe/decbench.py 300 8 2>&1 | tail -1)"
  accept "$L" "$T0"
  say "   VRAM $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
  say "   host RSS $(free -g | awk 'NR==2{print $3}')GB"
}

say "###### TP vs residency decomposition $(date -u) ######"
say "A (TP1 N=24) already measured: 23.74 tok/s median, accept 2.435"
rung "B_tp2_n24"  "0,1" 24  0.85 8192
rung "C_tp2_n104" "0,1" 104 0.95 81920
say ""
say "B-A isolates the second GPU (one variable). C-B is residency plus the"
say "mem_fraction/KV changes 104 experts require, so read it as an operating"
say "point, not a clean single-variable step."
say "###### done $(date -u) ######"
