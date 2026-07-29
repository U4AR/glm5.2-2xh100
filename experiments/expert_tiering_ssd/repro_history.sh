#!/usr/bin/env bash
# Can today's tree still reproduce the 2026-06-30 headline on the SAME inputs?
#
# The claim under test is run_fast.sh's own header: KEEP=2 + MTP depth-3 =
# 34.1 tok/s on 2xH100, GPU_EXPERTS=104. A lot has changed in the tree since --
# tier movement, the incremental swap, placement strategies, the KV pool default
# -- and if that number no longer comes back, something regressed and the cause
# matters more than any new measurement.
#
# Reproduce the 06-30 conditions exactly, not approximately:
#   MODE=safe        06-30's "KEEP=2 sub" predates the 07-07 residency filter
#                    (3e39132). Under those semantics `sub` WAS today's `safe`,
#                    so `safe` is the faithful replay and today's `sub2` is not.
#   MAX_TOTAL_TOKENS=4096   the benchmark cap of the era, before the 81920 KV
#                    pool fix. Restored here because it is part of the config
#                    being replayed, not because 4096 is a good default.
#   MEM_FRACTION=0.95, GPU_EXPERTS=104, MTP depth-3, decbench 200 5
#                    -- the 5-run median the headline was quoted from.
#
# THREE boots, and that is not optional. This project has documented a
# boot-to-boot spread of 31.16-38.67 tok/s for one unchanged config, and its own
# RESULTS.md concludes: "use multi-run or converged measurements, not one fresh
# boot". A single median landing under 34 is therefore NOT evidence of a
# regression -- 34.1 itself sits inside that band. What would be evidence is
# every boot landing below the band.
set -uo pipefail
cd /data/models/RunGLM
M=logs/repro_history.log
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

say "###### 06-30 headline repro (target 34.1 tok/s, band 31.2-38.7) $(date -u) ######"

for boot in 1 2 3; do
  L="logs/repro_hist_b${boot}.log"
  say ""
  say "=== boot $boot : TP2 GPU_EXPERTS=104 safe2 MTP-d3 mem0.95 maxtok4096 ==="
  kill_server
  rm -f "$L"
  setsid env \
    GPU_EXPERTS=104 MEM_FRACTION=0.95 MAX_TOTAL_TOKENS=4096 \
    MODE=safe KEEP=2 MTP=1 SPEC_STEPS=3 SPEC_DRAFT_TOKENS=4 \
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    ./run_fast.sh > "$L" 2>&1 < /dev/null &
  disown
  wait_ready "$L" || { say "   BOOT FAILED"; grep -E "Error|Traceback|out of memory" "$L" | tail -4 | tee -a "$M"; continue; }
  warm
  # the headline was a 5-run median of decbench 200
  say "   $(.venv/bin/python bench/perf_probe/decbench.py 200 5 2>&1 | tail -1)"
  say "   $(grep -oE 'accept len: [0-9.]+' "$L" | tail -30 | awk '{s+=$3;n++} END{printf "accept %.3f (n=%d)", s/n, n}')"
done

say ""
say "Read the three medians as a set. 34.1 sits inside the documented 31.2-38.7"
say "boot-to-boot band, so one low boot proves nothing; three low boots do."
say "###### repro complete $(date -u) ######"
