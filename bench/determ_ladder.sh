#!/usr/bin/env bash
# Which half of the prefetch makes decode non-reproducible?
#
# Production, 5 runs, temperature 0: ONE completion hash, five times. The base
# decode path is a function. Prefetch-on produced step counts [73, 77, 75] for
# the same prompt, which a fixed numerics difference cannot do -- W4AFP8 rounding
# differently from W4A8 is still deterministic and would give the same answer
# every run. So the accept drop is a RACE, and this splits the path to find it.
#
# The prefetch has two consumers of one prediction, and they can be enabled
# apart:
#   ROUTE   -- pf_landed:     the GPU computes the landed expert from its slot
#   CPUSKIP -- pf_landed_cpu: the CPU is told (-1 sentinel) not to compute it
# Enabling exactly one is numerically WRONG on purpose -- ROUTE alone counts the
# expert twice, CPUSKIP alone counts it zero times -- but wrongness is not what
# is being measured here. Determinism is. A deliberate double-count still has to
# produce the same double-count every run.
#
#   D-base    everything off                 control IN THIS CONFIG, not prod's
#   D-gather  GATHER only                    bytes move, nothing reads them.
#                                            MUST hash-match D-base; if it does
#                                            not, the transfer is clobbering
#                                            live weights and nothing after this
#                                            row means anything
#   D-route   GATHER+ROUTE                   GPU-side consumer alone
#   D-full    GATHER+ROUTE+CPUSKIP           the shipped combination
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/determinism.json

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/d_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/d_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 5 --tokens 200 --tier 2 \
    --label "$label" --out "$OUT" 2>&1 | tail -14
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/d_$label.log" | tail -1 || echo none
}

FUSED="KT_PRED_FUSED=1 KT_PRED_POINT=pre"

boot D-base   KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot D-gather KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0 $FUSED
boot D-route  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=0 $FUSED
boot D-full   KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 $FUSED

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/d_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== determ_ladder done ==="
