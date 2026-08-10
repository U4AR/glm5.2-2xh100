#!/usr/bin/env bash
# Why does a CORRECT prefetch still cost 20% of accept length?
#
# With the stale-mask defect fixed, the numbers are finally trustworthy and they
# are bad: 66.11 ms/step against a 66.17 baseline (a wash), and accept 2.273 vs
# 2.857 -- 88 forward steps every single run, not 70. Deterministic, so it is a
# real effect and no longer an artefact.
#
# The standing explanation is numerics: a landed expert is computed by the
# cutlass W4AFP8 kernel instead of the CPU packed-int4 W4A8 one, the target's
# logits shift, and the NextN draft -- which is NOT prefetched -- mispredicts
# more often. That is plausible but unproven, and 20% is a lot to blame on
# rounding. The alternative is a residual correctness defect that determinism
# alone cannot rule out: a reproducible wrong answer is still wrong.
#
# WHERE the completions first disagree separates the two. Rounding-level
# differences track each other for many tokens before drifting; an expert
# dropped, double-counted, or routed to the wrong slot shows up in the first few.
#
# The MTP=0 pair is the clean room. With no speculative decoding there is no
# draft, no accept length, and tok/s is just 1000/ms_per_step -- so the text
# comparison isolates what prefetch does to the TARGET MODEL ALONE, and the
# timing comparison gives the step-rate verdict with nothing to confound it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
DET=bench/profile_out/why_accept_det.json
RATE=bench/profile_out/why_accept_rate.json
TXT=bench/profile_out/why_accept_txt
rm -rf "$TXT"; mkdir -p "$TXT"

boot () {
  local label="$1"; local mtp="$2"; shift 2
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label (MTP=$mtp): $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP="$mtp" MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    nohup ./run_fast.sh > "$SP/w_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/w_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP $label"; return 1; }
  .venv/bin/python bench/determinism.py --runs 3 --tokens 200 --tier 2 \
    --label "$label" --out "$DET" --save-text "$TXT" 2>&1 | tail -10
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier 2 --runs 3 \
    --tokens 200 --label "$label" --out "$RATE" 2>&1 | tail -2
  echo -n "  counters: "
  grep -a "kt-prefetch\] step" "$SP/w_$label.log" | tail -1 || echo none
}

OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1"

boot W0-base 0 $OFF
boot W0-full 0 $ON
boot W1-base 1 $OFF
boot W1-full 1 $ON

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/w_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== why_accept done ==="
