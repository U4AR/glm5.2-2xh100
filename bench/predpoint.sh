#!/usr/bin/env bash
# Item 3's actionable finding, tested on the SHIPPED path instead of the bench
# mechanism -- and it needs no code change at all.
#
# The attribution said the whole 3.1 ms gap is the fetch effect, and that the
# cause is WHICH hidden state the predictor shows to the next layer's router:
# the pre-MoE input (74.3% whole-layer coverage) or the post-layer residual
# stream (79.5%). `deepseek_v2.py` already carries that switch --
# KT_PRED_POINT=pre|post -- and the post hook feeds _kt_pred_emit, which drives
# pf_issue, so it is a real mechanism and not just an instrument.
#
# Prediction to be judged against, committed before the run: PP-post should land
# near CPF-d1's 70.43 and about 3 ms below PP-pre. If it does, the finding
# reproduces on the shipped path and the change is a one-line default. If PP-post
# lands with PP-pre, the gain belonged to bench/chain_prefetch.py's own code, not
# to the choice of state, and item 3 reopens.
#
# The fused selection kernel (item 2) is verified and benched first, while the
# GPU is empty -- it needs no server and the box has ~150 MiB free once one is up.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/prefetch_rate.json

while pgrep -f "bash bench/drop_raw_acc.sh" >/dev/null 2>&1 ||
      pgrep -f "bash bench/drop_acc.sh" >/dev/null 2>&1 ||
      pgrep -f "bash bench/chain_predict_run.sh" >/dev/null 2>&1; do sleep 30; done
echo "=== drop_raw queue clear ==="
sleep 20

bash bench/_kill_servers.sh
echo "=== item 2: fused selection kernel, verify then bench ==="
.venv/bin/python bench/pred_select_kernel.py --verify --trials 100 2>&1 | tail -12
.venv/bin/python bench/pred_select_kernel.py --bench 2>&1 | tail -6

boot () {
  local label="$1"; shift
  bash bench/_kill_servers.sh >/dev/null
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/pp_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/pp_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -3
}
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"

boot PP-base $ON KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot PP-pre  $ON KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 KT_PRED_POINT=pre
boot PP-post $ON KT_PRED_LOOKAHEAD=1 KT_PREFETCH_DEPTH=1 KT_PRED_POINT=post

echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/pp_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== predpoint done ==="
