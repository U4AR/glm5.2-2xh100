# BATCH C -- the direct-predictor rows voided by the NEXTN draft-capture bug.
# MEM_FRACTION 0.94 (vs 0.95) frees ~1GB/card WITHOUT changing which experts are
# resident, so coverage -- and comparability with the d1/d2 rows -- is untouched;
# only the KV pool shrinks, which a 200-token generation never notices. The
# unpatched d3 died on capture, and CPF-d3 later died on the VRAM cliff this
# config sits on (100 GPU experts + 4 landing slots), so both are guarded now.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
until grep -qa "=== acc2 done ===" "$SP/chain_acc2.out"; do sleep 30; done
OUT=bench/profile_out/prefetch_rate.json
boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/redo_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/redo_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -3
  echo -n "  counters: "; grep -a "kt-prefetch\] step" "$SP/redo_$label.log" | tail -1 || echo "NONE -- VOID"
}
# Same-config reference at the new MEM_FRACTION, so d3/d4 are read against a
# baseline measured in THEIR box state rather than the earlier one.
boot REDO-base KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0
boot REDO-d3 KT_PRED_LOOKAHEAD=3 KT_PREFETCH_DEPTH=3
boot REDO-d4 KT_PRED_LOOKAHEAD=4 KT_PREFETCH_DEPTH=4
echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/redo_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -s -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== redo done ==="
