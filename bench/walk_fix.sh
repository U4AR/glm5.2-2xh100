#!/usr/bin/env bash
# THE FIX for Stage H2's biggest attributed term.
#
# The substitution SEARCH cost 6.55 ms/step (87 us/layer, ~20 tiny kernels: a
# topk across all 256 experts, two argsorts, gathers, scatters). But
# mask_and_remap_expert_ids already writes -1 -- the kernel's skip sentinel --
# into every non-resident slot, so DROPPING them is free. KT_CHAIN_PF_SUB=drop
# keeps the resident slots, zeroes the rest, renormalises: ~3 kernels.
#
# Four rows, same box state, read against yesterday's ladder measured hours ago
# on the same day (s0 66.59 / s0b 66.67, noise 0.08; s2 73.42; s6 90.30):
#   DROP-s3   isolates the new substitution cost   (vs search's s3 79.97)
#   DROP-s6   total walk cost with the fix         (vs search's s6 90.30)
#   *-real    the shippable configuration, prefetch EFFECT ON, both modes, so
#             the tok/s comparison is same-boot-state rather than cross-day.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
OUT=bench/profile_out/prefetch_rate.json
boot () {
  local label="$1"; shift
  pkill -f sglang.launch_server >/dev/null 2>&1 || true
  sleep 8
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
  rm -f /dev/shm/ktstore_*
  echo "=== booting $label: $* ==="
  env "$@" GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
    RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
    KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
    nohup ./run_fast.sh > "$SP/fix_$label.log" 2>&1 &
  for i in $(seq 1 300); do
    curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
    grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/fix_$label.log" && { echo "BOOTFAIL $label"; return 1; }
    sleep 10
  done
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOTUP $label"; return 1; }
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2-top2 --runs 3 --tokens 200 \
    --label "$label" --out "$OUT" 2>&1 | tail -3
}
OFF="KT_PREFETCH_GATHER=0 KT_PREFETCH_ROUTE=0 KT_PREFETCH_CPUSKIP=0"
ON="KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1"

boot DROP-s3 $OFF KT_CHAIN_PF=1 KT_CHAIN_PF_DEPTH=2 KT_CHAIN_PF_SUB=drop KT_CHAIN_PF_STAGE=3
boot DROP-s6 $OFF KT_CHAIN_PF=1 KT_CHAIN_PF_DEPTH=2 KT_CHAIN_PF_SUB=drop KT_CHAIN_PF_STAGE=6
boot SEARCH-real $ON KT_CHAIN_PF=1 KT_CHAIN_PF_DEPTH=2 KT_CHAIN_PF_K=2 KT_CHAIN_PF_SUB=search
boot DROP-real   $ON KT_CHAIN_PF=1 KT_CHAIN_PF_DEPTH=2 KT_CHAIN_PF_SUB=drop
# Reference in THIS box state, so the improvement is not read across a reboot.
boot FIX-base $OFF KT_CHAIN_PF=0

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/fix_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== walk fix done ==="
