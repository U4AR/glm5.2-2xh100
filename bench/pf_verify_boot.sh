#!/usr/bin/env bash
# Is the gather producing the layout cutlass actually consumes?
#
# The gather has only ever been checked against an OLDER GATHER
# (gather_kernel_probe.py --verify, byte-identical at 8/64/256 blocks). Two
# implementations agreeing says nothing about whether either is right. If kt's
# RAWINT4 store packs nibbles differently from the W4AFP8 checkpoint -- order,
# sign convention, group axis -- then every prefetched expert has been computing
# plausible garbage, deterministically, and the only symptom is that quality
# quietly drops. Which is exactly what the fixed prefetch does: 2.273 accept vs
# 2.857, reproducible to the step.
#
# Ground truth is free. Most experts are GPU-resident, so the correct cutlass
# bytes for expert e already sit at w13_weight[logical_to_gpu_index[e]]. Gather
# that same expert into a landing slot; the two must be byte-identical on all
# four tensors. W4AFp8MoEMethod.process_weights_after_loading rewrites only the
# scales, never w13_weight/w2_weight, so a raw weight copy is the right thing and
# the comparison is exact.
#
# The check runs inside pf_build_table, during weight load, so this never needs
# to serve a request -- boot until the layer-40 line lands, then stop.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
bash bench/_kill_servers.sh >/dev/null

KT_PREFETCH_VERIFY=1 KT_PREFETCH_VERIFY_LAYERS=3,40 \
  KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
  KT_PRED_FUSED=1 KT_PRED_POINT=pre \
  GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
  RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
  KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/pfverify.log" 2>&1 &

for i in $(seq 1 200); do
  grep -qa "kt-prefetch\]\[verify\] layer=40 .*w2_scale" "$SP/pfverify.log" && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/pfverify.log" && { echo BOOTFAIL; break; }
  sleep 5
done

echo "=== verify lines ==="
grep -a "kt-prefetch\]\[verify\]" "$SP/pfverify.log" || echo "NO VERIFY LINES"
bash bench/_kill_servers.sh >/dev/null
echo "=== pf_verify done ==="
