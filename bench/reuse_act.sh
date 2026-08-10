#!/usr/bin/env bash
# Turn slot reuse ON and price it.
#
# Measured first (bench/reuse_probe.sh, mode 1): 41.9% of every fetch is an
# expert the landing slot ALREADY holds -- 0.67 of 1.59 experts per layer-call.
# At the Stage H15 slope of ~5.1 ms per expert/call that is 3.40 ms/step of
# bytes moved for nothing. Mode 1 itself was free (64.38 vs 64.33 ms/step), so
# the probe is not paying for its own answer.
#
# PREDICTION, recorded before the run so it can be wrong:
#   tier 0 cost  10.84 -> ~7.4 ms   (cost is bytes; 0.67 fewer experts x 5.1)
#   tier 2 step  64.33 -> ~61.0 ms  (net 1.88 -> ~5.3, about 8% over 66.21)
# If tier 0 does NOT fall by ~3.4 ms then the reuse is not removing bytes and
# nothing else in the row matters.
#
# WHY THE TEXT MAY LEGITIMATELY CHANGE. Reuse frees slots, so a layer can cover
# MORE experts than before; each newly covered expert moves from the CPU int4
# kernel to the cutlass GPU kernel. Same expert, different arithmetic. So the
# completion hash is NOT required to match C-d1P2's `45c4fbb027`. What IS
# required: determinism across repeats, coherent prose, and an accept length in
# the normal 2.7-2.9 band. A changed hash is expected; garbage is not.
#
# The verify runs FIRST, on a free GPU, and the boot is skipped if it fails --
# the kernel now publishes routing for experts it deliberately does not fetch,
# which is exactly the class of bug that produced fluent, quietly-wrong output
# in Stage H11.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, wherever it is
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_wait_for.sh"
RATE=bench/profile_out/reuse_act_rate.json
DET=bench/profile_out/reuse_act_det.json

wait_for_marker "=== reuse_probe done ===" reuse_probe.out

bash bench/_kill_servers.sh >/dev/null
sleep 5

echo "=== verifying the reuse kernel against the reference path ==="
# Once, not twice: it compiles and runs 250 trials x 6 flag combos.
VOUT="$SP/reuse_verify.txt"
.venv/bin/python bench/pred_fused_kernel.py --verify --trials 250 > "$VOUT" 2>&1
tail -15 "$VOUT"
if ! grep -qa "VERIFY OK" "$VOUT"; then
  echo "VERIFY FAILED -- not booting. Fix the kernel first."
  exit 1
fi

echo "=== booting R-act: KT_PREFETCH_REUSE=2 (acting) ==="
KT_PREFETCH_GATHER=1 KT_PREFETCH_ROUTE=1 KT_PREFETCH_CPUSKIP=1 \
KT_PRED_FUSED=1 KT_PRED_POINT=pre KT_PREFETCH_FRESH=1 KT_PRED_FUSED_DEPTH=1 \
KT_PREFETCH_REUSE=2 \
GPU_EXPERTS=100 KT_STORE_SHM=1 KT_PREFETCH_SLOTS=4 KT_PREFETCH_BLOCKS=8 \
RUNGLM_TOPK_MODE=safe2 MTP=1 MEM_FRACTION=0.94 \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/r_act.log" 2>&1 &

for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$SP/r_act.log" && { echo "BOOTFAIL"; exit 1; }
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "NOT UP"; tail -30 "$SP/r_act.log"; exit 1; }

.venv/bin/python bench/determinism.py --runs 3 --tokens 200 --tier 2 \
  --label R-act --out "$DET" --save-text "$SP/r_act_text" 2>&1 | grep -aE "distinct|DETERMIN" || true
for t in 0 2; do
  .venv/bin/python bench/prefetch_rate.py --model GLM5.2 --tier $t --runs 3 \
    --tokens 200 --label "R-act-top$t" --out "$RATE" 2>&1 | grep -a "ms/step" || true
done
echo "--- counters ---"
grep -a "kt-prefetch\]" "$SP/r_act.log" | tail -4
echo "--- first 400 chars of the completion (coherence, by eye) ---"
head -c 400 "$SP/r_act_text/R-act.txt" 2>/dev/null || echo "(no text captured)"
echo
echo "=== restoring production ==="
bash bench/_kill_servers.sh >/dev/null
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  nohup ./run_fast.sh > "$SP/ract_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED"
echo "=== reuse_act done ==="
