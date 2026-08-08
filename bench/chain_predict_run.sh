#!/usr/bin/env bash
# Statistical comparison: does walking the hidden state forward through
# RESIDENT-ONLY experts predict future routing better than showing the current
# hidden state straight to a future layer's router?
#
# See bench/chain_predict.py for the arms and what each isolates. This script
# only boots the box in the one configuration the measurement needs, drives
# enough decode steps to make the numbers stable, prints the table, and puts
# production back.
#
# WHY EAGER (DISABLE_CUDA_GRAPH=1). The chain runs arbitrary Python per layer
# per step -- a real MoE forward, a renorm, a gate -- so it cannot be captured.
# Under graphs it would collect nothing during decode and silently report an
# empty table (the KT_DUMP_TOPK trap). Decode is slow this way and that is fine:
# this measures ACCURACY, and routing accuracy is a property of the trajectory,
# not of how the kernels were launched.
#
# WHY GPU_EXPERTS=104, safe2, top2. It reproduces the configuration Stage D2 was
# measured in, so the `direct` arm is an external cross-check on the whole
# harness: it should land near the 77.0% recall / 60.8% whole-layer coverage
# already recorded at depth 1, P=2. If it does not, nothing else in the table
# can be trusted.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad
LOG="$SP/chain_pred.log"
TOKENS=${TOKENS:-260}
REQS=${REQS:-2}

rm -f bench/profile_out/chain_predict.rank*.json

pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*

echo "=== booting chain-predict instrument (eager) ==="
KT_CHAIN_PRED=1 \
KT_CHAIN_DEPTH=${KT_CHAIN_DEPTH:-4} \
KT_CHAIN_STRIDE=${KT_CHAIN_STRIDE:-6} \
KT_CHAIN_P=${KT_CHAIN_P:-2,3,4} \
KT_CHAIN_K=2 \
KT_CHAIN_DUMP_EVERY=16 \
KT_CHAIN_EXACT=${KT_CHAIN_EXACT:-0} \
DISABLE_CUDA_GRAPH=1 MTP=0 \
GPU_EXPERTS=${GPU_EXPERTS:-104} RUNGLM_TOPK_MODE=safe2 \
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$LOG" 2>&1 &

for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  grep -qaE "Traceback|CUDA out of memory|Killed" "$LOG" && { echo "BOOTFAIL"; break; }
  sleep 10
done
if ! curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "NOT UP -- last 40 log lines:"; tail -40 "$LOG"; exit 1
fi

echo "=== driving $REQS x $TOKENS decode steps (eager, this is slow) ==="
# The body MUST be one line. A newline inside a JSON string literal is an
# invalid control character, and the server rejects the whole request with a
# 400 -- which is what silently produced two empty sweeps: the requests never
# reached the model at all, so the instrument had nothing to score and the
# report died on a missing file rather than on anything to do with the walk.
PROMPT="Explain in detail how a modern CPU's branch predictor works, request"
for i in $(seq 1 "$REQS"); do
  curl -s -m 2400 http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"GLM5.2\",\"max_tokens\":$TOKENS,\"temperature\":0.7,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT $i.\"}]}" \
    | .venv/bin/python -c "import json,sys; s=sys.stdin.read(); d=json.loads(s); \
        c=(d.get('choices') or [{}])[0]; \
        print('  reply chars:', len((c.get('message') or {}).get('content') or ''), \
              'finish:', c.get('finish_reason')) if c else \
        print('  request rejected:', s[:300])" \
    || echo "  request $i failed"
done

# The instrument dumps every 16 steps; give the last one a moment to land.
sleep 3
grep -a "kt-chain" "$LOG" | head -5
echo
echo "=== RESULT ==="
.venv/bin/python bench/chain_predict_report.py || true

echo "=== restoring production ==="
pkill -f sglang.launch_server >/dev/null 2>&1 || true
sleep 8
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 5; done
rm -f /dev/shm/ktstore_*
KT_GPU_PREFILL_THRESHOLD=0 TRITON_CACHE_DIR=/cache/nvme0/triton-cache \
  nohup ./run_fast.sh > "$SP/chain_prod.log" 2>&1 &
for i in $(seq 1 300); do
  curl -sf -m 3 http://127.0.0.1:8000/health_generate >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "PROD UP" || echo "PROD FAILED TO RESTORE"
echo "=== chain-predict done ==="
