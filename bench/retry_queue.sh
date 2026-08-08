#!/usr/bin/env bash
# Re-run the two measurements the readiness race killed, in order.
#
# The race: every boot loop polled with `curl -s .../health_generate && break`.
# `curl -s` exits 0 on an HTTP 503, so the moment the HTTP server started
# answering -- which is BEFORE the engine is ready -- the loop broke, the health
# gate passed for the same reason, and the requests went out into a server that
# was still warming up and got 400s back. Fixed with `curl -sf`, which exits
# non-zero on 4xx/5xx. It has always been latent; it only bit when the HTTP
# layer happened to come up two seconds ahead of the engine.
set -uo pipefail
cd /data/models/RunGLM
SP=/data/tmp/claude-1002/-data-models-RunGLM/0f5c5fd4-e086-4ca7-84f8-858b327967bf/scratchpad

while pgrep -f "bash bench/drop_acc.sh" >/dev/null 2>&1 ||
      pgrep -f "bash bench/chain_predict_run.sh" >/dev/null 2>&1; do sleep 20; done
sleep 20

bash bench/drop_acc.sh
bash bench/item3_attrib.sh
echo "=== retry queue done ==="
