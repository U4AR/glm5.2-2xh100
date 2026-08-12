#!/usr/bin/env bash
# Launch the benchmark server from the SAME config the watchdog restarts with.
#   bash bench/deepswe/start_server.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/server_env.sh"
cd /data/models/RunGLM
setsid nohup ./run_fast.sh >> "$HERE/server.log" 2>&1 < /dev/null &
echo "launched; waiting for health"
for _ in $(seq 1 40); do
  sleep 15
  [ "$(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:8000/health)" = "200" ] && break
done
echo "health=$(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:8000/health)"
# A boot-time scheduler exception is survivable-looking and fatal an hour later,
# so assert on it rather than trusting the health check.
echo "exceptions=$(grep -cE 'Scheduler hit an exception|OutOfMemoryError|Not enough memory' "$HERE/server.log")"
grep -oE 'context_length=[0-9]+|max_total_tokens=[0-9]+|gpu_experts=[0-9]+|sleep_on_idle=[01]' "$HERE/server.log" | tail -4
