#!/usr/bin/env bash
# Reorder the queue: rung J first (asked for twice), then H/I, then the ladder.
# Kept in a FILE rather than an inline `bash -c` string because the pkill
# patterns below would otherwise match the caller's own command line and kill
# the launching shell -- which is exactly what happened on the first attempt
# (exit 144, driver never started).
set -uo pipefail
cd /data/models/RunGLM

for p in $(pgrep -f "[s]glang.launch_server"); do kill "$p" 2>/dev/null; done
for i in $(seq 1 40); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 3; done
for p in $(pgrep -f "[s]glang"); do kill -9 "$p" 2>/dev/null; done
sleep 10

bash experiments/expert_tiering_ssd/ssd_at_full_cpu.sh
bash experiments/expert_tiering_ssd/rtx6000_no_thrash.sh
bash experiments/expert_tiering_ssd/min_vram_25tps.sh
