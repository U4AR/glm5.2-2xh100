#!/usr/bin/env bash
# How much does the SAME server, unchanged, vary between measurements?
# Every comparison on the board (33.45 vs 29.67, 34.82 vs 31.07) is a
# difference of one measurement against another. RESULTS.md's own repeat BOOTS
# of one config span 31.16-38.67, so the boot-to-boot term is huge -- but the
# within-boot term has never been measured at all, and it is the one that
# decides whether a single "converged" number means anything.
set -uo pipefail
cd /data/models/RunGLM
O=logs/noise_within_boot.log
: > $O
for i in $(seq 1 10); do
  echo "block $i: $(.venv/bin/python bench/perf_probe/decbench.py 300 12 2>&1 | tail -1)" | tee -a $O
done
echo "=== within-boot noise complete ===" | tee -a $O
