#!/usr/bin/env bash
# Grade a patch file with the OFFICIAL SWE-bench Pro evaluator, locally.
#
#   bash bench/swebenchpro/run_eval.sh gold_patches.json out_gold
#   bash bench/swebenchpro/run_eval.sh patches_top2.json  out_top2
#
# The upstream script resolves `dockerfiles/` and `run_scripts/` relative to its
# own directory, so it must be run from there. It reports an instance resolved
# only when every FAIL_TO_PASS and PASS_TO_PASS test passes.
#
# Run gold_patches.json FIRST. Gold must score 5/5; anything less means the
# harness is broken and no model number from it means anything.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
PATCHES="${1:-gold_patches.json}"
OUT="${2:-out_gold}"
WORKERS="${WORKERS:-2}"          # each worker runs a full pytest suite in a container
PY="${PY:-/data/models/RunGLM/.venv/bin/python}"

mkdir -p "$HERE/$OUT"
cd ../swebenchpro_upstream
exec "$PY" swe_bench_pro_eval.py \
  --raw_sample_path "$HERE/instances.csv" \
  --patch_path "$HERE/$PATCHES" \
  --output_dir "$HERE/$OUT" \
  --scripts_dir run_scripts \
  --dockerhub_username jefzda \
  --num_workers "$WORKERS" \
  --use_local_docker
