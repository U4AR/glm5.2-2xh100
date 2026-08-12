#!/usr/bin/env bash
# Run the 5 DeepSWE v1.1 border tasks against the local GLM-5.2 server at one tier.
#
#   bash bench/deepswe/run_tier.sh top2      # fast tier (expert substitution, KEEP=2)
#   bash bench/deepswe/run_tier.sh top8      # full routing, same boot
#
# WHY THESE FIVE: they are the tasks where Datacurve's own graded runs of
# glm-5-2::effort=high passed some of 4 rollouts and failed others -- the measured
# border of the model's ability, not a guess. Pulled from
# https://deepswe.datacurve.ai/artifacts/v1.1/trials.json (see select_border.py).
# Chosen for the lowest peak context among that set, because our server tops out
# at 81920 tokens and a task that overflows it fails for a memory reason rather
# than an intelligence one.
#
# WHAT MATCHES THE PUBLISHED RUNS: harness (mini-swe-agent via Pier), prompts,
# task images, step budget (unlimited), cost limit (disabled), and sampling --
# mini.yaml pins no temperature, so both the graded runs and these use the
# provider's model default. model_class=litellm forces the chat-completions path,
# which is what the graded runs used via the zai provider; leaving it on "auto"
# would silently switch to the Responses API because our model string starts
# with openai/.
#
# WHAT DOES NOT MATCH, and why:
#   * agent wall-clock timeout x4 (90 min -> 6 h). The graded runs used a cloud
#     endpoint several times faster per token; at the stock timeout every task
#     here would score as agent_timeout, which measures this box, not the tier.
#   * --cpus/--memory ignore. Rootless Docker on this host has no cgroup CPU
#     delegation, so NanoCPUs cannot be set at all. Affects test-suite speed only.
# Neither touches the prompts, the model's decisions, or the pass criterion.
set -euo pipefail
cd "$(dirname "$0")"

TIER="${1:-top2}"
BASE="${LLM_BASE:-http://10.0.2.2:8000/v1}"   # 10.0.2.2 = this host, as seen from
                                              # a rootless-Docker container
TASKS=(
  wazero-multi-module-snapshots        # glm-5-2 high: 1/4
  httpx-deterministic-cookie-store     # 2/4
  onedump-dump-encryption-pipeline     # 2/4
  ts-pattern-match-each                # 2/4
  psd-tools-blend-range-api            # 3/4
)

for t in "${TASKS[@]}"; do
  job="${TIER}_${t}"
  if [ -f "jobs/$job/result.json" ]; then
    echo "[skip] $job already has a result"
    continue
  fi
  echo "=== $TIER :: $t ==="
  pier run -p "deep-swe/tasks/$t" \
    -a mini-swe-agent -m "openai/GLM5.2-${TIER}" \
    --ak model_class=litellm \
    --ae "OPENAI_BASE_URL=$BASE" \
    --ae OPENAI_API_KEY=dummy --ae MSWEA_API_KEY=dummy \
    --cpus ignore --memory ignore \
    --agent-timeout-multiplier 4 -n 1 \
    -o jobs --job-name "$job" -y || echo "[warn] $job errored; continuing"
done

echo "=== $TIER done ==="
python3 report.py "$TIER"
