#!/usr/bin/env bash
# Portable safe2 + MTP + decode-time adaptive expert-cache entry point.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/config.sh"

if [ "${AUTO_PROFILE:-1}" = "1" ]; then
  while IFS='=' read -r key value; do
    [ -n "$key" ] || continue
    if [ -z "${!key:-}" ]; then
      export "$key=$value"
    fi
  done < <(python3 "$REPO/scripts/hardware_profile.py" --shell --adaptive)
fi

python3 "$REPO/scripts/hardware_profile.py" --check --adaptive --weights-dir "$W4AFP8_MODEL"
echo "[run_adaptive] profile=${RUNGLM_PROFILE:-manual} gpu_experts=$GPU_EXPERTS tp=$TP_SIZE cpuinfer=$CPUINFER"
exec bash "$REPO/experiments/adaptive_expert_cache/decode_cache/boot_adaptive_mtp.sh"
