# Shared, MACHINE-INDEPENDENT paths for the bench harnesses.
#
# Every harness in this directory used to hardcode one developer's scratch
# directory (`/data/tmp/claude-1002/<session-uuid>/scratchpad`), which exists on
# exactly one box. Sourcing this file instead makes the whole ladder runnable
# anywhere:
#
#   SP                  where logs, per-run markers and coherence texts land.
#                       Override with RUNGLM_SCRATCH; otherwise it follows
#                       TMPDIR, and finally /tmp.
#   TRITON_CACHE_DIR    the launchers need this OFF the root disk -- a full root
#                       disk makes the server SIGQUIT mid-run. /cache/nvme0 is
#                       this box's fast scratch; elsewhere it falls back under
#                       $SP, which is wherever the operator pointed it.
#
# Nothing here overrides a value the caller already exported.
SP="${RUNGLM_SCRATCH:-${TMPDIR:-/tmp}/runglm-bench}"
mkdir -p "$SP"

if [ -z "${TRITON_CACHE_DIR:-}" ]; then
  if [ -d /cache/nvme0 ]; then
    TRITON_CACHE_DIR=/cache/nvme0/triton-cache
  else
    TRITON_CACHE_DIR="$SP/triton-cache"
  fi
fi
export TRITON_CACHE_DIR
mkdir -p "$TRITON_CACHE_DIR"
