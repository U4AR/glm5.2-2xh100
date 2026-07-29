#!/usr/bin/env bash
# Fetch the ~373 GB W4AFP8 checkpoint, safe to run CONCURRENTLY with setup.sh.
#
# Why this exists as its own script: the download is network/disk bound and the
# kt-kernel build is CPU bound, so on a fresh machine they should overlap
# instead of running end to end. setup.sh starts this in the background and
# waits for it at the end, but it is also fine to run on its own:
#
#     ./scripts/download_weights.sh              # foreground
#     ./scripts/download_weights.sh &            # alongside anything else
#
# It deliberately does NOT use "$VENV". setup.sh recreates that venv and pip
# installs into it while this runs; a downloader importing huggingface_hub out
# of a directory that is being rewritten underneath it is a race. So this
# bootstraps a tiny private venv (.dl-venv) holding only huggingface_hub, which
# nothing else touches.
#
# Resumable: snapshot_download skips files that are already complete, so an
# interrupted run costs only the partial shard.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/config.sh"

DL_VENV="${DL_VENV:-$REPO/.dl-venv}"
PYTHON="${PYTHON:-python3.12}"
LOCK="$W4AFP8_MODEL.download.lock"

log() { echo "[download-weights] $*"; }

# --- already complete? ------------------------------------------------------
# Checked against the checkpoint's own shard index rather than a marker file, so
# a half-finished download from a killed run is correctly seen as incomplete.
# Uses the system python: no dependency on either venv existing yet.
complete() {
  python3 - "$W4AFP8_MODEL" <<'PY'
import json, os, sys
d = sys.argv[1]
index = os.path.join(d, "model.safetensors.index.json")
if not os.path.isfile(index):
    sys.exit(1)
try:
    shards = set(json.load(open(index))["weight_map"].values())
except (ValueError, KeyError):
    sys.exit(1)
for name in sorted(shards) + ["config.json", "tokenizer.json"]:
    path = os.path.join(d, name)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        sys.exit(1)
sys.exit(0)
PY
}

if complete; then
  log "checkpoint already complete at $W4AFP8_MODEL -- nothing to do"
  exit 0
fi

# --- one downloader at a time ----------------------------------------------
# Two concurrent snapshot_download runs into one local_dir corrupt each other's
# partial files. If setup.sh started one, a manual invocation should attach to
# that rather than race it.
mkdir -p "$(dirname "$LOCK")"
if ! ( set -o noclobber; echo $$ > "$LOCK" ) 2>/dev/null; then
  other="$(cat "$LOCK" 2>/dev/null || true)"
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    log "another download is already running (pid $other); waiting for it"
    while kill -0 "$other" 2>/dev/null; do sleep 10; done
    complete && { log "the other download finished successfully"; exit 0; }
    log "the other download exited without completing the checkpoint" >&2
    exit 1
  fi
  log "removing a stale lock from pid ${other:-unknown}"
  rm -f "$LOCK"
  echo $$ > "$LOCK"
fi
trap 'rm -f "$LOCK"' EXIT

# --- private interpreter ----------------------------------------------------
if [ ! -x "$DL_VENV/bin/python" ]; then
  command -v "$PYTHON" >/dev/null || PYTHON=python3
  log "creating $DL_VENV for the downloader"
  "$PYTHON" -m venv "$DL_VENV" || { log "could not create $DL_VENV" >&2; exit 1; }
fi
"$DL_VENV/bin/python" -c 'import huggingface_hub' 2>/dev/null || {
  log "installing huggingface_hub into $DL_VENV"
  "$DL_VENV/bin/python" -m pip install --quiet --upgrade pip || true
  "$DL_VENV/bin/python" -m pip install --quiet huggingface_hub || {
    log "could not install huggingface_hub" >&2; exit 1; }
}

log "fetching into $W4AFP8_MODEL (~373 GB, resumable, HF_MAX_WORKERS=${HF_MAX_WORKERS:-1})"
"$DL_VENV/bin/python" "$REPO/int4_scripts/download_w4afp8.py" || {
  log "download FAILED -- rerun this script, it resumes" >&2
  exit 1
}

complete || { log "download reported success but the shard index is incomplete" >&2; exit 1; }
log "checkpoint complete at $W4AFP8_MODEL"
