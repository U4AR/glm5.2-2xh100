#!/usr/bin/env bash
# Clone-to-build bootstrap. Run this on the final GPU pod.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/config.sh"

PYTHON="${PYTHON:-python3.12}"
KT_REPO="${KT_REPO:-https://github.com/U4AR/ktransformers.git}"
KT_BRANCH="${KT_BRANCH:-glm5.2-2xh100-stable}"
KT_COMMIT="${KT_COMMIT:-512802b9025d149681401f1c63519afa8caa34ea}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

command -v "$PYTHON" >/dev/null || {
  echo "Python 3.12 is required (or set PYTHON=/path/to/python3.12)." >&2
  exit 1
}
command -v nvcc >/dev/null || {
  echo "CUDA toolkit/nvcc is required to build for this machine." >&2
  exit 1
}
"$PYTHON" "$REPO/scripts/hardware_profile.py" --check --weights-dir "$W4AFP8_MODEL"

if [ "$INSTALL_SYSTEM_DEPS" = "1" ]; then
  if [ "$(id -u)" = "0" ]; then
    APT=(apt-get)
  else
    command -v sudo >/dev/null || {
      echo "INSTALL_SYSTEM_DEPS=1 requires root or sudo." >&2
      exit 1
    }
    APT=(sudo apt-get)
  fi
  "${APT[@]}" update
  "${APT[@]}" install -y git git-lfs pkg-config libhwloc-dev libnuma-dev build-essential
fi

# --- start the 373 GB checkpoint download NOW, in parallel with the build ----
# The download is network/disk bound and the kt-kernel build is CPU bound, so
# running them end to end wastes whichever resource is idle. Started here, after
# the apt block (which may install python3-venv) and before the build, which is
# the long pole. It runs in its own venv so setup's pip installs cannot disturb
# it -- see scripts/download_weights.sh. DOWNLOAD_WEIGHTS=0 skips it, e.g. when
# the checkpoint is already on a mounted volume.
DOWNLOAD_WEIGHTS="${DOWNLOAD_WEIGHTS:-1}"
DL_PID=""
if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then
  mkdir -p "$REPO/logs"
  "$REPO/scripts/download_weights.sh" > "$REPO/logs/download_weights.log" 2>&1 &
  DL_PID=$!
  echo "[setup] weights download started in background (pid $DL_PID); progress: tail -f $REPO/logs/download_weights.log"
fi

if [ ! -d "$REPO/ktransformers/.git" ]; then
  git clone --recursive --branch "$KT_BRANCH" "$KT_REPO" "$REPO/ktransformers"
fi
git -C "$REPO/ktransformers" fetch origin "$KT_BRANCH"
if ! git -C "$REPO/ktransformers" cat-file -e "$KT_COMMIT^{commit}" 2>/dev/null ||
   ! git -C "$REPO/ktransformers" merge-base --is-ancestor \
     "$KT_COMMIT" HEAD; then
  echo "KTransformers HEAD must contain validated commit $KT_COMMIT." >&2
  echo "Preserving the existing checkout; update it with git pull --ff-only." >&2
  exit 1
fi
git -C "$REPO/ktransformers" submodule update --init --recursive

# Preserve source overlays before install overwrites .venv. Never preserve the
# tracked Zen4/Hopper .so: kt-kernel must be compiled on this machine.
OVERLAY_BACKUP="$(mktemp -d)"
trap 'rm -rf "$OVERLAY_BACKUP"' EXIT
while IFS= read -r tracked; do
  case "$tracked" in
    *.py)
      relative="${tracked#".venv/"}"
      mkdir -p "$OVERLAY_BACKUP/$(dirname "$relative")"
      cp "$REPO/$tracked" "$OVERLAY_BACKUP/$relative"
      ;;
  esac
done < <(git -C "$REPO" ls-files '.venv/**')

"$PYTHON" -m venv "$VENV"
# Never reuse a binary from an earlier setup attempt or copied workspace.
# kt-kernel must be compiled against this machine's CPU, Python, torch and CUDA.
find "$VENV/lib/python3.12/site-packages/kt_kernel" -maxdepth 1 \
  -type f -name 'kt_kernel_ext*.so' -delete 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install --upgrade pip
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$VENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$VENV"
export CMAKE_LIBRARY_PATH="$VENV/lib"
export CMAKE_INCLUDE_PATH="$VENV/include"

KT_INSTALL_ARGS=()
CPU_FLAGS="$(awk -F: '/^flags[[:space:]]*:/{print $2; exit}' /proc/cpuinfo)"
if [[ " $CPU_FLAGS " != *" avx512_vnni "* ]]; then
  # The preflight above has already required explicit RUNGLM_ALLOW_AVX2=1 and
  # verified AVX2+FMA. Build against the AVX2 baseline rather than -march=native
  # so the result remains usable across comparable AVX2 hosts.
  export CPUINFER_CPU_INSTRUCT=AVX2
  export CPUINFER_ENABLE_AMX=OFF
  KT_INSTALL_ARGS=(--manual)
fi
CPUINFER_USE_CUDA=1 "$REPO/ktransformers/install.sh" all "${KT_INSTALL_ARGS[@]}"
python -m pip install -r "$REPO/requirements-lock.txt"
"$REPO/scripts/apply_runtime_overlays.sh" "$OVERLAY_BACKUP"

ACTIVATE="$VENV/bin/activate"
grep -q 'RUNGLM venv libraries' "$ACTIVATE" || {
  printf '\n# RUNGLM venv libraries\nexport LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"\n' >> "$ACTIVATE"
}
kt doctor

# --- rejoin the background download -----------------------------------------
# `wait` is guarded because set -e would otherwise abort here on a failed
# download and lose the "the build succeeded" half of the result, which is the
# expensive half and is worth reporting even when the fetch needs a retry.
if [ -n "${DL_PID:-}" ]; then
  echo "[setup] build finished; waiting for the weights download (pid $DL_PID)"
  DL_RC=0
  wait "$DL_PID" || DL_RC=$?
  if [ "$DL_RC" != "0" ]; then
    echo "Build OK, but the weights download failed (exit $DL_RC)." >&2
    echo "See $REPO/logs/download_weights.log; rerun ./scripts/download_weights.sh to resume." >&2
    exit "$DL_RC"
  fi
  echo "Setup complete (build + weights). Next: ./run_adaptive.sh"
else
  echo "Setup complete. Next: ./scripts/download_weights.sh, then ./run_adaptive.sh"
fi
