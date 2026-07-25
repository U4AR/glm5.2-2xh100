#!/usr/bin/env bash
# Clone-to-build bootstrap. Run this on the final GPU pod.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/config.sh"

PYTHON="${PYTHON:-python3.12}"
KT_REPO="${KT_REPO:-https://github.com/U4AR/ktransformers.git}"
KT_BRANCH="${KT_BRANCH:-glm5.2-2xh100-stable}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"

command -v "$PYTHON" >/dev/null || {
  echo "Python 3.12 is required (or set PYTHON=/path/to/python3.12)." >&2
  exit 1
}
command -v nvcc >/dev/null || {
  echo "CUDA toolkit/nvcc is required to build for this machine." >&2
  exit 1
}
python3 "$REPO/scripts/hardware_profile.py" --check --weights-dir "$W4AFP8_MODEL"

if [ "$INSTALL_SYSTEM_DEPS" = "1" ]; then
  sudo apt-get update
  sudo apt-get install -y git git-lfs pkg-config libhwloc-dev libnuma-dev build-essential
fi

if [ ! -d "$REPO/ktransformers/.git" ]; then
  git clone --recursive --branch "$KT_BRANCH" "$KT_REPO" "$REPO/ktransformers"
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
# A fresh clone contains the original machine's extension only as a patch
# artifact. Remove it explicitly before compiling for the current CPU/GPU.
find "$VENV/lib/python3.12/site-packages/kt_kernel" -maxdepth 1 \
  -type f -name 'kt_kernel_ext*.so' -delete 2>/dev/null || true
source "$VENV/bin/activate"
python -m pip install --upgrade pip
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$VENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$VENV"
export CMAKE_LIBRARY_PATH="$VENV/lib"
export CMAKE_INCLUDE_PATH="$VENV/include"

CPUINFER_USE_CUDA=1 "$REPO/ktransformers/install.sh"
python -m pip install -r "$REPO/requirements-lock.txt"
"$REPO/scripts/apply_runtime_overlays.sh" "$OVERLAY_BACKUP"

ACTIVATE="$VENV/bin/activate"
grep -q 'RUNGLM venv libraries' "$ACTIVATE" || {
  printf '\\n# RUNGLM venv libraries\\nexport LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:${LD_LIBRARY_PATH:-}"\\n' >> "$ACTIVATE"
}
kt doctor
echo "Setup complete. Next: python int4_scripts/download_w4afp8.py"
